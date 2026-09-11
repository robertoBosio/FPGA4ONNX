import argparse
import csv
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from statistics import median


RAILNAME_MAP = {
    "VCCPSINTFP": "u76",
    "VCCINTLP": "u77",
    "VCCPSAUX": "u78",
    "VCCPSPLL": "u87",
    "MGTRAVCC": "u85",
    "MGTRAVTT": "u86",
    "VCCPSDDR": "u93",
    "VCCOPS": "u88",
    "VCCOPS3": "u15",
    "VCCPSDDRPLL": "u92",
    "VCCINT": "u79",
    "VCCBRAM": "u81",
    "VCCAUX": "u80",
    "VCC1V2": "u84",
    "VCC3V3": "u16",
    "VADJ_FMC": "u65",
    "MGTAVCC": "u74",
    "MGTAVTT": "u75",
}


@dataclass
class Rail:
    name: str
    power_path: str


def read_float(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        return float(f.read().strip())


def discover_rails(selected: set[str]) -> list[Rail]:
    rails: list[Rail] = []
    for entry in os.listdir("/sys/class/hwmon"):
        base = os.path.join("/sys/class/hwmon", entry)
        name_path = os.path.join(base, "name")
        if not os.path.isfile(name_path):
            continue
        try:
            with open(name_path, "r", encoding="utf-8") as f:
                sensor_name = f.read().strip()
        except OSError:
            continue
        if not sensor_name.startswith("ina"):
            continue
        matched = None
        for rail_name, chip in RAILNAME_MAP.items():
            if chip in sensor_name:
                matched = rail_name
                break
        if matched is None or matched not in selected:
            continue
        power_path = os.path.join(base, "power1_input")
        if os.path.isfile(power_path):
            rails.append(Rail(matched, power_path))
    rails.sort(key=lambda r: r.name)
    missing = sorted(selected - {r.name for r in rails})
    if missing:
        raise RuntimeError(f"Missing selected rails: {missing}")
    return rails


def sample_once(rails: list[Rail]) -> dict[str, float]:
    values = {}
    for rail in rails:
        # hwmon INA226 power1_input is exposed in microwatts.
        values[rail.name] = read_float(rail.power_path) / 1_000_000.0
    values["total_selected_w"] = sum(values.values())
    return values


def parse_benchmark(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    m = re.search(r"LATENCY ms: avg ([0-9.]+) median ([0-9.]+) p90 ([0-9.]+) min ([0-9.]+) max ([0-9.]+)", stdout)
    if m:
        out.update(
            {
                "latency_avg_ms": float(m.group(1)),
                "latency_median_ms": float(m.group(2)),
                "latency_p90_ms": float(m.group(3)),
                "latency_min_ms": float(m.group(4)),
                "latency_max_ms": float(m.group(5)),
            }
        )
    m = re.search(r"WALL ms: ([0-9.]+) total_inferences ([0-9]+)", stdout)
    if m:
        out["wall_ms"] = float(m.group(1))
        out["total_inferences"] = int(m.group(2))
    m = re.search(r"THROUGHPUT inf/s: ([0-9.]+)", stdout)
    if m:
        out["throughput_inf_s"] = float(m.group(1))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rails", default="VCCINT,VCCBRAM,VCCAUX,VCCINTLP,VCCPSINTFP,VCCPSDDR,VCCPSAUX")
    parser.add_argument("--sample-period", type=float, default=0.02)
    parser.add_argument("--samples", default="vitisai_power_samples.csv")
    parser.add_argument("--summary", default="vitisai_power_summary.json")
    parser.add_argument("--stdout", default="vitisai_power_benchmark_stdout.log")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.command:
        raise SystemExit("missing command after --")
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command after --")

    selected = {x.strip() for x in args.rails.split(",") if x.strip()}
    rails = discover_rails(selected)
    rail_names = [r.name for r in rails]

    env = os.environ.copy()
    env["LD_PRELOAD"] = "/usr/lib/libvaip_ort.so"

    start_perf_ns = time.perf_counter_ns()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    samples: list[dict[str, float]] = []
    try:
        while proc.poll() is None:
            t_ns = time.perf_counter_ns()
            row = sample_once(rails)
            row["timestamp_perf_ns"] = t_ns
            row["elapsed_s"] = (t_ns - start_perf_ns) / 1e9
            samples.append(row)
            time.sleep(args.sample_period)
        stdout, _ = proc.communicate(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()

    end_perf_ns = time.perf_counter_ns()
    with open(args.stdout, "w", encoding="utf-8") as f:
        f.write(stdout or "")

    with open(args.samples, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp_perf_ns", "elapsed_s", *rail_names, "total_selected_w"])
        writer.writeheader()
        for row in samples:
            writer.writerow(row)

    totals = [row["total_selected_w"] for row in samples]
    parsed = parse_benchmark(stdout or "")
    mean_power = sum(totals) / len(totals) if totals else float("nan")
    summary = {
        "command": command,
        "returncode": proc.returncode,
        "rails": rail_names,
        "sample_period_s": args.sample_period,
        "num_power_samples": len(samples),
        "measured_wall_s_wrapper": (end_perf_ns - start_perf_ns) / 1e9,
        "mean_selected_power_w": mean_power,
        "median_selected_power_w": median(totals) if totals else float("nan"),
        "min_selected_power_w": min(totals) if totals else float("nan"),
        "max_selected_power_w": max(totals) if totals else float("nan"),
        "rail_mean_w": {name: (sum(row[name] for row in samples) / len(samples) if samples else float("nan")) for name in rail_names},
        **parsed,
    }
    if "throughput_inf_s" in summary and summary["throughput_inf_s"] > 0:
        summary["energy_selected_j_per_inf"] = mean_power / summary["throughput_inf_s"]

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
