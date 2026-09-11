#!/usr/bin/env python3
"""Benchmark TensorRT YOLO engines on Jetson with tegrastats power logging."""

import argparse
import csv
import json
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pycuda.autoinit  # noqa: F401 initializes CUDA context
import pycuda.driver as cuda
import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

POM_5V_IN_RE = re.compile(r"POM_5V_IN\s+(\d+)\/(\d+)")
POM_5V_GPU_RE = re.compile(r"POM_5V_GPU\s+(\d+)\/(\d+)")
POM_5V_CPU_RE = re.compile(r"POM_5V_CPU\s+(\d+)\/(\d+)")
GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize engine: {engine_path}")
    return engine


class TRTSession:
    def __init__(self, engine):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.host_inputs = []
        self.device_inputs = []
        self.host_outputs = []
        self.device_outputs = []
        self.input_shapes = []
        self.output_shapes = []

        for binding in engine:
            shape = tuple(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
                self.input_shapes.append(shape)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)
                self.output_shapes.append(shape)

        if len(self.input_shapes) != 1:
            raise RuntimeError(f"expected one input, got {len(self.input_shapes)}")

    @property
    def input_shape(self):
        return self.input_shapes[0]

    @property
    def batch_size(self):
        return self.input_shapes[0][0]

    def set_input(self, x):
        np.copyto(self.host_inputs[0], x.ravel())

    def submit(self):
        for host_mem, device_mem in zip(self.host_inputs, self.device_inputs):
            cuda.memcpy_htod_async(device_mem, host_mem, self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for host_mem, device_mem in zip(self.host_outputs, self.device_outputs):
            cuda.memcpy_dtoh_async(host_mem, device_mem, self.stream)

    def infer(self):
        self.submit()
        self.stream.synchronize()


def find_tegrastats():
    for candidate in (shutil.which("tegrastats"), "/usr/bin/tegrastats"):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def tegrastats_reader_thread(proc, log_file):
    for line in proc.stdout:
        log_file.write(f"{time.perf_counter():.9f}\t{line}")
    log_file.flush()


def start_tegrastats(log_path, interval_ms):
    tegra_bin = find_tegrastats()
    if tegra_bin is None:
        return None, None, None
    stdbuf_bin = shutil.which("stdbuf")
    cmd = [tegra_bin, "--interval", str(interval_ms)]
    if stdbuf_bin:
        cmd = [stdbuf_bin, "-oL"] + cmd
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    reader = threading.Thread(target=tegrastats_reader_thread, args=(proc, log_file), daemon=True)
    reader.start()
    return proc, log_file, reader


def stop_tegrastats(proc, log_file, reader):
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if reader is not None:
        reader.join(timeout=2)
    if log_file is not None:
        log_file.close()


def extract_pair(regex, text):
    match = regex.search(text)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


def extract_single(regex, text):
    match = regex.search(text)
    return int(match.group(1)) if match else None


def parse_tegrastats(log_path, t_start=None, t_end=None):
    rows = []
    with open(log_path) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if "\t" not in raw:
                continue
            ts_str, line = raw.split("\t", 1)
            try:
                ts = float(ts_str)
            except ValueError:
                continue
            if t_start is not None and ts < t_start:
                continue
            if t_end is not None and ts > t_end:
                continue
            pom_in_inst, pom_in_avg = extract_pair(POM_5V_IN_RE, line)
            pom_gpu_inst, pom_gpu_avg = extract_pair(POM_5V_GPU_RE, line)
            pom_cpu_inst, pom_cpu_avg = extract_pair(POM_5V_CPU_RE, line)
            rows.append({
                "ts": ts,
                "pom_in_inst_mw": pom_in_inst,
                "pom_in_avg_mw": pom_in_avg,
                "pom_gpu_inst_mw": pom_gpu_inst,
                "pom_gpu_avg_mw": pom_gpu_avg,
                "pom_cpu_inst_mw": pom_cpu_inst,
                "pom_cpu_avg_mw": pom_cpu_avg,
                "gr3d_pct": extract_single(GR3D_RE, line),
            })
    return rows


def summarize_tegrastats_rows(rows):
    if not rows:
        return {"samples": 0}

    def values(key):
        return np.array([r[key] for r in rows if r[key] is not None], dtype=np.float64)

    out = {"samples": len(rows), "t_first": rows[0]["ts"], "t_last": rows[-1]["ts"]}
    for prefix, key in (("power_in", "pom_in_inst_mw"), ("power_gpu", "pom_gpu_inst_mw"), ("power_cpu", "pom_cpu_inst_mw")):
        arr = values(key)
        if len(arr):
            out[f"{prefix}_mean_mw"] = float(arr.mean())
            out[f"{prefix}_std_mw"] = float(arr.std())
            out[f"{prefix}_p50_mw"] = float(np.percentile(arr, 50))
            out[f"{prefix}_p90_mw"] = float(np.percentile(arr, 90))
            out[f"{prefix}_p99_mw"] = float(np.percentile(arr, 99))
    gr3d = values("gr3d_pct")
    if len(gr3d):
        out["gr3d_mean_pct"] = float(gr3d.mean())
        out["gr3d_p50_pct"] = float(np.percentile(gr3d, 50))
        out["gr3d_p90_pct"] = float(np.percentile(gr3d, 90))
        out["gr3d_p99_pct"] = float(np.percentile(gr3d, 99))
    return out


def benchmark_model(name, engine_path, out_dir, warmup, runs, interval_ms, seed, streams):
    print(f"\n=== {name} ===", flush=True)
    engine = load_engine(engine_path)
    sessions = [TRTSession(engine) for _ in range(streams)]
    session = sessions[0]
    rng = np.random.RandomState(seed)
    for stream_idx, stream_session in enumerate(sessions):
        x = rng.rand(*stream_session.input_shape).astype(np.float32)
        stream_session.set_input(x)

    for _ in range(warmup):
        for stream_session in sessions:
            stream_session.submit()
        for stream_session in sessions:
            stream_session.stream.synchronize()

    tegra_log = out_dir / f"tegrastats_{name}.log"
    tegra_proc, tegra_file, tegra_reader = start_tegrastats(tegra_log, interval_ms)
    times = []
    benchmark_t0 = None
    benchmark_t1 = None
    try:
        if tegra_proc is not None:
            time.sleep(0.2)
        benchmark_t0 = time.perf_counter()
        completed = 0
        while completed < runs:
            active_sessions = sessions[: min(streams, runs - completed)]
            submitted = []
            for stream_session in active_sessions:
                t0 = time.perf_counter()
                stream_session.submit()
                submitted.append((stream_session, t0))
            for stream_session, t0 in submitted:
                stream_session.stream.synchronize()
                times.append(time.perf_counter() - t0)
            completed += len(active_sessions)
        benchmark_t1 = time.perf_counter()
    finally:
        stop_tegrastats(tegra_proc, tegra_file, tegra_reader)

    batch_times = np.array(times, dtype=np.float64)
    image_times = batch_times / session.batch_size
    power_rows = parse_tegrastats(tegra_log, t_start=benchmark_t0, t_end=benchmark_t1)
    power = summarize_tegrastats_rows(power_rows)
    power_in_w = power.get("power_in_mean_mw")
    gpu_w = power.get("power_gpu_mean_mw")

    result = {
        "model": name,
        "engine_path": str(engine_path),
        "trt_version": trt.__version__,
        "input_shape": list(session.input_shape),
        "output_shapes": [list(s) for s in session.output_shapes],
        "batch_size": int(session.batch_size),
        "inference_streams": streams,
        "warmup": warmup,
        "runs": runs,
        "latency_mean_ms": float(image_times.mean() * 1000),
        "latency_std_ms": float(image_times.std() * 1000),
        "latency_p50_ms": float(np.percentile(image_times, 50) * 1000),
        "latency_p90_ms": float(np.percentile(image_times, 90) * 1000),
        "latency_p99_ms": float(np.percentile(image_times, 99) * 1000),
        "throughput_fps": float(runs * session.batch_size / (benchmark_t1 - benchmark_t0)),
        "power": power,
        "board_power_mean_w": None if power_in_w is None else float(power_in_w / 1000.0),
        "gpu_power_mean_w": None if gpu_w is None else float(gpu_w / 1000.0),
        "energy_per_image_j": None if power_in_w is None else float(
            (power_in_w / 1000.0) * (benchmark_t1 - benchmark_t0) / (runs * session.batch_size)
        ),
        "tegrastats_log": str(tegra_log),
    }
    print(json.dumps({k: result[k] for k in ("model", "latency_mean_ms", "throughput_fps", "board_power_mean_w", "gpu_power_mean_w", "energy_per_image_j")}, indent=2), flush=True)
    return result


def write_csv(path, results):
    fields = [
        "model", "latency_mean_ms", "latency_std_ms", "latency_p50_ms", "latency_p90_ms", "latency_p99_ms",
        "throughput_fps", "board_power_mean_w", "gpu_power_mean_w", "energy_per_image_j",
        "batch_size", "inference_streams", "runs", "engine_path", "tegrastats_log",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field) for field in fields})


def main():
    parser = argparse.ArgumentParser()
    artifact_dir = Path(__file__).resolve().parent
    parser.add_argument("--engine-dir", type=Path, default=artifact_dir)
    parser.add_argument("--out-dir", type=Path, default=artifact_dir / "results")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--streams", type=int, default=1, help="Independent TensorRT contexts/CUDA streams to run concurrently.")
    parser.add_argument("--interval-ms", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.streams <= 0:
        parser.error("--streams must be > 0")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    engines = [
        ("yolov5nu", args.engine_dir / "yolov5nu_b1.trt"),
        ("yolov8n", args.engine_dir / "yolov8n_b1.trt"),
        ("yolov10n", args.engine_dir / "yolov10n_b1.trt"),
    ]
    missing = [str(path) for _, path in engines if not path.exists()]
    if missing:
        raise SystemExit(f"missing engines: {missing}")

    results = [
        benchmark_model(name, path, out_dir, args.warmup, args.runs, args.interval_ms, args.seed, args.streams)
        for name, path in engines
    ]
    summary_json = out_dir / "yolo_trt_power_summary.json"
    summary_csv = out_dir / "yolo_trt_power_summary.csv"
    with open(summary_json, "w") as f:
        json.dump(results, f, indent=2)
    write_csv(summary_csv, results)
    print(f"\nSaved {summary_json}")
    print(f"Saved {summary_csv}")


if __name__ == "__main__":
    main()
