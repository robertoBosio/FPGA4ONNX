# Evaluation Artifact

This artifact reproduces the paper's evaluation. It contains deployable models
and evaluation scripts, but not the compiler flow or measured results. Commands
below write outputs locally; do not add generated outputs to the artifact.

## Contents

```text
ours/<model>/
  accelerated_<model>.onnx   Rewritten ONNX model with the FPGA package
  original_model_qcdq.onnx   CPU reference model
  libcustomop.so             Custom ONNX Runtime operator for the target board
  throughput_test.py         Raw-accelerator throughput benchmark

scripts/
  onnx_inference_coco.py     Correctness, throughput, and power measurements
  coco_map_ort_ultralytics.py
  coco_eval_existing.py

baselines/vitis-ai/
  Vitis-AI-compatible ONNX models, runner, and power wrapper

baselines/jetson-nano/
  TensorRT benchmark script and the three benchmarked ONNX models
```

## Dataset

Download COCO `val2017` images and annotations separately. Set these variables
on the ZCU102 before running the COCO-based commands:

```bash
export COCO_IMAGES=/path/to/coco/images/val2017
export COCO_ANNOTATIONS=/path/to/coco/annotations/instances_val2017.json
export ARTIFACTS=/path/to/artifacts
```

## ZCU102: Proposed Framework

Run these commands in the board image that provides PYNQ, XRT, and the ONNX
Runtime build compatible with `libcustomop.so`. The custom operator
programs the FPGA from the package embedded in the rewritten ONNX model.

For each model, set `MODEL` to `yolov5nu`, `yolov8n`, or `yolov10n` before
running a command:

```bash
export MODEL=yolov5nu
```

### Correctness

```bash
cd "$ARTIFACTS/ours/$MODEL"
python3 "$ARTIFACTS/scripts/onnx_inference_coco.py" \
  --mode correctness \
  --model "accelerated_${MODEL}.onnx" \
  --original-model original_model_qcdq.onnx \
  --custom-op libcustomop.so \
  --images "$COCO_IMAGES" \
  --annotations "$COCO_ANNOTATIONS" \
  --num-images 10 --num-workers 0 --inflight-runs 1 \
  --atol 12.0 --rtol 0.10
```

### Raw Accelerator Throughput

```bash
cd "$ARTIFACTS/ours/$MODEL"
python3 throughput_test.py
```

### Complete ONNX Throughput

The selected in-flight depths are 8 for YOLOv5nu, 5 for YOLOv8n, and 8 for
YOLOv10n. Set `N` to the selected value or to a sweep point:

```bash
export N=8  # YOLOv5nu; use 5 for YOLOv8n and 8 for YOLOv10n
```

```bash
cd "$ARTIFACTS/ours/$MODEL"
python3 "$ARTIFACTS/scripts/onnx_inference_coco.py" \
  --mode speed \
  --model "accelerated_${MODEL}.onnx" \
  --original-model original_model_qcdq.onnx \
  --custom-op libcustomop.so \
  --images "$COCO_IMAGES" \
  --annotations "$COCO_ANNOTATIONS" \
  --num-images 100 --num-workers 0 --warmup-batches 5 \
  --inflight-runs "$N" --skip-original \
  --results-file "performance_inflight${N}.txt"
```

### Power and Energy

Use the selected concurrency for the model and the same seven ZCU102 rails as
the Vitis AI comparison:

```bash
cd "$ARTIFACTS/ours/$MODEL"
python3 "$ARTIFACTS/scripts/onnx_inference_coco.py" \
  --mode speed \
  --model "accelerated_${MODEL}.onnx" \
  --original-model original_model_qcdq.onnx \
  --custom-op libcustomop.so \
  --images "$COCO_IMAGES" \
  --annotations "$COCO_ANNOTATIONS" \
  --num-images 100 --num-workers 0 --warmup-batches 5 \
  --inflight-runs "$N" --skip-original --power-record --power-sample-period 0.02 \
  --power-rails VCCINT,VCCBRAM,VCCAUX,VCCINTLP,VCCPSINTFP,VCCPSDDR,VCCPSAUX \
  --results-file performance_power.txt --power-file power_samples.csv \
  --runs-file inference_runs.json
```

### COCO Accuracy

YOLOv5nu and YOLOv8n use the default Ultralytics NMS postprocessing:

```bash
cd "$ARTIFACTS/ours/$MODEL"
python3 "$ARTIFACTS/scripts/coco_map_ort_ultralytics.py" \
  --model "accelerated_${MODEL}.onnx" \
  --custom-op libcustomop.so \
  --images "$COCO_IMAGES" --annotations "$COCO_ANNOTATIONS" \
  --num-images 5000 --inflight-runs 1 \
  --output detections.json --summary coco_summary.json
```

For YOLOv10n, append `--postprocess yolov10` to the command above.

## ZCU102: Vitis AI

The reported baseline uses Vitis AI 3.0, ONNX Runtime 1.14, VAIP 1.0, and the
`DPUCZDX8G_ISA1_B4096` target. Build the runner in an environment providing
the Vitis AI ONNX Runtime headers and libraries:

```bash
cd "$ARTIFACTS/baselines/vitis-ai"
g++ -std=c++17 -O2 run_onnx_vitisai_random.cpp -o run_onnx_vitisai_random \
  -I/usr/include -I/usr/include/onnxruntime -lonnxruntime -pthread
```

Run the four-worker, 200-iteration random-input benchmark for each supplied
model. For example:

```bash
python3 vitisai_power_benchmark.py \
  --samples power_samples.csv --summary power_summary.json --stdout benchmark.log \
  -- ./run_onnx_vitisai_random yolov5nu_leaky_xint8_vai.onnx - /dev/null 200 4
```

Repeat with `yolov8n_leaky_xint8_vai.onnx` and
`yolov10n_leaky_xint8_vai.onnx`.

## Jetson Nano: TensorRT

Use the Jetson Nano TensorRT environment with TensorRT 7.1.3, PyCUDA, and
`tegrastats`. TensorRT engines are platform-specific and must be built locally.
The provided YOLOv10n ONNX model is the TensorRT-compatible rewritten model
used for the benchmark.

```bash
export JETSON_ARTIFACTS=/path/to/artifacts/baselines/jetson-nano
mkdir -p "$JETSON_ARTIFACTS/engines"

trtexec --onnx="$JETSON_ARTIFACTS/yolov5nu_opset11.onnx" \
  --saveEngine="$JETSON_ARTIFACTS/engines/yolov5nu_b1.trt" \
  --fp16 --explicitBatch --workspace=1024

trtexec --onnx="$JETSON_ARTIFACTS/yolov8n_opset11.onnx" \
  --saveEngine="$JETSON_ARTIFACTS/engines/yolov8n_b1.trt" \
  --fp16 --explicitBatch --workspace=1024

trtexec --onnx="$JETSON_ARTIFACTS/yolov10n_opset11.onnx" \
  --saveEngine="$JETSON_ARTIFACTS/engines/yolov10n_b1.trt" \
  --fp16 --explicitBatch --workspace=1024
```

Run the single-stream power benchmark with ten warmups, 100 timed inferences,
and 20 ms power sampling:

```bash
python3 "$JETSON_ARTIFACTS/benchmark_yolo_trt_power.py" \
  --engine-dir "$JETSON_ARTIFACTS/engines" \
  --out-dir "$JETSON_ARTIFACTS/results" \
  --warmup 10 --runs 100 --streams 1 --interval-ms 20 --seed 0
```

## Custom-Node Metadata and Runtime Interface

Each rewritten `accelerated_<model>.onnx` model contains an FPGA-backed custom
node. Its inputs and outputs are the dynamic boundary tensors of the accelerated
region. The node also carries an `accelerator_package`
attribute that describes the generated deployment:

```text
custom_node(
  inputs  = [dynamic input tensors],
  outputs = [dynamic output tensors],
  attributes = {
    accelerator_package: {
      fpga_image:              encoded bitstream,
      hardware_description:    encoded hardware handoff,
      input_ports:             names, formats, and static values,
      output_ports:            names and formats,
      internal_buffers:        accelerator memory metadata,
      target_configuration:    generated build and board information
    }
  }
)
```

The package is serialized as JSON and embedded in the ONNX node. Binary assets,
including the bitstream and static tensors such as weights, are encoded so they
can be stored in the model. Dynamic input tensors are not embedded: ONNX Runtime
supplies them whenever it invokes the node.

The custom-operator library consumes this metadata through the normal ONNX
Runtime custom-operator lifecycle:

```text
initialize(node attributes):
  read and decode the accelerator package
  program the FPGA image
  configure the accelerator control and data interfaces
  allocate runtime buffers and load static tensors

execute(dynamic input tensors):
  validate the boundary tensor interface
  transfer inputs to the accelerator
  wait for accelerator outputs
  transfer outputs into ONNX Runtime tensors
```

Initialization is performed once and the configured accelerator is reused by
subsequent node invocations. The application only registers the custom-operator
library and runs the rewritten model through the standard ONNX Runtime API.
