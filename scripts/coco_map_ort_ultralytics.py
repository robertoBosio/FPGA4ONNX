import argparse
import inspect
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


COCO80_TO_COCO91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def make_session(model_path: str, custom_op: str | None):
    so = ort.SessionOptions()
    if custom_op:
        so.register_custom_ops_library(os.path.abspath(custom_op))
    return ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])


def nms_with_compatible_signature(prediction, conf: float, iou: float, max_det: int, nc: int):
    kwargs = {
        "prediction": prediction,
        "conf_thres": conf,
        "iou_thres": iou,
        "classes": None,
        "agnostic": False,
        "multi_label": False,
        "max_det": max_det,
        "nc": nc,
    }
    sig = inspect.signature(non_max_suppression)
    return non_max_suppression(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def preprocess_image(path: str, imgsz: int, stride: int, input_type: str):
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=stride)
    resized = letterbox(image=image)
    array = resized[..., ::-1].transpose((2, 0, 1))
    array = np.ascontiguousarray(array)[None]

    if "float16" in input_type:
        array = array.astype(np.float16) / np.float16(255.0)
    else:
        array = array.astype(np.float32) / np.float32(255.0)

    return image, array


def run_one(session, input_name: str, input_type: str, image_root: str, img: dict, imgsz: int, stride: int):
    path = os.path.join(image_root, img["file_name"])
    orig_img, input_array = preprocess_image(path, imgsz, stride, input_type)
    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: input_array})
    t1 = time.perf_counter()
    return img, orig_img, input_array.shape, outputs, t1 - t0


def postprocess_nms(
    img: dict,
    orig_img,
    input_shape,
    outputs,
    conf: float,
    iou: float,
    max_det: int,
    nc: int,
):
    preds = [torch.from_numpy(output).float() for output in outputs]
    preds_arg = preds[0] if len(preds) == 1 else preds
    detections = nms_with_compatible_signature(preds_arg, conf, iou, max_det, nc)

    result = []
    pred = detections[0]
    if len(pred):
        pred[:, :4] = ops.scale_boxes(input_shape[2:], pred[:, :4], orig_img.shape)
        pred[:, :4] = pred[:, :4].clamp(min=0)
        for *xyxy, score, cls in pred[:, :6].tolist():
            cls_idx = int(cls)
            if cls_idx < 0 or cls_idx >= len(COCO80_TO_COCO91):
                continue
            x1, y1, x2, y2 = xyxy
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            result.append(
                {
                    "image_id": int(img["id"]),
                    "category_id": int(COCO80_TO_COCO91[cls_idx]),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score),
                }
            )
    return result


def postprocess_yolov10(img: dict, orig_img, input_shape, outputs, conf: float):
    out = np.asarray(outputs[0])
    if out.ndim == 3:
        out = out[0]

    result = []
    boxes = torch.from_numpy(out[:, :4].astype(np.float32))
    if len(boxes):
        boxes = ops.scale_boxes(input_shape[2:], boxes, orig_img.shape).clamp(min=0)

    for box, row in zip(boxes.tolist(), out.tolist()):
        score = float(row[4])
        if score < conf:
            continue
        cls_idx = int(row[5])
        if cls_idx < 0 or cls_idx >= len(COCO80_TO_COCO91):
            continue
        x1, y1, x2, y2 = box
        result.append(
            {
                "image_id": int(img["id"]),
                "category_id": int(COCO80_TO_COCO91[cls_idx]),
                "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
                "score": score,
            }
        )
    return result


def evaluate_coco(annotations: str, detections_path: str, image_ids: list[int]):
    coco_gt = COCO(annotations)
    if os.path.getsize(detections_path) == 0:
        raise RuntimeError(f"Empty detection file: {detections_path}")
    coco_dt = coco_gt.loadRes(detections_path)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "AP_50_95": float(coco_eval.stats[0]),
        "AP_50": float(coco_eval.stats[1]),
        "AP_75": float(coco_eval.stats[2]),
        "AP_small": float(coco_eval.stats[3]),
        "AP_medium": float(coco_eval.stats[4]),
        "AP_large": float(coco_eval.stats[5]),
        "AR_1": float(coco_eval.stats[6]),
        "AR_10": float(coco_eval.stats[7]),
        "AR_100": float(coco_eval.stats[8]),
        "AR_small": float(coco_eval.stats[9]),
        "AR_medium": float(coco_eval.stats[10]),
        "AR_large": float(coco_eval.stats[11]),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="COCO mAP for ORT custom-op YOLO models using Ultralytics postprocess.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--custom-op", default=None)
    parser.add_argument("--images", default="/home/datasets/coco/images/val2017")
    parser.add_argument("--annotations", default="/home/datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--inflight-runs", type=int, default=1)
    parser.add_argument("--postprocess", choices=("ultralytics-nms", "yolov10"), default="ultralytics-nms")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    session = make_session(args.model, args.custom_op)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type

    coco = COCO(args.annotations)
    all_image_ids = list(sorted(coco.imgs.keys()))
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.start_index >= len(all_image_ids):
        raise ValueError(f"--start-index {args.start_index} is beyond dataset length {len(all_image_ids)}")
    image_ids = all_image_ids[args.start_index :]
    if args.num_images > 0:
        image_ids = image_ids[: args.num_images]
    images = coco.loadImgs(image_ids)

    print(f"model: {args.model}")
    print(f"custom_op: {args.custom_op}")
    print(f"onnxruntime: {ort.__version__}")
    print(f"input: {input_name} {input_info.shape} {input_type}")
    print(f"start_index: {args.start_index}")
    print(f"images: {len(images)}")
    print(f"conf: {args.conf} iou: {args.iou} max_det: {args.max_det}")
    print(f"inflight_runs: {args.inflight_runs}")
    print(f"postprocess: {args.postprocess}")

    detections = []
    run_times = []
    t_start = time.perf_counter()
    processed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.inflight_runs)) as executor:
        future_to_idx = {
            executor.submit(
                run_one,
                session,
                input_name,
                input_type,
                args.images,
                img,
                args.imgsz,
                args.stride,
            ): idx
            for idx, img in enumerate(images)
        }
        for future in as_completed(future_to_idx):
            img, orig_img, input_shape, outputs, run_time = future.result()
            if args.postprocess == "yolov10":
                detections.extend(postprocess_yolov10(img, orig_img, input_shape, outputs, args.conf))
            else:
                detections.extend(
                    postprocess_nms(
                        img,
                        orig_img,
                        input_shape,
                        outputs,
                        args.conf,
                        args.iou,
                        args.max_det,
                        args.nc,
                    )
                )
            run_times.append(run_time)
            processed += 1
            if processed % 50 == 0 or processed == len(images):
                elapsed = time.perf_counter() - t_start
                print(f"processed {processed}/{len(images)} images, elapsed={elapsed:.1f}s, overall_fps={processed / elapsed:.2f}")

    t_end = time.perf_counter()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(detections, f)

    print(f"detections: {len(detections)}")
    print(f"detections_json: {args.output}")
    stats = None
    if args.skip_eval:
        print("Skipping COCOeval.")
    else:
        print("Running COCOeval...")
        stats = evaluate_coco(args.annotations, args.output, image_ids)

    run_arr = np.asarray(run_times, dtype=np.float64)
    summary = {
        "model": args.model,
        "custom_op": args.custom_op,
        "postprocess": args.postprocess,
        "start_index": args.start_index,
        "images": len(images),
        "detections": len(detections),
        "total_time_s": float(t_end - t_start),
        "overall_fps": float(len(images) / (t_end - t_start)),
        "session_run_mean_ms": float(run_arr.mean() * 1000.0),
        "session_run_p50_ms": float(np.percentile(run_arr, 50) * 1000.0),
        "session_run_p90_ms": float(np.percentile(run_arr, 90) * 1000.0),
        "session_run_p95_ms": float(np.percentile(run_arr, 95) * 1000.0),
        "coco": stats,
    }

    print("Summary JSON:")
    print(json.dumps(summary, indent=2))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"summary_json: {args.summary}")


if __name__ == "__main__":
    main()
