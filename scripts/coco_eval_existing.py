import argparse
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(description="Evaluate an existing COCO detection JSON on a subset.")
    parser.add_argument("--annotations", default="/home/datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--detections", required=True)
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--summary", default=None)
    args = parser.parse_args()

    coco_gt = COCO(args.annotations)
    image_ids = list(sorted(coco_gt.imgs.keys()))[args.start_index :]
    if args.num_images > 0:
        image_ids = image_ids[: args.num_images]

    coco_dt = coco_gt.loadRes(args.detections)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    summary = {
        "annotations": args.annotations,
        "detections": args.detections,
        "start_index": args.start_index,
        "images": len(image_ids),
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
    print("Summary JSON:")
    print(json.dumps(summary, indent=2))
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"summary_json: {args.summary}")


if __name__ == "__main__":
    main()
