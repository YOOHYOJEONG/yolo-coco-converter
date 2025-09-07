import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import cv2


def load_coco(coco_json: Path):
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        if "image_id" in ann:
            anns_by_image[ann["image_id"]].append(ann)
    cats = {c["id"]: c for c in data.get("categories", [])}
    
    return images, anns_by_image, cats


def load_classes_txt(classes_txt: Optional[Path]) -> Optional[List[str]]:
    if classes_txt and classes_txt.exists():
        return [ln.strip() for ln in classes_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    
    return None


def build_catid_to_yoloid(cats: Dict[int, dict], classes_txt: Optional[List[str]]) -> Dict[int, int]:
    if classes_txt:
        name_to_yolo = {name: i for i, name in enumerate(classes_txt)}
        mapping = {}
        for cid, c in cats.items():
            name = c.get("name", f"class_{cid}")
            if name in name_to_yolo:
                mapping[cid] = name_to_yolo[name]
        return mapping
    # classes.txt 없으면 COCO category id 정렬 순서로 0..K-1
    sorted_ids = sorted(cats.keys())
    
    return {cid: i for i, cid in enumerate(sorted_ids)}


def ensure_image_size(im_meta: dict, images_dir: Optional[Path]) -> Tuple[int, int]:
    W = im_meta.get("width", None)
    H = im_meta.get("height", None)
    if W and H:
        return int(W), int(H)
    if not images_dir:
        raise ValueError(f"Image size missing for {im_meta.get('file_name')}; provide --images_dir to infer.")
    img_path = images_dir / im_meta["file_name"]
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image to infer size: {img_path}")
    h, w = img.shape[:2]
    
    return int(w), int(h)


def clamp_bbox_xywh(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, x)
    y1 = max(0.0, y)
    x2 = min(float(W), x + w)
    y2 = min(float(H), y + h)
    w2 = max(0.0, x2 - x1)
    h2 = max(0.0, y2 - y1)
    
    return x1, y1, w2, h2


def coco_box_to_yolo(x: float, y: float, w: float, h: float, W: int, H: int) -> Tuple[float, float, float, float]:
    xc = (x + w / 2.0) / W if W > 0 else 0.0
    yc = (y + h / 2.0) / H if H > 0 else 0.0
    wn = w / W if W > 0 else 0.0
    hn = h / H if H > 0 else 0.0

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)
    
    return xc, yc, wn, hn


def write_labels_for_image(
    im_meta: dict,
    anns: List[dict],
    out_root: Path,
    W: int, H: int,
    catid2yolo: Dict[int, int],
    precision: int,
    keep_crowd: bool,
    include_empty: bool,
):
    rel = Path(im_meta["file_name"]).with_suffix(".txt")
    out_path = out_root / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for ann in anns:
        if not keep_crowd and int(ann.get("iscrowd", 0)) == 1:
            continue
        
        cat_id = ann.get("category_id")
        if cat_id not in catid2yolo:
          continue
      
        yolo_id = catid2yolo[cat_id]

        bbox = ann.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        
        x, y, w, h = map(float, bbox)

        x, y, w, h = clamp_bbox_xywh(x, y, w, h, W, H)
        if w <= 0 or h <= 0:
            continue

        xc, yc, wn, hn = coco_box_to_yolo(x, y, w, h, W, H)
        fmt = f"{{:d}} {{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}} {{:.{precision}f}}"
        
        lines.append(fmt.format(yolo_id, xc, yc, wn, hn))

    if lines or include_empty:
        out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def convert_coco_to_yolo(
    coco_json: Path,
    labels_out: Path,
    classes_txt_path: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    precision: int = 6,
    keep_crowd: bool = False,
    include_empty: bool = False,
):
    images, anns_by_image, cats = load_coco(coco_json)
    classes_txt = load_classes_txt(classes_txt_path)
    catid2yolo = build_catid_to_yoloid(cats, classes_txt)

    if not catid2yolo:
        raise ValueError("No category mapping produced. Check categories or --classes.")

    for image_id in tqdm(sorted(images.keys())):
        im = images[image_id]
        W, H = ensure_image_size(im, images_dir)
        anns = anns_by_image.get(image_id, [])
        write_labels_for_image(
            im_meta=im,
            anns=anns,
            out_root=labels_out,
            W=W, H=H,
            catid2yolo=catid2yolo,
            precision=precision,
            keep_crowd=keep_crowd,
            include_empty=include_empty,
        )
        

def parse_args():
    ap = argparse.ArgumentParser(description="Convert COCO JSON annotations to YOLO txt labels.")
    ap.add_argument("--coco_json", required=True, type=Path, help="Path to COCO annotations .json")
    ap.add_argument("--labels_out", required=True, type=Path, help="Output root for YOLO txt labels")
    ap.add_argument("--classes", type=Path, default=None, help="Optional classes.txt (one name per line, YOLO id order)")
    ap.add_argument("--images_dir", type=Path, default=None, help="Root of images to infer sizes if COCO lacks width/height")
    ap.add_argument("--precision", type=int, default=6, help="Decimal digits for normalized coords")
    ap.add_argument("--keep_crowd", action="store_true", help="Include iscrowd=1 annotations")
    ap.add_argument("--include_empty", action="store_true", help="Write empty .txt for images with no annotations")
    return ap.parse_args()


def main():
    args = parse_args()
    convert_coco_to_yolo(
        coco_json=args.coco_json,
        labels_out=args.labels_out,
        classes_txt_path=args.classes,
        images_dir=args.images_dir,
        precision=args.precision,
        keep_crowd=args.keep_crowd,
        include_empty=args.include_empty,
    )
    print(f"YOLO labels written under: {args.labels_out}")

if __name__ == "__main__":
    main()

'''
# usage
python coco2yolo.py \
  --coco_json /data/annotations/instances_train.json \
  --labels_out /data/labels/train \
  --images_dir /data/images/custom \
  --classes /data/classes.txt \
  --precision 6 \
  --include_empty
'''