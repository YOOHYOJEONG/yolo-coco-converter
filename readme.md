# YOLO / COCO Format Converter

This repository provides simple utilities to convert annotation files between  
**YOLO format** (`.txt`) and **COCO format** (`.json`).

Currently supported:
-  **YOLO → COCO** conversion (`yolo2coco.py`)
-  **COCO → YOLO** conversion (`coco2yolo.py`)

---

##  YOLO → COCO

`yolo2coco.py` converts labeling files in YOLO format into a COCO-style annotation JSON.

### Usage

```bash
python yolo2coco.py \
  --images_dir ./dataset/train/images \
  --labels_dir ./dataset/train/labels \
  --classes    ./classes.txt \
  --output     ./output/train_coco.json   
```   
#### Arguments (YOLO → COCO)

| Argument        | Description                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| `--images_dir`  | Directory containing the dataset images                                                                       |
| `--labels_dir`  | Directory containing YOLO label (`.txt`) files                                                                |
| `--classes`     | Text file that lists class names (one class per line – **must follow the same order as YOLO class IDs**)       |
| `--output`      | Output path of the converted **COCO** annotation file (including file name and `.json` extension)              |


####  Example of `classes.txt`

The `classes.txt` file should contain one class name **per line**.  
**Classes must be written in the same order as their YOLO class IDs.**

Example:     
car   
person    

---   

##  COCO → YOLO    
coco2yolo.py converts labeling files in COCO JSON format into YOLO-style label files (.txt).   
Each output .txt corresponds to a single image and contains normalized bounding boxes.   

### Usage
```bash
python coco2yolo.py \
  --coco_json ./annotations/instances_train.json \
  --labels_out ./labels/train \
  --images_dir ./images/train \
  --classes ./classes.txt \
  --precision 6 \
  --include_empty
```   

#### Arguments (COCO → YOLO)

| Argument        | Description                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------------|
| `--coco_json`  | Path to the input COCO annotations JSON file |
| `--labels_out` | Output root directory where YOLO .txt labels will be saved  |
| `--classes` | Optional classes.txt file (one class name per line, in YOLO ID order). If not provided, COCO categories are used in sorted order |
| `--images_dir` | Directory containing dataset images. Used to infer image width/height if missing in the COCO JSON |
| `--precision` | Decimal digits for normalized YOLO coordinates (default: 6) |
| `--keep_crowd` | If set, keeps annotations where iscrowd=1 (ignored by default) |
| `--include_empty` | If set, writes empty .txt files for images with no annotations |
   

####  Example of `classes.txt`

The `classes.txt` file should contain one class name **per line**.  
**Classes must be written in the same order as their YOLO class IDs.**

Example:     
car   
person  

#### Output Format
Each .txt file will have one line per object in the format:   
`<class_id> <x_center> <y_center> <width> <height>`   
