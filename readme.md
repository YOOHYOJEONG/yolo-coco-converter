# YOLO / COCO Format Converter

This repository provides simple utilities to convert annotation files between  
**YOLO format** (`.txt`) and **COCO format** (`.json`).

Currently supported:
-  **YOLO → COCO** conversion (`convert_yolo2coco.py`)
-  **COCO → YOLO** conversion (coming soon...)

---

##  YOLO → COCO

`convert_yolo2coco.py` converts labeling files in YOLO format into a COCO-style annotation JSON.

### Usage

```bash
python convert_yolo2coco.py \
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
Coming Soon,,,,