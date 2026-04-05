# Models folder

This project works out of the box with the classical OpenCV detector.

Optional future upgrade:
- place a trained file here as `best.pt`
- then extend `src/infer.py` or add an Ultralytics-based loader

Example future command after you prepare a dataset:
```bash
yolo detect train data=data.yaml model=yolo11n.pt epochs=50 imgsz=640
```
