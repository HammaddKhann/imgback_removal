# src/model_configs.py

MODEL_CONFIGS = {
    "yolov8x": {
        "type": "yolo",
        "weights": "yolov8x-seg.pt",
        "imgsz": 1120,
        "conf": 0.20,
        "iou": 0.6
    },
    "yolov9c": {
        "type": "yolo",
        "weights": "yolov9c-seg.pt",
        "imgsz": 1240,
        "conf": 0.20,
        "iou": 0.6
    },
    "u2net": {
        "type": "rembg"
    },
    "deeplabv3": {
        "type": "deeplab"
    }
}
