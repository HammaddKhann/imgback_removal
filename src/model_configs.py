MODEL_CONFIGS = {
    # YOLOv8x - Balanced speed and accuracy
    "yolov8x": {
        "type": "yolo",
        "weights": "yolov8x-seg.pt",
        "imgsz": 1280,  # Increased for better quality
        "conf": 0.25,    # Slightly higher confidence
        "iou": 0.7,      # Higher IoU for better overlap handling
    },
    
    # YOLOv9c - Latest architecture
    "yolov9c": {
        "type": "yolo",
        "weights": "yolov9c-seg.pt",
        "imgsz": 1280,   # Increased
        "conf": 0.25,    # Slightly higher
        "iou": 0.7,      # Higher IoU
    },
    
    # U2Net - Specialized for background removal
    "u2net": {
        "type": "rembg",
        "model_name": "u2net"  # Can also try: "u2netp", "u2net_human_seg"
    },
    
    # Alternative: Try isnet for better quality
    "isnet": {
        "type": "rembg",
        "model_name": "isnet-general-use"
    },
    
    # DeepLab - Good for semantic segmentation
    "deeplabv3": {
        "type": "deeplab"
    }
}

# Optional: Model-specific post-processing configs
POST_PROCESS_CONFIGS = {
    "yolov8x": {
        "morph_kernel_size": 7,
        "morph_iterations": 3,
        "dilation_iterations": 1
    },
    "yolov9c": {
        "morph_kernel_size": 7,
        "morph_iterations": 3,
        "dilation_iterations": 1
    },
    "u2net": {
        "morph_kernel_size": 5,
        "morph_iterations": 2,
        "dilation_iterations": 0
    },
    "isnet": {
        "morph_kernel_size": 5,
        "morph_iterations": 2,
        "dilation_iterations": 0
    },
    "deeplabv3": {
        "morph_kernel_size": 5,
        "morph_iterations": 2,
        "dilation_iterations": 0
    }
}