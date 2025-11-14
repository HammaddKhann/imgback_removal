# src/main.py

import os, cv2, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
from rembg import remove
import torchvision

from model_configs import MODEL_CONFIGS
from evaluate import evaluate_output
from utils import ensure_dir, load_images, save_image

INPUT_DIR = "sample_input"
OUTPUT_DIR = "sample_output"
ensure_dir(OUTPUT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def keep_largest_component(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(img)

    largest_label = 1 + np.argmax(stats[1:, 4])  # skip background
    keep_mask = (labels == largest_label).astype(np.uint8) * 255
    return cv2.bitwise_and(img, img, mask=keep_mask)


def process_yolo(cfg, img):
    model = YOLO(cfg["weights"])
    preds = model.predict(
        source=img,
        imgsz=cfg["imgsz"],
        conf=cfg["conf"],
        iou=cfg["iou"],
        device=DEVICE,
        retina_masks=True,
        verbose=False
    )

    H, W = img.shape[:2]
    black_bg = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for r in preds:
        if r.masks is None:
            continue
        for m in r.masks.data.cpu().numpy():
            m = cv2.resize(m, (W, H))
            m = (m > 0.5).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=4)
            m = cv2.dilate(m, kernel, iterations=1)
            if cv2.countNonZero(m) < 800:
                continue
            obj = cv2.bitwise_and(img, img, mask=m)
            black_bg = cv2.bitwise_or(black_bg, obj)

    black_bg = keep_largest_component(black_bg)
    return black_bg


def process_rembg(_, img):
    result_bytes = remove(img)
    nparr = np.frombuffer(result_bytes, np.uint8)
    output = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return keep_largest_component(output)


def process_deeplab(_, img):
    model = torch.hub.load("pytorch/vision", "deeplabv3_resnet101", pretrained=True).to(DEVICE)
    model.eval()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        out = model(tensor)["out"]
    mask = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8) * 255
    result = cv2.bitwise_and(img, img, mask=mask)
    return keep_largest_component(result)



PROCESSORS = {
    "yolo": process_yolo,
    "rembg": process_rembg,
    # "deeplab": process_deeplab
}

print(f"\nDetected device: {DEVICE.upper()}")
images = load_images(INPUT_DIR)
print(f"Found {len(images)} images in {INPUT_DIR}\n")

for img_path in tqdm(images, desc="Processing images"):
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    assert img is not None, f"Could not read {img_path}"

    results = {}
    for model_name, cfg in MODEL_CONFIGS.items():
        try:
            func = PROCESSORS[cfg["type"]]
            output = func(cfg, img)
            results[model_name] = output
        except Exception as e:
            print(f"{model_name} failed on {img_name}: {e}")

    scores = {m: evaluate_output(o) for m, o in results.items() if o is not None}
    if not scores:
        print(f"No valid outputs for {img_name}. Skipping.")
        continue

    best_model = max(scores, key=scores.get)
    best_img = results[best_model]

    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_best_{best_model}.jpg")
    save_image(out_path, best_img)

    print(f"\n{img_name} results:")
    for k, v in scores.items():
        print(f"   {k}: {v:.4f}")
    print(f"Best model for {img_name}: {best_model}\n")

print(f"\nAll results saved in '{OUTPUT_DIR}'")
