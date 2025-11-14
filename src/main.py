import os, cv2, numpy as np, torch
from tqdm import tqdm
from ultralytics import YOLO
from rembg import remove

from model_configs import MODEL_CONFIGS
from evaluate import evaluate_batch  # Changed from evaluate_output
from utils import ensure_dir, load_images, save_image

INPUT_DIR = "sample_input"
OUTPUT_DIR = "sample_output"
ensure_dir(OUTPUT_DIR)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache for loaded models to avoid reloading
MODEL_CACHE = {}


def keep_largest_component(img):
    """Keep only the largest connected component in the image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(img)

    largest_label = 1 + np.argmax(stats[1:, 4])  # skip background
    keep_mask = (labels == largest_label).astype(np.uint8) * 255
    return cv2.bitwise_and(img, img, mask=keep_mask)


def process_yolo(cfg, img):
    """Process image with YOLO segmentation model."""
    # Use cached model if available
    model_key = cfg["weights"]
    if model_key not in MODEL_CACHE:
        print(f"  Loading {model_key}...")
        MODEL_CACHE[model_key] = YOLO(cfg["weights"])
    model = MODEL_CACHE[model_key]
    
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


def process_rembg(cfg, img):
    """Process image with rembg (U2Net/ISNet) model."""
    try:
        from rembg import remove, new_session
        
        # Get model name from config
        model_name = cfg.get("model_name", "u2net")
        
        # Convert image to bytes for rembg
        _, img_encoded = cv2.imencode('.png', img)
        img_bytes = img_encoded.tobytes()
        
        # Create session and remove background
        session = new_session(model_name)
        result_bytes = remove(img_bytes, session=session)
        
        # Convert back to image
        nparr = np.frombuffer(result_bytes, np.uint8)
        output = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if decode was successful
        if output is None:
            print(f"    Warning: Failed to decode output for {model_name}")
            return None
        
        return keep_largest_component(output)
        
    except Exception as e:
        print(f"    Error in rembg processing: {e}")
        return None

from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

def process_deeplab(cfg, img):
    """Process image with DeepLabv3 model."""
    try:
        if "deeplab" not in MODEL_CACHE:
            print("  Loading DeepLabv3...")

            weights = DeepLabV3_ResNet101_Weights.DEFAULT
            MODEL_CACHE["deeplab"] = deeplabv3_resnet101(
                weights=weights
            ).to(DEVICE)

            MODEL_CACHE["deeplab"].eval()
        
        model = MODEL_CACHE["deeplab"]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = tensor.to(DEVICE)
        
        with torch.no_grad():
            out = model(tensor)["out"]
        
        mask = out.argmax(1).squeeze().cpu().numpy().astype(np.uint8) * 255
        result = cv2.bitwise_and(img, img, mask=mask)
        return keep_largest_component(result)
    except Exception as e:
        print(f"  DeepLab Error: {e}")
        return None

PROCESSORS = {
    "yolo": process_yolo,
    "rembg": process_rembg,
    "deeplab": process_deeplab
}


def main():
    """Main processing pipeline."""
    print("="*70)
    print("BACKGROUND REMOVAL - ENHANCED EVALUATION SYSTEM")
    print("="*70)
    print(f"\nDevice: {DEVICE.upper()}")
    
    images = load_images(INPUT_DIR)
    print(f"Found {len(images)} image(s) in '{INPUT_DIR}'")
    print(f"Models configured: {', '.join(MODEL_CONFIGS.keys())}")
    print("="*70 + "\n")

    if len(images) == 0:
        print(f"No images found in '{INPUT_DIR}'. Please add images to process.")
        return

    # Process each image
    for img_idx, img_path in enumerate(images, 1):
        img_name = os.path.basename(img_path)
        
        print(f"\n{'='*70}")
        print(f"Processing [{img_idx}/{len(images)}]: {img_name}")
        print(f"{'='*70}")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read {img_path}")
            continue
        
        print(f"📐 Image size: {img.shape[1]}x{img.shape[0]} pixels")
        
        # Process with all models
        results = {}
        print("\nProcessing with models:")
        
        for model_name, cfg in MODEL_CONFIGS.items():
            try:
                print(f"  • {model_name}...", end=" ", flush=True)
                func = PROCESSORS[cfg["type"]]
                output = func(cfg, img)
                results[model_name] = output
                print("✓")
            except Exception as e:
                print(f"✗ Failed: {e}")
                results[model_name] = None

        # Filter out failed results
        valid_results = {m: o for m, o in results.items() if o is not None}
        
        if not valid_results:
            print(f"\n❌ No valid outputs for {img_name}. Skipping.")
            continue

        # Evaluate all results with detailed scores
        print("\n" + "="*70)
        print("EVALUATION RESULTS (Enhanced 5-Metric System)")
        print("="*70)
        
        scores = evaluate_batch(valid_results, debug=True)  # debug=True for detailed output

        # Find best model
        best_model = max(scores, key=scores.get)
        best_img = results[best_model]

        # Save best result
        base_name = os.path.splitext(img_name)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_best_{best_model}.png")
        save_image(out_path, best_img)
        
        print(f"\nSaved best result: {os.path.basename(out_path)}")
        
        # Optional: Save all results for comparison
        if len(valid_results) > 1:
            save_all = input("\n💡 Save all model outputs for comparison? (y/n): ").lower()
            if save_all == 'y':
                for model_name, output in valid_results.items():
                    model_out_path = os.path.join(
                        OUTPUT_DIR, 
                        f"{base_name}_{model_name}_score{scores[model_name]:.3f}.png"
                    )
                    save_image(model_out_path, output)
                print(f"✓ Saved all {len(valid_results)} outputs to '{OUTPUT_DIR}'")

    print("\n" + "="*70)
    print("ALL PROCESSING COMPLETE!")
    print(f"Results saved in '{OUTPUT_DIR}'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()