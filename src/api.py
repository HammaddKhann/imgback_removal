# src/api.py

import os
import cv2
import numpy as np
import torch
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from rembg import remove, new_session
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights

from model_configs import MODEL_CONFIGS
from evaluate import evaluate_batch  # Changed to use evaluate_batch

app = FastAPI(
    title="Background Removal API",
    description="AI-powered background removal using multiple models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Starting API server on device: {DEVICE}")

# Cache for loaded models
MODEL_CACHE = {}


def keep_largest_component(img):
    """Keep only the largest connected component in the image."""
    if img is None:
        return None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(img)

    largest_label = 1 + np.argmax(stats[1:, 4])
    keep_mask = (labels == largest_label).astype(np.uint8) * 255
    return cv2.bitwise_and(img, img, mask=keep_mask)


def process_yolo(cfg, img):
    """Process image with YOLO segmentation model."""
    try:
        model_key = cfg["weights"]
        if model_key not in MODEL_CACHE:
            print(f"Loading {model_key}...")
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
    except Exception as e:
        print(f"YOLO Error: {e}")
        return None


def process_rembg(cfg, img):
    """Process image with rembg (U2Net/ISNet) model."""
    try:
        # Get model name from config
        model_name = cfg.get("model_name", "u2net")
        
        # Convert image to bytes for rembg
        success, img_encoded = cv2.imencode('.png', img)
        if not success:
            print(f"Failed to encode image")
            return None
            
        img_bytes = img_encoded.tobytes()
        
        # Create session and remove background
        session = new_session(model_name)
        result_bytes = remove(img_bytes, session=session)
        
        # Check if we got valid output
        if not result_bytes or len(result_bytes) == 0:
            print(f"Rembg returned empty result")
            return None
        
        # Convert back to image
        nparr = np.frombuffer(result_bytes, np.uint8)
        output = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Verify decode was successful
        if output is None or output.size == 0:
            print(f"Failed to decode rembg output")
            return None
            
        return keep_largest_component(output)
        
    except Exception as e:
        print(f"Rembg error: {e}")
        return None


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


def numpy_to_base64(img):
    """Convert numpy array to base64 string for HTML display"""
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"


def read_image_file(file_bytes):
    """Read uploaded file into numpy array"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the upload page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Background Removal</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-section {
                text-align: center;
                margin-bottom: 30px;
            }
            #fileInput {
                display: none;
            }
            .upload-btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                transition: transform 0.2s;
            }
            .upload-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            #fileName {
                margin-top: 10px;
                color: #666;
                font-style: italic;
            }
            .loader {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #667eea;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
                display: none;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .results {
                display: none;
                margin-top: 30px;
            }
            .image-comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .image-box {
                text-align: center;
            }
            .image-box h3 {
                color: #333;
                margin-bottom: 10px;
            }
            .image-box img {
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }
            .scores {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
            }
            .scores h3 {
                color: #333;
                margin-bottom: 15px;
            }
            .score-item {
                padding: 8px;
                margin: 5px 0;
                border-radius: 5px;
                background: white;
            }
            .score-item.best {
                background: #d4edda;
                border: 2px solid #28a745;
                font-weight: 600;
            }
            .download-btn {
                background: #28a745;
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                text-decoration: none;
                display: inline-block;
                transition: transform 0.2s;
            }
            .download-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Background Removal</h1>
            
            <div class="upload-section">
                <input type="file" id="fileInput" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('fileInput').click()">
                    📁 Choose Image
                </button>
                <div id="fileName"></div>
            </div>
            
            <div class="loader" id="loader"></div>
            <div class="error" id="error"></div>
            
            <div class="results" id="results">
                <div class="image-comparison">
                    <div class="image-box">
                        <h3>Original Image</h3>
                        <img id="originalImg" src="">
                    </div>
                    <div class="image-box">
                        <h3>Processed Image (Best Result)</h3>
                        <img id="processedImg" src="">
                    </div>
                </div>
                
                <div class="scores" id="scoresDiv"></div>
                
                <div style="text-align: center;">
                    <a class="download-btn" id="downloadBtn" download="processed_image.png">
                        💾 Download Processed Image
                    </a>
                </div>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const fileName = document.getElementById('fileName');
            const loader = document.getElementById('loader');
            const results = document.getElementById('results');
            const error = document.getElementById('error');
            
            fileInput.addEventListener('change', async (e) => {
                const file = e.target.files[0];
                if (!file) return;
                
                fileName.textContent = `Selected: ${file.name}`;
                
                // Show loader
                loader.style.display = 'block';
                results.style.display = 'none';
                error.style.display = 'none';
                
                // Create form data
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    // Upload and process
                    const response = await fetch('/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Processing failed');
                    }
                    
                    const data = await response.json();
                    
                    // Display results
                    document.getElementById('originalImg').src = data.original_image;
                    document.getElementById('processedImg').src = data.processed_image;
                    document.getElementById('downloadBtn').href = data.processed_image;
                    
                    
                    // Show results
                    loader.style.display = 'none';
                    results.style.display = 'block';
                    
                } catch (err) {
                    loader.style.display = 'none';
                    error.textContent = `Error: ${err.message}`;
                    error.style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": len(MODEL_CACHE),
        "available_models": list(MODEL_CONFIGS.keys())
    }


@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image with all models and return the best result"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read uploaded file
        contents = await file.read()
        
        # Validate file size (10MB limit)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 10MB")
        
        img = read_image_file(contents)
        print(f"Processing image: {file.filename}, shape: {img.shape}")
        
        # Process with all models
        results = {}
        for model_name, cfg in MODEL_CONFIGS.items():
            try:
                print(f"Processing with {model_name}...")
                func = PROCESSORS[cfg["type"]]
                output = func(cfg, img)
                results[model_name] = output
            except Exception as e:
                print(f"{model_name} failed: {e}")
                results[model_name] = None
        
        # Filter valid results
        valid_results = {m: o for m, o in results.items() if o is not None}
        
        if not valid_results:
            raise HTTPException(status_code=500, detail="No valid outputs generated from any model")
        
        # Evaluate all results using enhanced evaluation
        scores = evaluate_batch(valid_results, debug=False)  # debug=False for API
        
        # Get best model
        best_model = max(scores, key=scores.get)
        best_img = results[best_model]
        
        print(f"Best model: {best_model} with score: {scores[best_model]:.4f}")
        
        # Convert images to base64 for display
        original_base64 = numpy_to_base64(img)
        processed_base64 = numpy_to_base64(best_img)
        
        return {
            "original_image": original_base64,
            "processed_image": processed_base64,
            "best_model": best_model,
            "scores": scores,
            "message": f"Successfully processed with {len(valid_results)} models. Best result from {best_model}."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)