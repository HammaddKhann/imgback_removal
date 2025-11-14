# src/utils.py

import os, cv2

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_images(folder):
    supported = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(supported)]

def save_image(path, image):
    cv2.imwrite(path, image)
