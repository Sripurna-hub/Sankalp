import os
import base64

from PIL import Image
from backend.config import UPLOADS_DIR

def save_uploaded_image(uploaded_file, file_id):
    save_path = os.path.join(UPLOADS_DIR, file_id)
    pil_img = Image.open(uploaded_file)
    pil_img.save(save_path)
    return save_path

def save_uploaded_video(uploaded_file, file_id):
    save_path = os.path.join(UPLOADS_DIR, file_id)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    return save_path

def encode_image_to_base64(image_path):
    """Encodes image to base64 string for embedded Folium map HTML popups."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        pass
    return None
