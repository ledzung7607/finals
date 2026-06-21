import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import base64
import json
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from segment_food import extract_food_crops

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# CẤU HÌNH
# =========================
MODEL_PATH = os.getenv("FOOD_CLASS_MODEL", "best_food_model.h5")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "50"))

# Chọn đúng kiểu preprocess giống lúc train:
# - "rescale": model train với ImageDataGenerator(rescale=1./255) hoặc ảnh /255
# - "efficientnet": model train với EfficientNet preprocess_input
# - "none": model đã có layer Rescaling/Preprocessing bên trong
PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "rescale").lower().strip()

# Tên class phải giữ Y CHANG class_names.json, không đổi dấu gạch dưới thành khoảng trắng.
DEFAULT_CLASSES = [
    "ca_hu_kho",
    "canh_chua_co_ca",
    "canh_chua_khong_ca",
    "canh_rau",
    "com_trang",
    "dau_hu_sot_ca",
    "rau_xao",
    "suon_nuong",
    "thit_kho",
    "thit_kho_trung",
    "trung_chien",
]

PRICE_MAP = {
    "ca_hu_kho": {"name": "Cá hú kho", "price": 30000},
    "canh_chua_co_ca": {"name": "Canh chua có cá", "price": 25000},
    "canh_chua_khong_ca": {"name": "Canh chua", "price": 10000},
    "canh_rau": {"name": "Canh rau", "price": 7000},
    "com_trang": {"name": "Cơm trắng", "price": 10000},
    "dau_hu_sot_ca": {"name": "Đậu hũ sốt cà", "price": 25000},
    "rau_xao": {"name": "Rau xào", "price": 10000},
    "suon_nuong": {"name": "Sườn nướng", "price": 30000},
    "thit_kho": {"name": "Thịt kho", "price": 25000},
    "thit_kho_trung": {"name": "Thịt kho trứng", "price": 30000},
    "trung_chien": {"name": "Trứng chiên", "price": 25000},
}


def load_class_names() -> list[str]:
    path = Path(CLASS_NAMES_PATH)
    if not path.exists():
        print(f"⚠️ Không thấy {CLASS_NAMES_PATH}, dùng DEFAULT_CLASSES")
        return DEFAULT_CLASSES[:]

    data = json.loads(path.read_text(encoding="utf-8"))

    # Dạng list: ["ca_hu_kho", "canh_rau", ...]
    if isinstance(data, list):
        return [str(name).strip() for name in data if str(name).strip()]

    # Dạng dict: {"ca_hu_kho": 0, "canh_rau": 1, ...}
    if isinstance(data, dict):
        ordered = sorted(data.items(), key=lambda item: int(item[1]))
        return [str(name).strip() for name, _ in ordered if str(name).strip()]

    raise ValueError("class_names.json phải là list hoặc dict")


MENU_CLASSES = load_class_names()
class_model = None
model_error = None


def load_classifier() -> None:
    global class_model, model_error
    model_file = Path(MODEL_PATH)
    if not model_file.exists():
        class_model = None
        model_error = f"Không thấy file model: {MODEL_PATH}. Để app.py chung thư mục với best_food_model.h5"
        print("❌", model_error)
        return

    try:
        print(f"⏳ Đang tải model Keras: {MODEL_PATH}")
        class_model = load_model(str(model_file), compile=False)
        model_error = None
        print("✅ Đã tải model phân loại món ăn")
        print("✅ Class order đang dùng:", MENU_CLASSES)
    except Exception as exc:
        class_model = None
        model_error = f"Lỗi tải model: {exc}"
        print("❌", model_error)


load_classifier()


def encode_jpg_b64(img_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.resize(crop_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    arr = crop_rgb.astype(np.float32)

    if PREPROCESS_MODE == "efficientnet":
        arr = efficientnet_preprocess(arr)
    elif PREPROCESS_MODE == "none":
        pass
    else:
        arr = arr / 255.0

    return np.expand_dims(arr, axis=0)


def scores_to_probs(outputs: np.ndarray) -> np.ndarray:
    outputs = np.asarray(outputs)
    scores = outputs[0] if outputs.ndim == 2 else outputs.reshape(-1)

    # Nếu output chưa phải xác suất thì softmax.
    if np.any(scores < 0) or np.max(scores) > 1.0 or not np.isclose(np.sum(scores), 1.0, atol=0.15):
        return softmax_np(scores)
    return scores.astype(np.float32)


def item_from_key(class_key: str) -> dict:
    # Ưu tiên key đúng y chang class_names.json.
    if class_key in PRICE_MAP:
        return PRICE_MAP[class_key]

    # Fallback nếu lỡ class name có khoảng trắng thay vì dấu gạch dưới.
    normalized_key = class_key.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized_key in PRICE_MAP:
        return PRICE_MAP[normalized_key]

    return {"name": class_key, "price": 0}


def topk_predictions(probs: np.ndarray, k: int = 5) -> list[dict]:
    order = np.argsort(probs)[::-1][:k]
    out = []
    for idx in order:
        idx = int(idx)
        class_key = MENU_CLASSES[idx] if 0 <= idx < len(MENU_CLASSES) else f"idx_{idx}"
        item = item_from_key(class_key)
        out.append({
            "idx": idx,
            "class_key": class_key,
            "name": item["name"],
            "price": item["price"],
            "conf": round(float(probs[idx] * 100), 1),
        })
    return out


def predict_food(crop_bgr: np.ndarray) -> dict:
    if class_model is None:
        return {
            "class_key": None,
            "name": "Chưa có model",
            "price": 0,
            "conf": 0.0,
            "has_food": False,
            "top5": [],
        }

    x = preprocess_crop(crop_bgr)
    outputs = class_model.predict(x, verbose=0)
    probs = scores_to_probs(outputs)

    idx = int(np.argmax(probs))
    conf = round(float(probs[idx] * 100), 1)

    if idx < 0 or idx >= len(MENU_CLASSES):
        return {
            "class_key": None,
            "name": "Chưa rõ",
            "price": 0,
            "conf": conf,
            "has_food": False,
            "top5": topk_predictions(probs, 5),
        }

    class_key = MENU_CLASSES[idx]
    item = item_from_key(class_key)
    has_food = conf >= CONF_THRESHOLD and item["price"] > 0

    return {
        "class_key": class_key,
        "name": item["name"] if has_food else "Chưa rõ",
        "price": int(item["price"]) if has_food else 0,
        "conf": conf,
        "has_food": has_food,
        "top5": topk_predictions(probs, 5),
    }


@app.get("/")
def home():
    return {
        "success": True,
        "message": "Food tray API is running. POST ảnh vào /scan-khay",
        "model_loaded": class_model is not None,
        "model_error": model_error,
        "class_order": MENU_CLASSES,
        "preprocess_mode": PREPROCESS_MODE,
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "model_loaded": class_model is not None,
        "model_error": model_error,
        "model_path": MODEL_PATH,
        "class_order": MENU_CLASSES,
        "preprocess_mode": PREPROCESS_MODE,
        "crop_mode": "fixed_5_tray_compartments_no_fixed_label",
        "note": "Không ép tên theo slot. Tên món lấy từ model + class_names.json.",
    }


@app.post("/scan-khay")
async def scan_tray(file: UploadFile = File(...)):
    contents = await file.read()
    img_raw = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img_raw is None:
        return JSONResponse(status_code=400, content={"success": False, "error": "Không đọc được ảnh upload"})

    try:
        crops, debug_boxed, debug_mask, original_boxed, original_boxes = extract_food_crops(img_raw)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Lỗi cắt khay: {exc}"})

    total_bill = 0
    results_list = []

    for slot, crop in enumerate(crops, start=1):
        pred = predict_food(crop)
        total_bill += int(pred["price"])

        x0, y0, x1, y1 = original_boxes[slot - 1]
        results_list.append({
            "slot": slot,
            "has_food": pred["has_food"],
            "name": pred["name"],
            "price": pred["price"],
            "conf": pred["conf"],
            "class_key": pred["class_key"],
            "top5": pred["top5"],
            "image": encode_jpg_b64(crop),
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        })

    return {
        "success": True,
        "total": total_bill,
        "items": results_list,
        "debug_image": encode_jpg_b64(original_boxed),
        "model_loaded": class_model is not None,
        "model_error": model_error,
        "class_order": MENU_CLASSES,
        "preprocess_mode": PREPROCESS_MODE,
    }


@app.post("/debug-predict")
async def debug_predict(file: UploadFile = File(...)):
    """Trả top 5 index/class cho từng crop để kiểm tra model đoán gì."""
    contents = await file.read()
    img_raw = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img_raw is None:
        return JSONResponse(status_code=400, content={"success": False, "error": "Không đọc được ảnh upload"})

    try:
        crops, _, _, original_boxed, _ = extract_food_crops(img_raw)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Lỗi cắt khay: {exc}"})

    debug_items = []
    for slot, crop in enumerate(crops, start=1):
        pred = predict_food(crop)
        debug_items.append({
            "slot": slot,
            "predicted": pred["name"],
            "class_key": pred["class_key"],
            "conf": pred["conf"],
            "top5": pred["top5"],
            "image": encode_jpg_b64(crop),
        })

    return {
        "success": True,
        "items": debug_items,
        "debug_image": encode_jpg_b64(original_boxed),
        "class_order": MENU_CLASSES,
        "preprocess_mode": PREPROCESS_MODE,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
