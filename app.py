import os
import json
import numpy as np
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------- Configuration ----------------
APP_NAME = "Leaf Disease Detection System"

app = Flask(__name__)
app.secret_key = "change_this_secret_to_a_random_string"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
DATA_FOLDER = os.path.join(BASE_DIR, "data")
USERS_FILE = os.path.join(DATA_FOLDER, "users.json")
TREATMENTS_FILE = os.path.join(DATA_FOLDER, "treatments.json")
DISEASES_FILE = os.path.join(DATA_FOLDER, "disease_info.json")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
IMG_SIZE = (128, 128)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

# ---------------- Model & Data Loading ----------------
MODEL = None
LABELS = {}
TREATMENTS = {}
DISEASES = {}

# Load model
try:
    if os.path.exists(MODEL_PATH):
        MODEL = load_model(MODEL_PATH)
        print("✓ Model loaded")
    else:
        print("⚠ model.h5 not found.")
except Exception as e:
    print("✗ Error loading model:", e)

# Load labels
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        LABELS = json.load(f)
    LABELS = {int(k): v for k, v in LABELS.items()}

# Load treatments
if os.path.exists(TREATMENTS_FILE):
    with open(TREATMENTS_FILE, "r", encoding="utf-8") as f:
        TREATMENTS = json.load(f)
else:
    TREATMENTS = {"Unknown___Unknown": "No treatment available."}
    with open(TREATMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(TREATMENTS, f, indent=4)

# Load disease info
if os.path.exists(DISEASES_FILE):
    with open(DISEASES_FILE, "r", encoding="utf-8") as f:
        DISEASES = json.load(f)

# ---------------- Background Mapping ----------------
BG_MAPPING = {
    "Apple___Apple_scab": "apple_scab.jpg",
    "Apple___Black_rot": "apple_black_rot.jpg",
    "Apple___Cedar_apple_rust": "apple_cedar_rust.jpg",
    "Apple___healthy": "apple_healthy.jpg",
    "Corn_(maize)_Cercospora_leaf_spot_Gray_leaf_spot": "corn_cercospora.jpg",
    "Corn_(maize)_Common_rust": "corn_common_rust.jpg",
    "Corn_(maize)_Northern_Leaf_Blight": "corn_northern_blight.jpg",
    "Corn_(maize)_healthy": "corn_healthy.jpg",
    "Pepper_bell___Bacterial_spot": "pepper_bacterial_spot.jpg",
    "Pepper_bell___healthy": "pepper_healthy.jpg",
    "Potato___Early_blight": "potato_early_blight.jpg",
    "Potato___Late_blight": "potato_late_blight.jpg",
    "Potato___healthy": "potato_healthy.jpg",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot.jpg",
    "Tomato___Early_blight": "tomato_early_blight.jpg",
    "Tomato___Late_blight": "tomato_late_blight.jpg",
    "Tomato___Leaf_Mold": "tomato_leaf_mold.jpg",
    "Tomato___Septoria_leaf_spot": "tomato_septoria_leaf_spot.jpg",
    "Tomato___Spider_mites_Two_spotted_spider_mite": "tomato_spider_mites.jpg",
    "Tomato___Target_Spot": "tomato_target_spot.jpg",
    "Tomato___Tomato_YellowLeaf_Curl_Virus": "tomato_yellow_leaf_curl.jpg",
    "Tomato___Tomato_mosaic_virus": "tomato_mosaic.jpg",
    "Tomato___healthy": "tomato_healthy.jpg"
}

# ---------------- Users DB ----------------
def load_users():
    if not os.path.exists(USERS_FILE):
        default = {"admin": {"password_hash": generate_password_hash("admin123"), "role": "admin"}}
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=4)
        return default
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users_dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=4)

USERS = load_users()

# ---------------- Auth Decorator ----------------
def login_required(role=None):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if "username" not in session:
                return redirect(url_for("login"))
            if role and session.get("role") != role:
                flash("Access denied.", "error")
                return redirect(url_for("login"))
            return f(*args, **kwargs)
        return wrapped
    return decorator

# ---------------- Prediction ----------------
def predict_from_image(save_path):
    if MODEL is None:
        raise RuntimeError("Model not loaded.")
    
    img = image.load_img(save_path, target_size=IMG_SIZE)
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    
    preds = MODEL.predict(x)[0]
    cls_index = int(np.argmax(preds))
    confidence = round(float(preds[cls_index]) * 100, 2)
    
    full_label = LABELS.get(cls_index, "Unknown___Unknown")
    if "_" in full_label:
        leaf, disease = full_label.split("_", 1)
    else:
        leaf = disease = full_label
    
    treatment = TREATMENTS.get(full_label, "No treatment info.")
    img_rel_path = os.path.relpath(save_path, os.path.join(BASE_DIR, "static")).replace("\\", "/")
    
    # Background image
    bg_image_file = BG_MAPPING.get(full_label, "default_bg.png")
    bg_image_path = f"images/bg/{bg_image_file}"  # Use forward slashes

    return {
        "leaf_name": leaf.replace("_", " "),
        "disease": disease.replace("_", " "),
        "confidence": confidence,
        "treatment": treatment,
        "img_path": img_rel_path,
        "disease_key": full_label,
        "bg_image": bg_image_path
    }

# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        role = request.form.get("role", "user")
        username = request.form.get("username")
        password = request.form.get("password")
        
        users = load_users()
        user = users.get(username)
        
        if user and check_password_hash(user["password_hash"], password) and user["role"] == role:
            session["username"] = username
            session["role"] = role
            return redirect(url_for("admin_dashboard" if role=="admin" else "user_dashboard"))
        
        flash("Invalid login.", "error")
    
    return render_template("login.html", app_name=APP_NAME)

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method=="POST":
        username = request.form["username"]
        password = request.form["password"]
        confirm = request.form["confirm"]
        
        if password != confirm:
            flash("Passwords do not match.", "error")
            return redirect(url_for("register"))
        
        users = load_users()
        if username in users:
            flash("Username already exists.", "error")
            return redirect(url_for("register"))
        
        users[username] = {"password_hash": generate_password_hash(password), "role": "user"}
        save_users(users)
        flash("Registration successful.", "success")
        return redirect(url_for("login"))
    
    return render_template("register.html", app_name=APP_NAME)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- User Dashboard ----------------
@app.route("/user", methods=["GET","POST"])
@login_required(role="user")
def user_dashboard():
    if request.method=="POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)
            try:
                result = predict_from_image(save_path)
                return render_template("result.html", username=session.get("username"), app_name=APP_NAME, **result)
            except Exception as e:
                flash(str(e), "error")
    
    return render_template("user_dashboard.html", username=session.get("username"), app_name=APP_NAME)

# ---------------- Admin Dashboard ----------------
@app.route("/admin")
@login_required(role="admin")
def admin_dashboard():
    uploads = os.listdir(UPLOAD_FOLDER)
    users = load_users()
    return render_template("admin_dashboard.html", username=session.get("username"), uploads=len(uploads), users=len(users), app_name=APP_NAME)

@app.route("/admin/uploads")
@login_required(role="admin")
def admin_uploads():
    files = sorted(os.listdir(UPLOAD_FOLDER), reverse=True)
    return render_template("uploads.html", files=files, app_name=APP_NAME)

@app.route("/admin/delete/<filename>", methods=["POST"])
@login_required(role="admin")
def admin_delete_file(filename):
    safe_name = secure_filename(filename)
    full = os.path.join(UPLOAD_FOLDER, safe_name)
    if os.path.exists(full):
        os.remove(full)
        flash("File deleted.", "success")
    return redirect(url_for("admin_uploads"))

@app.route("/admin/treatments", methods=["GET","POST"])
@login_required(role="admin")
def admin_treatments():
    if request.method=="POST":
        for key in TREATMENTS:
            TREATMENTS[key] = request.form.get(key, TREATMENTS[key])
        with open(TREATMENTS_FILE,"w",encoding="utf-8") as f:
            json.dump(TREATMENTS,f,indent=4)
        flash("Updated.","success")
    
    return render_template("treatments.html", treatments=TREATMENTS, app_name=APP_NAME)

@app.route("/uploads/<path:filename>")
@login_required(role="admin")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# ---------------- Disease Info Page ----------------
@app.route("/disease/<path:disease_key>")
@login_required(role="user")
def disease_info(disease_key):
    disease_key = disease_key.replace('%20', '_')
    disease_info = DISEASES.get(disease_key)
    if not disease_info:
        return render_template("disease_not_found.html", disease_key=disease_key), 404
    return render_template("disease_info.html", disease_info=disease_info)

# ---------------- Run App ----------------
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


