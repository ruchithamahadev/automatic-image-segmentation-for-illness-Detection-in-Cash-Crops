CropDiseaseDetection - Ready to run demo
--------------------------------------

What's included:
- Flask app (app.py) — works out-of-the-box even if TensorFlow isn't installed (returns demo/random predictions).
- Small sample dataset (dataset/train and dataset/test with 6 classes, 3 images each)
- Training script (train_model.py) that will produce model/my_model.h5 when you install TensorFlow and run it.
- Simple templates and CSS.
- requirements.txt with dependencies (tensorflow is commented out; install it if you want to train/run a real model).

How to run (recommended):
1. Install Python 3.8+ and create a virtual environment:
   python -m venv env
   source env/bin/activate   # on Linux/macOS
   env\\Scripts\\activate    # on Windows (PowerShell/CMD)

2. Install dependencies:
   pip install -r requirements.txt
   If you want to train or run a real model, also install TensorFlow:
   pip install tensorflow

3. (Optional) Train the model with your dataset (requires TensorFlow):
   python train_model.py
   This will save model/my_model.h5.

4. Run the Flask app:
   python app.py
   Then open http://127.0.0.1:5000 in your browser.

Notes:
- The included app will run without TensorFlow and will return demo/random predictions to show the flow.
- Replace or expand dataset/ with your real PlantVillage images before training.
- If you want segmentation (AISA / GrabCut) integrated, reply and I will add segmentation utilities and example preprocessing steps.

Location of the zip file: /mnt/data/CropDiseaseDetection.zip
