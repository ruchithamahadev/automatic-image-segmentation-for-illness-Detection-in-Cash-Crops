from flask import Flask, render_template, request
import numpy as np, os, random

app = Flask(__name__)

labels = ['Apple Healthy','Apple Scab','Corn Blight','Tomato Leaf Mold','Potato Early Blight','Grape Black Rot']

# Try to load a real model if tensorflow is available; otherwise, fall back to random prediction
MODEL = None
try:
    from tensorflow.keras.models import load_model
    if os.path.exists(os.path.join('model','my_model.h5')):
        MODEL = load_model(os.path.join('model','my_model.h5'))
except Exception as e:
    # TensorFlow not present or model can't be loaded
    MODEL = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    f = request.files['file']
    save_path = os.path.join('static', f.filename)
    f.save(save_path)
    # If real model loaded, use it. Otherwise pick a random label.
    if MODEL is not None:
        from tensorflow.keras.preprocessing import image
        img = image.load_img(save_path, target_size=(30,30))
        x = image.img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        pred = MODEL.predict(x)
        cls = int(np.argmax(pred, axis=1)[0])
        disease = labels[cls] if cls < len(labels) else 'Unknown'
    else:
        disease = random.choice(labels) + " (demo - install tensorflow & train model for real predictions)"
    return render_template('result.html', disease=disease, img_path='/' + save_path.replace('\\','/'))

if __name__ == '__main__':
    app.run(debug=True)
