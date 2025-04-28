from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)
model = load_model("fake_image_detector.h5")

def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224), color_mode="rgb")
    img_array = img_to_array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_score = model.predict(img_array)[0][0]
    confidence = round(prediction_score * 100, 2) if prediction_score >= 0.5 else round((1 - prediction_score) * 100, 2)
    label = "Fake" if prediction_score >= 0.5 else "Real"
    return label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join("static", "uploaded_image.jpg")
            file.save(image_path)
            prediction, confidence = predict_image(image_path)
            return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
