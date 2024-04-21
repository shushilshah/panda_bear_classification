from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)

# Load the pre-trained image classification model
model = tf.keras.models.load_model('pandabear.h5')

# Define class labels
class_labels = ['Panda', 'Bear']


def preprocess_image(image):
    # assuming the model expects images of size 224x224
    img = image.resize((128, 128))
    img = np.asarray(img) / 255.0  # normalize pixel values to [0, 1]
    return img


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Read the uploaded image
        image = Image.open(BytesIO(file.read()))

        # Preprocess the image
        processed_image = preprocess_image(image)
        processed_image = np.expand_dims(
            processed_image, axis=0)  # add batch dimension

        # Perform prediction
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        # Render the result template with image and classification result
        return render_template('result.html', image_data=file.read(), predicted_class=predicted_class, confidence=float(predictions[0][predicted_class_index]))


if __name__ == '__main__':
    app.run(debug=True)
