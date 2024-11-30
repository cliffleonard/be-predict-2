from werkzeug.utils import secure_filename
from flask import Flask, request, send_from_directory
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image
from flask import jsonify
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import normalize

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the Model
model   = tf.keras.models.load_model('best_model2.h5')
classes = ['doberman', 'malamute', 'maltese', 'miniature_pinscher', 'shitzu', 'siberian_husky']
size    = 512

@app.route('/predict', methods=['POST'])
def predict():

    image = request.files['image']
    name  = save_file(image)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], name)

    image     = cv2.imread(image_path)
    image     = Image.fromarray(image)
    image     = image.resize((size, size))

    data_norm     = normalize(np.array(image), axis=1)
    data_reshaped = np.expand_dims(data_norm, axis = 0)

    pred   = model.predict(data_reshaped)
    y_pred = np.argmax(pred, axis=1)
    result = classes[y_pred[0]]

    probabilities = [{"predict": classes[i], "persen": f"{pred[0][i] * 100:.2f}%"} for i in range(len(classes))]
    probabilities_sorted = sorted(probabilities, key=lambda x: float(x["persen"].strip('%')), reverse=True)

    data = {'image_path': image_path,
            'predict': result,
            'probabilities': probabilities_sorted
            }
            
    return returnAPI(200, 'Success', data)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

def save_file(image):
    name = secure_filename(image.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], name)

    try:
        os.remove(path)
    except OSError:
        pass
        
    image.save(path)
    return name

def returnAPI(code=200, message='', data=[]):
    status = 'success'
    if code != 200:
        status = 'failed'
    returnArray = {
        'code': code,
        'status': status,
        'message': message,
        'data': data
    }
    return jsonify(returnArray)

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5288, debug=True)