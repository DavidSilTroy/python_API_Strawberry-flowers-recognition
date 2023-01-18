import base64
from flask import Flask, request, Response, send_file, jsonify
from PIL import Image
import jsonpickle
import numpy as np
import cv2
import io
from flask_restful import reqparse, abort, Api, Resource
import strw_detect

import json



app = Flask("StrawberryFlowersAPI")
api = Api(app)
y7_stw = strw_detect.StrwbDetection()

# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    print('Checking picture')
    # Get the image data from the POST request
    image_data = request.files['image'].read()
    # Decode image data as a numpy array
    image = np.frombuffer(image_data, np.uint8)
    # Decode image as a color image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Process the image (e.g. resize, crop, etc.)
    processed_image = cv2.resize(image, (800, 600))
    # Encode processed image to jpeg format
    ret, buffer = cv2.imencode('.jpg', processed_image)
    # Create a response with the processed image
    response = app.make_response(buffer.tobytes())
    response.headers.set('Content-Type', 'image/jpeg')
    return response

@app.route('/api/image-strawberry', methods=['POST'])
def detect_objects_image():
    # Get the image data from the POST request
    image_data = request.files['image'].read()
    #object detection
    img_detected, result = strw_detect.strw_detect(image_data)
    # Encode image in JPEG format
    _, img_encoded = cv2.imencode('.jpg', img_detected)
    # Convert the image to bytes
    img_bytes = img_encoded.tobytes()
    # Return the image as response
    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')

@app.route('/api/data-strawberry', methods=['POST'])
def detect_objects_data():
    # Get the image data from the POST request
    image_data = request.files['image'].read()
    #object detection
    img_detected, result = strw_detect.strw_detect(image_data)
    # Encode image in JPEG format
    _, img_encoded = cv2.imencode('.jpg', img_detected)
    # Convert the image to bytes
    img_bytes = img_encoded.tobytes()
    # Convert the image bytes to a base64 encoded string
    img_base64 = base64.b64encode(img_bytes).decode()
    # Create a dictionary to hold the image and result
    response = {
        'image': img_base64,
        'result': result
    }
    # Return the response as JSON
    return jsonify(response)

@app.route('/api/detect', methods=['POST'])
def detect_objects_data_class():
    # Get the image data from the POST request
    image_data = request.files['image'].read()
    #object detection
    img_detected, result = y7_stw.detect_strw_flowers(image_data)
    # Encode image in JPEG format
    _, img_encoded = cv2.imencode('.jpg', img_detected)
    # Convert the image to bytes
    img_bytes = img_encoded.tobytes()
    # Return the image as response
    return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    


if __name__ == '__main__':
    app.run(debug=True)