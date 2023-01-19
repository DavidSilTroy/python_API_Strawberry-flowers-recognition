import base64
from flask import Flask, request, Response, send_file, jsonify
from PIL import Image
import jsonpickle
import numpy as np
import cv2
import io
import strw_detect


app = Flask("StrawberryFlowersAPI")
y7_stw = strw_detect.StrwbDetection()

@app.route('/')
def index():
    return "Try with going to /api"

@app.route('/api')
def apindex():
    return "Hello world from API for strwberry detection"


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    print('Checking picture')
    if request.files['image']:
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
    else:
        return "nothing"


@app.route('/api/image-strawberry', methods=['POST'])
def get_image_from_object_detection():
    if not y7_stw:
        y7_stw = strw_detect.StrwbDetection()
    
    if request.files['image']:
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
    else:
        return "nothing"

@app.route('/api/data-strawberry', methods=['POST'])
def get_data_from_object_detection():
    if not y7_stw:
        y7_stw = strw_detect.StrwbDetection()
    
    if request.files['image']:
        # Get the image data from the POST request
        image_data = request.files['image'].read()
        #object detection
        img_detected, result = y7_stw.detect_strw_flowers(image_data)
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
    else:
        return "nothing"



if __name__ == '__main__':
    app.run()