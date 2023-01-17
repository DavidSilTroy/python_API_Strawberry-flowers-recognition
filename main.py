from flask import Flask, request, Response
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

@app.route('/detect-objects', methods=['POST'])
def detect_objects():
    # Get the image data from the POST request
    image_data = request.files['image'].read()
    # Decode image data as a numpy array
    image = np.frombuffer(image_data, np.uint8)
    # Decode image as a color image
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Run the object detector
    # objects =  

    strw_detect.strw_detect(source='fortest', weights=['rtrain-2.pt'],conf_thres= 0.5, img_size=640,name="fromapi")
    # Create a response with the object detections
    # response = app.make_response(json.dumps(objects))
    # response.headers.set('Content-Type', 'application/json')
    # return response
    return 'haciendo xd'


if __name__ == '__main__':
    app.run(debug=True)