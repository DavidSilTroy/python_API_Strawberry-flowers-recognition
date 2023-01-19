from flask import Flask, request, Response, send_file, jsonify
import numpy as np
import cv2


app = Flask("StrawberryFlowersAPI")

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

if __name__ == '__main__':
    app.run()