import base64
from flask import Flask, request, Response, send_file, jsonify
import io
import strw_detect


y7_stw = strw_detect.StrwbDetection()
app = Flask("StrawberryFlowersAPI")

@app.route('/')
def index():
    return "Try with going to /api"

@app.route('/api')
def apindex():
    return "Hello world from API for strwberry detection"


@app.route('/api/image-to_base64', methods=['POST'])
def get_image_to_base64():
    if request.files['image']:
        # Get the image data from the POST request
        image_data = request.files['image'].read()
        # Encode image data to base64
        base64_image = base64.b64encode(image_data)
        # Create json response with base64 image
        response = jsonify({"image": base64_image.decode()})
        response.headers.set('Content-Type', 'application/json')
        return response
    else:
        return "nothing"

@app.route('/api/image-strawberry', methods=['POST'])
def get_image_from_object_detection():
    image = 0
    if request.is_json: 
        # Get the image data from the request body
        image_data = request.get_json()['image']
        # Decode base64 image to bytes
        image_bytes = base64.b64decode(image_data)
        image = image_bytes
    elif request.files['image']:
        # Get the image data from the POST request
        image_data = request.files['image'].read()
        image = image_data
    else:
        return "nothing"
    
    if isinstance(image, bytes):
        #object detection
        img_detected, result = y7_stw.detect_strw_flowers(image)
        # Convert the image to bytes
        img_bytes = img_detected.tobytes()
        # Return the image as response
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    else:
        return "problem detecting objects.."



@app.route('/api/data-strawberry', methods=['POST'])
def get_data_from_object_detection():
    image = 0
    if request.is_json: 
        # Get the image data from the request body
        image_data = request.get_json()['image']
        # Decode base64 image to bytes
        image_bytes = base64.b64decode(image_data)
        image = image_bytes
    elif request.files['image']:
        # Get the image data from the POST request
        image_data = request.files['image'].read()
        image = image_data 
    else:
        return "nothing"
    
    if isinstance(image, bytes):
        #object detection
        img_detected, result = y7_stw.detect_strw_flowers(image)
        # Convert the image to bytes
        img_bytes = img_detected.tobytes()
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
        return "problem detecting objects.."



if __name__ == '__main__':
    app.run()