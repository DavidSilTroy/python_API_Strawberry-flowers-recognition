from flask import Flask, request, Response, send_file, jsonify
import io
#For encoding and decoding
import base64
#customized python script from detect() in YOLOv7
import obj_detection
#to handle request errors
import werkzeug
#Cross-Origin Resource Sharing
from flask_cors import CORS 

#Initializing the YOLOv7 detectiong
# y7_model = obj_detection.Initialization(weights='yourModel.pt')
y7_model = obj_detection.Initialization()
#initializing Flask
app = Flask("Object Recognition API with YOLOv7")
#Cross-Origin Resource Sharing for the app
CORS(app)

#default index
@app.route('/')
def index():
    return "Try with going to /api"

#API index
@app.route('/api')
def apindex():
    return "Hello world from API for strwberry detection"

#transfrom image to base64
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

#get image in binary or in base64 format to send back an image
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
        img_detected, result = y7_model.detection(image)
        # Convert the image to bytes
        img_bytes = img_detected.tobytes()
        # Return the image as response
        return send_file(io.BytesIO(img_bytes), mimetype='image/jpeg')
    else:
        response = {
            'image': "Problem with the request",
            'result': request.get_json()['image']
        }
        # Return the response as JSON
        return jsonify(response)
        # return "problem detecting objects.."



#get image in binary or in base64 format to send back a json response
@app.route('/api/data-strawberry', methods=['POST'])
def get_data_from_object_detection():
    image = 0
    if request.is_json:         # Get the image data from the request body
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
        img_detected, result = y7_model.detection(image)
        try:
            # Convert the image to bytes
            img_bytes = img_detected.tobytes()
            # Convert the image bytes to a base64 encoded string
            img_base64 = base64.b64encode(img_bytes).decode()
            #Removing the ,
            if result.count(','):
                result = result.replace(',',', ')
                result = result[:-2]
            # Create a dictionary to hold the image and result
            response = {
                'image': img_base64,
                'result': result
            }
            # Return the response as JSON
            return jsonify(response)
        except:
            response = {
                'image': "Null",
                'result': result
            }
            # Return the response as JSON
            return jsonify(response)

    else:
        response = {
            'image': "Problem with the request",
            'result': request.get_json()['image']
        }
        # Return the response as JSON
        return jsonify(response)
        # return "problem detecting objects.."

#error handler for bad requests
@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    print(f'bad request! \n {e} \n\n', 400)
    return f'bad request! \n {e} \n\n', 400

#run the app once the script is run
if __name__ == '__main__':
    app.run()