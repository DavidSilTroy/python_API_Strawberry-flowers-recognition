import base64
from flask import Flask, request, Response, send_file, jsonify
from PIL import Image
import jsonpickle
import numpy as np
import cv2
import io



app = Flask("StrawberryFlowersAPI")


@app.route('/')
def index():
    return "Try with going to /api"

@app.route('/api')
def apindex():
    return "Hello world from API for strwberry detection"


    

if __name__ == '__main__':
    app.run()