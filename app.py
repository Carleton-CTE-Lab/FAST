#!/usr/bin/env python
# coding: utf-8

import io
import gc
import json
import torch
import tifffile
import cv2
import tempfile

import numpy as np
from flask import Flask, request, render_template, Response

from keras.models import load_model

# Custom visualization functions
import utils.utils as utils

# Load the model
model = load_model("model.keras", custom_objects={'label2_iou': utils.label2_iou}, compile=False)
config = json.load(open("config.json", "r"))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    img = request.files["image"]
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        img.save(temp_path)
    
    image = utils.read_image(temp_path)
    
    prediction_mask = utils.infer(image_tensor=image, model=model)
    
    input_image = cv2.imread(temp_path)
    img_rgb = utils.plot_samples_matplotlib([input_image, prediction_mask], (18, 14), prediction_mask)
    
    buffer = io.BytesIO()
    img_rgb.save(buffer, format="PNG")
    buffer.seek(0)
    img_bytes = buffer.getvalue()
    
    return Response(img_bytes, mimetype="image/png")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)