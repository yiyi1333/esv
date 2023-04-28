from io import BytesIO

import requests
import torch
import torchvision
from PIL import Image, ImageOps
from flask import Flask, request, jsonify
import json

from torchvision import transforms

app = Flask(__name__)



if __name__ == '__main__':
    app.run()