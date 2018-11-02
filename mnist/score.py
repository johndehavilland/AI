import argparse
import base64
import json
import os
import urllib.request
from io import BytesIO

import numpy as np
from azureml.core import Workspace
from azureml.core.model import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import requests

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def base64ToImg(path):
    print(path)
    response = requests.get(path)
    img = Image.open(BytesIO(response.content))
    return img

def preprocess_image(image):
    """Preprocess the input image."""
    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = loader(image).float()
    data = torch.tensor(image)
    data = data.unsqueeze(1)
    print(data.shape)
    return data

def init():
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('pytorch')
    model = Net()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()

def run(input_data):
    print("torch-version: " + torch.__version__)
    img = base64ToImg(json.loads(input_data)['data'])
    img = preprocess_image(img)

    # get prediction
    output = model(img)

    classes = [0, 1, 2,3,4,5,6,7,8,9]
    softmax = nn.Softmax(dim=1)
    pred_probs = softmax(model(img)).detach().numpy()[0]
    index = torch.argmax(output, 1)

    result = json.dumps({"label": classes[index], "probability": str(pred_probs[index])})
    return result
