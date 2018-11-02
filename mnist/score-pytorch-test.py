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

from score import Net

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

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
    return image

def init():
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('pytorch')
    model = Net.Net()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    model.eval()

def run(input_data):
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
