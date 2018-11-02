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
import torchvision.transforms.functional as Fun
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
from utils import load_data
from score import Net

# model.download(target_dir = '.', exists_ok = True)


ws = Workspace.from_config()

model=Model(ws, 'pytorch')
# verify the downloaded model file
os.stat('./pytorch_model.pt')


# os.makedirs('./data', exist_ok = True)

# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')

X_test = load_data('./data/test-images.gz', False) / 255.0
y_test = load_data('./data/test-labels.gz', True).reshape(-1)

def init():
    global model
    model_path = Model.get_model_path('pytorch')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()

def base64ToImg(path):
    print(path)
    base64ImgString = Image.open(path)
    return base64ImgString

loader = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


def preprocess_image(data):
    """load image, returns cuda tensor"""
    #image = Image.open(image_name)

    data = loader(data).float()
    data = Variable(data)
    data = data.unsqueeze(1)
    print(data.shape)
    return data  #assumes that you're using GPU

def run(img):
    # img = base64ToImg(json.loads(input_data)['data'])
    # img = preprocess_image(img)

    # get prediction
    output = model(img)

    classes = [0, 1, 2,3,4,5,6,7,8,9]
    softmax = nn.Softmax(dim=1)
    pred_probs = softmax(model(img)).detach().numpy()[0]
    index = torch.argmax(output, 1)

    result = json.dumps({"label": classes[index], "probability": str(pred_probs[index])})
    return result

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

if __name__ == '__main__':
    # note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the neural network converge faste
#    init()
    model = torch.load('./pytorch_model.pt', map_location=lambda storage, loc: storage)
    model.eval()
    print(type(X_test[0]))
    var_image = Variable(torch.Tensor(X_test[4]))
    
    import matplotlib.pyplot as plt
    # plt.imshow(var_image.reshape(28,28),cmap='gray')
    # plt.show()

    
    pixels = X_test[4].reshape(28, 28, 1)

    # test_img = preprocess_image(pixels)

    # rs = run(test_img)
    # print(rs)
    temp = X_test[4]
    temp=np.reshape(temp,(28,28))
    temp=temp*255
    im = Image.fromarray(temp).convert('L')
    
    im.save("4.png")
    input_data='{"data":"4.png"}'
    img = base64ToImg(json.loads(input_data)['data'])
    img = preprocess_image(img)
    rs = run(img)
    print(rs)
