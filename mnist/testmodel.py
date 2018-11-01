from azureml.core import Workspace
from azureml.core.model import Model

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as Fun
from torch.autograd import Variable

import numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
import os
import json
import base64
from io import BytesIO
from PIL import Image
import urllib.request

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

ws = Workspace.from_config()

model=Model(ws, 'pytorch')
# model.download(target_dir = '.', exists_ok = True)
import os 
# verify the downloaded model file
os.stat('./pytorch_model.pt')


# os.makedirs('./data', exist_ok = True)

# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
# urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')

from utils import load_data
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

def init():
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('pytorch_mnist')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()

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
