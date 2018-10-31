from azureml.core import Workspace
from azureml.core.model import Model
import torch
import torch.nn as nn
from torchvision import transforms
import json
import base64
from io import BytesIO
from PIL import Image


ws = Workspace.from_config()
model=Model(ws, 'pytorch-mnist')
model.download(target_dir = '.')
import os 
# verify the downloaded model file
os.stat('./model.pt')


from utils import load_data
X_test = load_data('../data/test-images.gz', False) / 255.0
y_test = load_data('../data/test-labels.gz', True).reshape(-1)

def init():
    global model
    model_path = Model.get_model_path('pytorch-hymenoptera')
    model = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.eval()

def confusion(prediction, truth):
    confusion_vector = prediction / truth

    true_positives = torch.sum(confusion_vector == 1).item()
    false_positives = torch.sum(confusion_vector == float('inf')).item()
    true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
    false_negatives = torch.sum(confusion_vector == 0).item()

    return true_positives, false_positives, true_negatives, false_negatives

if __name__ == '__main__':
    # note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the neural network converge faster

   
    main()

