import torch
from captum.attr import Saliency
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import io
from torch import nn
import torchvision

class DenseNet121(torch.nn.Module):
    def __init__(self, channels, height, width, nr_classes = 3):
        super(DenseNet121, self).__init__()

        # Init variables
        self.channels = channels
        self.height = height
        self.width = width
        self.nr_classes = nr_classes

        # Init modules
        # Backbone to extract features
        self.densenet121 = torchvision.models.densenet121(pretrained=False).features

        self.densenet121.conv0 = nn.Conv2d(1,64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        # FC-Layers
        # Compute in_features
        _in_features = torch.rand(1, self.channels, self.height, self.width)
        _in_features = self.densenet121(_in_features)
        print(_in_features.size())
        _in_features = _in_features.size(0) * _in_features.size(1) * _in_features.size(2) * _in_features.size(3)

        # Create FC1 Layer for classification
        self.fc1 = torch.nn.Linear(in_features=_in_features, out_features=self.nr_classes)
        return
    

    def forward(self, inputs):
        # Compute Backbone features
        features = self.densenet121(inputs)
        # Reshape features
        features = torch.reshape(features, (features.size(0), -1))
        # FC1-Layer
        out1 = self.fc1(features)
        return out1

def preprocess(image, size=224):
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
    ])
    return transform(image)

image_nr = 3
file='RetinalData.csv'
img_dir=r'C:\Users\zepin\Desktop\ROPPlusFormClassification\Retinal_Images\Skeletonization_Results\JPG'
dataset_metadata = pd.read_csv(file, sep=',', header=0)

img_code = dataset_metadata['ImageName'][image_nr].replace(' ', '')
label = dataset_metadata[' Class'][image_nr]

img_path = os.path.join(img_dir, img_code)

with open(img_path, 'rb') as f:
    binary_data = f.read()

# Create an in-memory binary stream
image_stream = io.BytesIO(binary_data)

# Open the image using PIL
image = Image.open(image_stream)
# Read the saved model 
input_size = [224, 224, 1]
model = DenseNet121(channels=input_size[2], height=input_size[0], width=input_size[1], nr_classes=3)
#model = DenseNet(num_classes=3)
model.load_state_dict(torch.load('multilabel_model.pth', map_location=torch.device('cpu')))
model.eval()
dl = Saliency(model)


X = preprocess(image)
X.requires_grad_()
scores = model(X.unsqueeze(0))
pred = scores.argmax(1)
print("Correct Label: " + str(label) + " Pred Label: " + str(pred.item()))
target = torch.tensor([label])
saliency = dl.attribute(X.unsqueeze(0), target=target)
plt.subplot(1, 2, 1)
X = X.permute(2, 1, 0)
plt.imshow(X.detach().numpy()) # just squeeze classes dim, because we have only one class
plt.title("Image")
plt.axis("off")

plt.subplot(1,2,2)
saliency = saliency.squeeze(0)
saliency = saliency.permute(2,1,0)
saliency = saliency.detach().numpy()
saliency = np.multiply(saliency, 1/np.amax(saliency))
plt.imshow(saliency)
plt.title("Map")
plt.axis('off')
plt.show()