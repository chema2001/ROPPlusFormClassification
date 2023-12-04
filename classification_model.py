import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from PIL import Image
from sklearn.model_selection import train_test_split 
import torchvision
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import io
import torchvision.transforms as transforms


class ROPDataSet(Dataset):
    def __init__(self, file, img_dir, train, transform=None, target_transform=None):

        dataset_metadata = pd.read_csv(file, sep=',', header=0)
        self.dataset_metadata = dataset_metadata

        X_train, X_test, y_train, y_test = train_test_split(dataset_metadata['ImageName'], dataset_metadata[' Class'], test_size=0.20, random_state=42, stratify= dataset_metadata[' Class'])
        
        if train:
            self.x_data, self.y_data = X_train.values, y_train.values
        else:
            self.x_data, self.y_data = X_test.values, y_test.values
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        
        img_code = self.x_data[idx].replace(' ', '')
        img_path = os.path.join(self.img_dir, img_code)


        with open(img_path, 'rb') as f:
            binary_data = f.read()

        # Create an in-memory binary stream
        image_stream = io.BytesIO(binary_data)

        # Open the image using PIL
        image = Image.open(image_stream)
        
       # label = self.y_data.iloc[idx] 
        label = self.y_data[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

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
    
def train_loop(dataloader, model, loss_fn, optimizer, y_loss, y_acc):
    train_loss, train_correct_label = 0,0
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # We put the model into training mode to make sure all the layers can be updated (e.g., dropout and batchnorm)
    model.train()
        
    for batch, data in enumerate(dataloader):
        # Compute prediction and loss
        image = data[0]
        label = data[1]

        out = model(image)
        loss = loss_fn(out, label)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        train_loss += loss.item()

        correct_label = (out.argmax(1) == label).type(torch.float).sum().item()
        train_correct_label += correct_label
    
    train_loss /= num_batches
    train_correct_label /= size

    # TODO: y_loss and y_acc are dictionaries, but you have to create them first, before use
    y_loss['train'].append(train_loss)
    y_acc['train'].append(train_correct_label)

def test_loop(dataloader, model, loss_fn,y_loss,y_acc):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, test_correct_label = 0, 0
    model.eval() 

    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            image = data[0]
            label = data[1]

            out = model(image)
            test_loss += loss_fn(out, label).item()
            test_correct_label += (out.argmax(1) == label).type(torch.float).sum().item()
            
    test_loss /= num_batches
    test_correct_label /= size

    
    # TODO: y_loss and y_acc are dictionaries, but you have to create them first, before use
    y_loss['test'].append(test_loss)
    y_acc['test'].append(test_correct_label)
    
    print(f"Test Error: \n Accuracy Label: {(100*test_correct_label):>0.1f}%,  \n Avg loss: {test_loss:>8f} \n")
    return test_correct_label

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(0,20)),
    torchvision.transforms.ToTensor(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
])

training_data=ROPDataSet(file='RetinalData.csv', img_dir=r'C:\Users\zepin\Desktop\ROPPlusFormClassification\Retinal_Images\Skeletonization_Results\JPG', train=True, transform=train_transforms)
test_data=ROPDataSet(file='RetinalData.csv', img_dir=r'C:\Users\zepin\Desktop\ROPPlusFormClassification\Retinal_Images\Skeletonization_Results\JPG', train=False, transform=test_transforms)

batchSize = 12
class_sample_count = [205, 1084]
weights = 1 /torch.Tensor(class_sample_count)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, batchSize)

train_dataloader = DataLoader(training_data, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

input_size = [224, 244, 1]
hidden_sizes = [128, 64]

model = DenseNet121(channels=input_size[2], height=input_size[0], width=input_size[1], nr_classes=2)

learning_rate = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# TODO: Think about creating this inside the functions or to provided them as arguments
# TODO: The idea is to keep the scope of variables clean
y_loss = {'train':[],'test':[] }  # loss history
y_acc = {'train':[],'test':[]}
x_epoch = []


epochs = 20
best_value = 0
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
        
    train_loop(train_dataloader, model, loss_fn, optimizer,y_loss,y_acc)
    acc = test_loop(test_dataloader, model, loss_fn,y_loss, y_acc)
    x_epoch.append(t+1)

    if acc>=best_value:
        best_value = acc
        torch.save(model.state_dict(), 'multilabel_model.pth')
        print("New best value obtained! Model saved!")
    
print("Done!")

print(y_loss)
print(y_acc)