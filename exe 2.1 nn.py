#پیاده سازی شبکه traffic sign برای تمرین اول

import os
import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, sampler
from transform import get_train_transforms, get_test_transforms, CLAHE_GRAY
#from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
#import argparse
#from torch import nn
#from model import TrafficSignNet
import torchvision.transforms as transforms
#from data import get_test_loader
from torchvision.utils import make_grid
#from train import valid_batch
#from torch import nn, optim
# To load the picked dataset
#import torch.utils.data.sampler as sampler
from torch import  optim
import PIL
import cv2
#from torchvision import transforms
import random

#loading data
training_file="data/train.p"
validation_file = "data/valid.p"
testing_file = "data/test.p"
with open(training_file , mode="rb") as f:
    train = pickle.load(f)
with open(validation_file , mode="rb") as f:
    valid = pickle.load(f)
with open(testing_file , mode="rb") as f:
    test = pickle.load(f)    
X_train , y_train = train['features'], train['labels']
X_valid , y_valid = valid['features'], valid['labels']
X_test , y_test = test['features'], test['labels']
n_train=len(X_train)
n_valid=len(X_valid)
n_test=len(X_test)
image_shape=X_train.shape
n_classes=len(np.unique(np.append(y_train,y_test)))
print("number of training examples=" ,n_train)
print("number of validation examples=" ,n_valid)
print("number of testing examples=" ,n_test)
print("image data shape=" ,image_shape)
print("number of classes=" ,n_classes)
#to visualize the data
fig , ax=plt.subplots()
ax.bar(range(n_classes),np.bincount(y_train),0.5,color='r')
ax.set_xlabel('signs')
ax.set_ylabel('count')
ax.set_title('the count of each signs')
plt.show()
plt.figure(figsize=(16,16))
for c in range(n_classes):
    i=random.choice(np.where(y_train==c)[0])
    plt.subplot(8,8,c+1)
    plt.axis('off')
    plt.title('class: {}'.format(c))
    plt.imshow(X_train[i])
    
class PickledDataset(Dataset):
    def __init__(self, file_path, transform=None):
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']
            self.labels = data['labels']
            self.count = len(self.labels)
            self.transform = transform

    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform is not None:
            feature = self.transform(feature)
        return (feature, self.labels[index])

    def __len__(self):
        return self.count


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class BaselineNet(nn.module):
    def __init__(self , gray=False):
        super(BaselineNet , self).__init__()
        input_chan=1 if gray else 3
        self.conv1 = nn.Conv2d(input_chan , 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(6,16,5)
        self.fc1=nn.Linear(16*5*5 ,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,43)
    def forward(self , x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1 , 16*5*5)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x

torch.manual_seed(1)
train_dataset = PickledDataset(training_file , transform=transforms.ToTensor())    
valid_dataset = PickledDataset(validation_file , transform=transforms.ToTensor())   
test_dataset = PickledDataset(testing_file , transform=transforms.ToTensor()) 
train_loader=DataLoader(train_dataset , batch_size=64 , shuffle=True)
valid_loader=DataLoader(valid_dataset , batch_size=64 , shuffle=False)
test_loader=DataLoader(test_dataset , batch_size=64 , shuffle=False)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_device(x,y):
    return x.to(device),y.to(device , dtype=torch.int64)

train_loader=WrappedDataLoader(train_loader , to_device)
valid_loader=WrappedDataLoader(valid_loader , to_device)
test_loader=WrappedDataLoader(test_loader , to_device) 

model=BaseLineNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(), lr=0.001 , momentum=0.9)
n_epochs=20

def loss_batch(model, loss_func, x, y, opt=None):
    loss = loss_func(model(x), y)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(x)

def valid_batch(model, loss_func, x, y):
    output = model(x)
    loss = loss_func(output, y)
    pred = torch.argmax(output, dim=1)
    correct = pred == y.view(*pred.shape)

    return loss.item(), torch.sum(correct).item(), len(x)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        # Train model
        model.train()
        losses, nums = zip(
            *[loss_batch(model, loss_func, x, y, opt) for x, y in train_dl])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # Validation model
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(
                *[valid_batch(model, loss_func, x, y) for x, y in valid_dl])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            print(f"[epoch{epoch+1}/{epochs}]"
                  f"train loss:{train_loss:.6f}\t"
                  f"validation loss:{valid_loss:.6f}\t",f"validation accuracy:{valid_accuracy:.3f}%")


def evaluate(model, loss_func, dl):
    model.eval()
    with torch.no_grad():
        losses, corrects, nums = zip(
            *[valid_batch(model, loss_func, x, y) for x, y in dl])
        test_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        test_accuracy = np.sum(corrects) / np.sum(nums) * 100

        print(f"Test loss: {test_loss:.6f}\t"
              f"Test accruacy: {test_accuracy:.3f}%")

fit(n_epochs , model, criterion , optimizer , train_loader, valid_loader)
evaluate(model, criterion , test_loader)

class CLAHE_GRAY:
    def __init__(self, clipLimit=2.5, tileGridSize=(4, 4)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def __call__(self, im):
        img_y = cv2.cvtColor(im, cv2.COLOR_RGB2YCrCb)[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=self.clipLimit,
                                tileGridSize=self.tileGridSize)
        img_y = clahe.apply(img_y)
        img_output = img_y.reshape(img_y.shape + (1,))
        return img_output

clahe=CLAHE_GRAY()
plt.figure(figsize=(16,16))
for c in range(n_classes):
    i=random.choice(np.where(y_train==c)[0])
    plt.subplot(8,8,c+1)
    plt.axis('off')
    plt.title('class:{}'.format(c))
    plt.imshow(clahe(X_train[i].squeeze(),cmap='gray'))

data_transforms=transforms.Compose([CLAHE_GRAY(),transforms.ToTensor()])
train_dataset = PickledDataset(training_file , transform=data_transforms)    
valid_dataset = PickledDataset(validation_file , transform=data_transforms)   
test_dataset = PickledDataset(testing_file , transform=data_transforms) 
train_loader=WrappedDataLoader(DataLoader(train_dataset , batch_size=64 , shuffle=True),to_device)
valid_loader=WrappedDataLoader(DataLoader(valid_dataset , batch_size=64 , shuffle=False),to_device)
test_loader=WrappedDataLoader(DataLoader(test_dataset , batch_size=64 , shuffle=False),to_device)
model=BaselineNet(gray=True).to(device)
optimizer=optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
fit(n_epochs , model, criterion , optimizer , train_loader , valid_loader)
evaluate(model,criterion , test_loader)

def preprocess(path):
    if not os.path.exists(f"{path}/train_gray.p"):
        for dataset in ['train', 'valid', 'test']:
            with open(f"{path}/{dataset}.p", mode='rb') as f:
                data = pickle.load(f)
                X = data['features']
                y = data['labels']

            clahe = CLAHE_GRAY()
            for i in range(len(X)):
                X[i] = clahe(X[i])

            X = X[:, :, :, 0]
            with open(f"{path}/{dataset}_gray.p", "wb") as f:
                pickle.dump({"features": X.reshape(
                    X.shape + (1,)), "labels": y}, f)

preprocess('data')
training_file='data/train_gray.p'
validation_file='data/valid_gray.p'
testing_file='data/test_gray.p'   
train_dataset = PickledDataset(training_file , transform=transforms.ToTensor())    
valid_dataset = PickledDataset(validation_file , transform=transforms.ToTensor())   
test_dataset = PickledDataset(testing_file , transform=transforms.ToTensor())
train_loader=WrappedDataLoader(DataLoader(train_dataset , batch_size=64 , shuffle=True),to_device)
valid_loader=WrappedDataLoader(DataLoader(valid_dataset , batch_size=64 , shuffle=False),to_device)
test_loader=WrappedDataLoader(DataLoader(test_dataset , batch_size=64 , shuffle=False),to_device)
                          
def extend_dataset(dataset):
    X = dataset.features
    y = dataset.labels
    num_classes = 43

    X_extended = np.empty([0] + list(dataset.features.shape)
                          [1:], dtype=dataset.features.dtype)
    y_extended = np.empty([0], dtype=dataset.labels.dtype)

    horizontally_flippable = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]
    vertically_flippable = [1, 5, 12, 15, 17]
    both_flippable = [32, 40]
    cross_flippable = np.array([
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38]
    ])

    for c in range(num_classes):
        X_extended = np.append(X_extended, X[y == c], axis=0)

        if c in horizontally_flippable:
            X_extended = np.append(
                X_extended, X[y == c][:, :, ::-1, :], axis=0)
        if c in vertically_flippable:
            X_extended = np.append(
                X_extended, X[y == c][:, ::-1, :, :], axis=0)
        if c in cross_flippable[:, 0]:
            flip_c = cross_flippable[cross_flippable[:, 0] == c][0][1]
            X_extended = np.append(
                X_extended, X[y == flip_c][:, :, ::-1, :], axis=0)
        if c in both_flippable:
            X_extended = np.append(
                X_extended, X[y == c][:, ::-1, ::-1, :], axis=0)

        y_extended = np.append(y_extended, np.full(
            X_extended.shape[0]-y_extended.shape[0], c, dtype=y_extended.dtype))

    dataset.features = X_extended
    dataset.labels = y_extended
    dataset.count = len(y_extended)

    return dataset

train_dataset=extend_dataset(train_dataset)
train_loader=WrappedDataLoader(DataLoader(train_dataset , batch_size=64 , shuffle=True),to_device)
fig , ax=plt.subplots()
ax.bar(range(n_classes),np.bincount(y_train),0.5,color='r')
ax.set_xlabel('signs')
ax.set_ylabel('count')
ax.set_title('the count of each signs')
plt.show()
plt.figure(figsize=(16,16))
for c in range(n_classes):
    i=random.choice(np.where(y_train==c)[0])
    plt.subplot(8,8,c+1)
    plt.axis('off')
    plt.title('class: {}'.format(c))
    plt.imshow(train_dataset.features[i].squeeze(),cmap='gray')

model=BaseLineNet(gray=True).to(device)
optimizer=optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
fit(n_epochs , model, criterion , optimizer , train_loader , valid_loader)
evaluate(model,criterion , test_loader)
train_dataset = extend_dataset(PickledDataset(training_file))
class_sample_count=np.bincount(train_dataset.labels)
weights=1/np.array([class_sample_count[y] for y in train_dataset.labels])
samp=sampler.WeightedRandomSampler(weights, 43*20000)
train_loader=WrappedDataLoader(DataLoader(train_dataset ,batch_size=64 , sampler=samp),to_device)
balanced_y_train=torch.LongTensor([]).to(device)
with torch.no_grad():
    for _,y in train_loader:
        balanced_y_train=torch.cat((balanced_y_train, y))
fig, ax=plt.subplots()
ax.bar(range(n_classes),np.bincount(balanced_y_train.cpu().numpy()) ,0.5 ,color='r')
ax.set_xlabel('signs')
ax.set_ylabel('counts')  
ax.set_title('the count of each sign')
plt.show()
train_data_transforms=transforms.Compose([transforms.ToPILImage(),transforms.RandomRotation(20,resample=PIL.Image.BICUBIC),transforms.RandomAffine(0,translate=(0.2,0.2), resample=PIL.Image.BICUBIC),transforms.RandomAffine(0,shear=20,resample=PIL.Image.BICUBIC),transforms.RandomAffine(0,scale=(0.8,1.2),resample=PIL.Image.BICUBIC)]),transforms.ToTensor()
test_data_transforms=transforms.ToTensor()
train_dataset = extend_dataset(PickledDataset(training_file, transform=train_data_transforms))
valid_dataset=PickledDataset(validation_file, transform=test_data_transforms) 
test_dataset=PickledDataset(testing_file, transform=test_data_transforms)

train_loader=WrappedDataLoader(DataLoader(train_dataset,batch_size=64 , sampler=samp),to_device) 
valid_loader=WrappedDataLoader(DataLoader(valid_dataset,batch_size=64 , shuffle=False),to_device)  
test_loader=WrappedDataLoader(DataLoader(test_dataset,batch_size=64 , shuffle=False),to_device) 

def convert_image_np(img):
    img = img.numpy().transpose((1, 2, 0)).squeeze()
    return img

with torch.no_grad():
    x,y=next(iter(train_loader))
    plt.figure(figsize=(16,16))
    for i in range(len(y)):
        plt.subplot(8,8,i+1)
        plt.axis('off')
        plt.title('class:{}'.format(y[i]))
        plt.imshow(convert_image_np(x[i].cpu()),cmap='gray')

model=BaseLineNet(gray=True).to(device)
optimizer=optim.SGD(model.parameters(),lr=0.001, momentum=0.9)
fit(n_epochs , model, criterion , optimizer , train_loader , valid_loader)
evaluate(model,criterion , test_loader)

# the model
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return F.log_softmax(x)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl, patience, checkpoint):
    wait = 0
    valid_loss_min = np.Inf
    for epoch in range(epochs):
        
        model.train()
        losses, nums = zip(
            *[loss_batch(model, loss_func, x, y, opt) for x, y in train_dl])
        train_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        
        model.eval()
        with torch.no_grad():
            losses, corrects, nums = zip(
                *[valid_batch(model, loss_func, x, y) for x, y in valid_dl])
            valid_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            valid_accuracy = np.sum(corrects) / np.sum(nums) * 100
            print(f"[epoch{epoch+1}/{epochs}]"
                  f"- Train loss: {train_loss:.6f}\t"
                  f"Validation loss: {valid_loss:.6f}\t",
                  f"Validation accruacy: {valid_accuracy:.3f}%")
           
            if valid_loss <= valid_loss_min:
                print(
                    f"- Validation loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...")
                torch.save(model.state_dict(), 'model.pt')
                valid_loss_min = valid_loss
                wait = 0
            
            else:
                wait += 1
                if wait >= patience:
                    print(
                        f"Terminated Training for Early Stopping at Epoch {epoch+1}")
                    return

n_epochs=100
model=TrafficSignNet().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.0001)
fit(n_epochs ,model , criterion , optimizer , train_loader , valid_loader)
check_point=torch.load('model.pt',map_location=device) 
model.load_state_dict(check_point)
evaluate(model , criterion , test_loader)

class Stn(nn.Module):
    def __init__(self):
        super(Stn, self).__init__()
        self.loc_net = nn.Sequential(
            nn.Conv2d(1, 50, 7),
            nn.MaxPool2d(2, 2),
            nn.ELU(),
            nn.Conv2d(50, 100, 5),
            nn.MaxPool2d(2, 2),
            nn.ELU()
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(100 * 4 * 4, 100),
            nn.ELU(),
            nn.Linear(100, 3 * 2)
        )
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.loc_net(x)
        xs = xs.view(-1, 100 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x 
# the model with stn
class TrafficSignNet(nn.Module):
    def __init__(self):
        super(TrafficSignNet, self).__init__()
        self.stn = Stn()
        self.conv1 = nn.Conv2d(1, 100, 5)
        self.conv1_bn = nn.BatchNorm2d(100)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(100, 150, 3)
        self.conv2_bn = nn.BatchNorm2d(150)
        self.conv3 = nn.Conv2d(150, 250, 1)
        self.conv3_bn = nn.BatchNorm2d(250)
        self.fc1 = nn.Linear(250 * 3 * 3, 350)
        self.fc1_bn = nn.BatchNorm1d(350)
        self.fc2 = nn.Linear(350, 43)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.stn(x)
        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(self.conv1_bn(x))
        x = self.pool(F.elu(self.conv2(x)))
        x = self.dropout(self.conv2_bn(x))
        x = self.pool(F.elu(self.conv3(x)))
        x = self.dropout(self.conv3_bn(x))
        x = x.view(-1, 250 * 3 * 3)
        x = F.elu(self.fc1(x))
        x = self.dropout(self.fc1_bn(x))
        x = self.fc2(x)
        return F.log_softmax(x)
model=TrafficSignNet().to(device)
optimizer=optim.Adam(model.parameters(),lr=0.0001)
fit(n_epochs ,model , criterion , optimizer , train_loader , valid_loader)

check_point=torch.load('model.pt',map_location=device) 
model.load_state_dict(check_point)
evaluate(model , criterion , test_loader)

def visualize_stn(dl, outfile):
    with torch.no_grad():
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_tensor = model.stn(data).cpu()

        input_grid = convert_image_np(make_grid(input_tensor))
        transformed_grid = convert_image_np(make_grid(transformed_tensor))

        
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches((16, 16))
        ax[0].imshow(input_grid)
        ax[0].set_title('Dataset Images')
        ax[0].axis('off')

        ax[1].imshow(transformed_grid)
        ax[1].set_title('Transformed Images')
        ax[1].axis('off')

        plt.savefig(outfile)
visualize_stn()        


#پیاده سازی شبکه های ANN و CNN تمرین دوم

import pandas as pd
import numpy as np
import seaborn as sns
#from PIL import Image
import cv2
import os
cnt=1
for dirname,_,filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if(cnt==2):
            break;
        print(os.path.join(dirname,filename))
        cnt+=1
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
#from torch import nn
import torch.nn.functional as F
from torch import optim
#import torch.optim as optim
from torchvision import models     
from sklearn.model_selection import train_test_split
#from torch.optimizers import Adam
data=pd.read_csv(r'/home/pegah/Desktop/socal2.csv')
#dataset = datasets.ImageFolder(root='./classify/dataset/training_set/, 
#transform = transforms.ToTensor())  
#loader = data.DataLoader(dataset, batch_size = 8, shuffle = True)
#data=pd.read_csv('/kaggle/input/house-prices-and-images-socal/socal2.csv')
print(data.head())
X_house_attributes=data[['n_citi','bed','bath','sqft','price']]
print(X_house_attributes)
print(X_house_attributes.shape)

plt.scatter(data.price,data.sqft)
plt.title("price vs square feet")
plt.scatter(data.bed,data.price)
plt.title("bedroom and price")
plt.xlabel("bedrooms")
plt.ylabel("price")
plt.show()
sns.despine
X_features=data[['n_citi','bed','bath','sqft','price']]
print(X_features)
print(X_features.shape)
#data preprocessing
b_m=max(X_features['bed'])
sqft_m=max(X_features['sqft'])
price_m=max(X_features['price'])
bath_m=max(X_features['bath'])
citi_m=max(X_features['citi'])
X_features['n_citi']=X_features['n_citi']/citi_m
X_features['bed']=X_features['bed']/b_m
X_features['sqft']=X_features['sqft']/sqft_m
X_features['bath']=X_features['bath']/bath_m
X_features['price']=X_features['price']/price_m
#resize the images
sample=cv2.imread('/kaggle/input/house-prices-and-images-socal/socal2/socal-pics/1.jpg')
plt.imshow(sample)
sample_resized=cv2.resize(sample,(64,64))
plt.imshow(sample_resized)
#import os
#import cv2
cnt=0
images_path='kaggle/input/house-prices-and-images-socal/socal2/socal_pics'
X_house_images=np.zeros((15474,64,64,3),dtype='uint32')
for i in range(15474):
    sample=cv2.imread(images_path+'/'+str(i)+'.jpg')
    imgs=cv2.resize(sample,(64,64))
    X_house_images[cnt]=imgs
    cnt+=1
print("No.of images:" , cnt)
X_house_images=X_house_images/255

# model definition
class Ann_Net(nn.Module):
    # define model elements
    def __init__(self, n_inputs):
        super(Ann_Net, self).__init__()
        self.layer = nn.Linear(n_inputs, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self,state):
        x = F.relu(self.layer(state))
        outputs = self.activation(x)
        return outputs
    # forward propagate input
    #def forward(self, X):
        #X = self.layer(X)
        #X = self.activation(X)
        #return X


class Cnn_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=1)

    def forward(self, t):
    # (1) input layer
        t = t

    # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

    # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

    # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

    # (6) output layer
        t = self.out(t)
    #t = F.softmax(t, dim=1)

        return t
  
#splitting into train and test
split=train_test_split(X_features , X_house_images , test_size=0.2 , random_state=42)
(Xatt_train , Xatt_test , Ximage_train , Ximage_test)=split
y_train, y_test=Xatt_train['price'].values , Xatt_test['price'].values
X1_train=Xatt_train[['n_citi','bed','bath','sqft']].values
X2_train=Ximage_train
X1_test=Xatt_test[['n_citi','bed','bath','sqft']].values
X2_test=Ximage_test
print(X1_train.shape)
print(X1_test.shape)
print(X2_train.shape)
print(X2_test.shape)
print(y_train.shape)
print(y_test.shape)

#a model
mlp=Ann_Net(X1_train.shape[1],regress=False)
cnn=Cnn_Net(64,64,3,regress=False)
#combine 2 outputs
class TwoInputsNet(nn.Module):
  def __init__(self):
    super(TwoInputsNet, self).__init__()
    self.conv = nn.Conv2d( in_channels=1, out_channels=6, kernel_size=5) 
    self.fc1 = nn.Linear( in_features=12 * 4 * 4, out_features=120 ) 
    self.fc2 = nn.Linear(in_features=120, out_features=60 )  

  def forward(self, input1, input2):
    c = self.conv(input1)
    f = self.fc1(input2)
    # now we can reshape `c` and `f` to 2D and concat them
    combined = torch.cat((c.view(c.size(0), -1),
                          f.view(f.size(0), -1)), dim=1)
    out = self.fc2(combined)
    return out
#combinedInput=torch.cat([mlp.output , cnn.output])
x=nn.Linear(4,activation="relu")( TwoInputsNet)
x=nn.Linear(1,activation="linear")(x)

Model=models(inputs=[mlp.input , cnn.input],outputs=x)
opt=optim.Adam(ler=1e-3 , decay=1e-3/200)
Model.compile(loss="mse",optimizer=opt)
#model = models.vgg16()
#print(model)
print("[info] training model:")
Model.fit(x=[X1_train , X2_train],y=y_train , validation_data=([X1_test,X2_test],y_test),epochs=50,batch_size=64)
#predict for test data that is new
attr_sample=data.loc[data['image_id']==4]
print(attr_sample)
image_sample=cv2.imread('/kaggle/input/house-prices-and-images-socal/socal2/socal_pics/4.jpg')
sample_resized=cv2.resize(image_sample,(64,64))
plt.imshow(sample_resized)
#preprocess for getting the predicted model
X1_final=np.zeros(4,dtype='float32')
X1_final[0]=attr_sample['n_citi']/citi_m
X1_final[1]=attr_sample['bed']/b_m
X1_final[2]=attr_sample['bath']/bath_m
X1_final[3]=attr_sample['sqft']/sqft_m
y_ground_truth=attr_sample['price']
X2_final=sample_resized/255.0
print(X1_final.shape," ",X2_final.shape)
y_pred=Model.predict([np.reshape(X1_final,(1,4)),np.reshape(X2_final,(1,64,64,3))])
print("actual price:",attr_sample['price'].values)
print("predicted price:" , y_pred*price_m)
    
    
    
    
    

