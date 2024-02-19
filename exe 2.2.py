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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

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
    
    
    
    
    

