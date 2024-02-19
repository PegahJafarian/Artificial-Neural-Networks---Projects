#تمرین بخش اول  autoencoder for image coloring
import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
import skimage
from PIL import Image
from skimage.io import imread,imshow,imread_collection,concatenate_images
from skimage.transform import resize
from skimage.util import crop,pad
from skimage.morphology import label
from skimage.color import rgb2gray,gray2rgb,rgb2lab,lab2rgb
from sklearn.model_selection import train_test_split
import torch
from torch import nn,optim
from torch.nn import functional as F
from torchvision import datasets ,transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore',category=UserWarning,module='skimage')
seed=42
random.seed=seed
np.random.seed=seed
epochs=50
batch_size=64
log_interval=50
#data
image_width=256
image_hight=256
image_channels=3
input_shape=(image_hight,image_width,1)
#train_path='/home/pegah/Desktop/art-images/dataset/dataset_updated/training_set/painting'
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform=transforms.ToTensor()
train_data='/home/pegah/Desktop/art-images/dataset/dataset_updated/training_set/painting'
validation_data='/home/pegah/Desktop/art-images/dataset/dataset_updated/validation_set/painting'
num_worker=0
batch_size=20
train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_worker=num_worker)
validation_loader=torch.utils.data.DataLoader(validation_data, batch_size=batch_size,num_worker=num_worker)
#for image in enumerate(images):
 #   try:
  #      Image.open('/home/pegah/Desktop/art-images/dataset/dataset_updated/training_set/painting/'+image[1])
   # except OSError:
    #    os.remove('/home/pegah/Desktop/art-images/dataset/dataset_updated/training_set/painting/'+image[1])
#def imshow(img):
    #img=img/2 + 0.5
    #plt.imshow(np.transpose(img,(1,2,0)))
#classes=['sculpture','engraving','drawings','painting','iconography']
#dataiter=iter(train_loader)
#images,labels=dataiter.next()
#images=images.numpy()
#fig=plt.figure(figsize=(25,4))
#for idx in np.arange(20):
    #ax=fig.add_subplot(2,20/2,idx+1,xticks=[],yticks=[])
    #imshow(images[idx])
    #ax.set_title(classes[labels[idx]])

#rgb_img=np.squeeze(images[3])
#channels=['red channel','green channel','blue channel']
#fig=plt.figure(figsize=(36,36))
#for idx in np.arange(rgb_img.shape[0]):
    #ax=fig.add_subplot(1,3,idx+1)
    #img=rgb_img[idx]
    #ax.imshow(img,cmap='gray')
    #ax.set_title(channels[idx])
    #width, height=img.shape
    #thresh=img.max()/2.5
    #for x in range(width):
     #   for y in range(height):
      #      val=round(img[x][y],2) if img[x][y]!=0 else 0
       #     ax.annotate(str(val),xy=(x,y),horizontalalignment='center', verticalalignment='center',size=8, color='white' if img[x][y]<thresh else 'black')

train_ids=next(os.walk(train_data))[2]
X_train=np.zeros((len(train_ids)-86,image_hight,image_width,image_channels),dtype=np.uint8)
missing_count=0
print('getting train images')
sys.stdout.flush()
for n,id_ in tqdm(enumerate(train_ids),total=len(train_ids)):
    path=train_data+id_+''
    try:
        img=imread(path)
        img=resize(img,(image_hight,image_width),mode='constant',preserve_range=True)
        X_train[n-missing_count]=img
    except:
        missing_count+=1
            
X_train=X_train.astype('float32')/255
print("total missing:"+str(missing_count))
imshow(X_train[5])
plt.show()
#split the data
x_train,x_test=train_test_split(X_train,test_size=20,random_state=seed)
#transform = transforms.ToTensor()
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
       
        #Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 3, 2, stride=2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
              
        return x


model = ConvAutoencoder()
print(model)

#class customLoss(nn.Module):
 #   def __init__(self):
  #      super(customLoss,self).__init__()
   #     self.mse_loss=nn.MSELoss(reduction="sum")
        
    #def forward(self , x_recon, x , mu , logvar):
     #   loss_MSE=self.mse_loss(x_recon , x)
      #  loss_KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
       # return loss_MSE + loss_KLD

#Loss function
#criterion1 = nn.MAELoss()
criterion=nn.MSELoss()

#Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer1=torch.optim.RMSprop(model.parameters(),lr=0.001)
#oprimizer2=torch.optim.Rprop(model.parameters(),lr=0.001)
#optimizer10=torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer11=torch.optim.RMSprop(model.parameters(),lr=0.0001)
#optimizer21=torch.optim.Rprop(model.parameters(),lr=0.0001)
#loss_mse=customLoss()
#def get_device():
 #   if torch.cuda.is_available():
  #      device = 'cuda:0'
   # else:
    #    device = 'cpu'
    #return device

#device = get_device()
#print(device)
#model.to(device)
val_losses=[]
train_losses=[]
def train(epoch):
    model.train()
    train_loss=0 
    for batch_inx , (data , _) in enumerate(train_loader):
        data=data.to(device)
        optimizer.zero_grad()
        recon_batch , mu, logvar=model(data)
        loss=nn.MSELoss(recon_batch , data , mu , logvar)
        loss.backward()
        train_loss+=loss.item()
        optimizer.step()
        if batch_inx%log_interval==0:
            print('train epoch:{}[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(epoch,batch_inx*len(data),len(train_loader),100.*batch_inx/len(train_loader),loss.item()/len(data)))
            print('===>epoch:{}average loss:{:.4f}'.format(epoch,train_loss/len(train_loader)))
            train_losses.append(train_loss/len(train_loader))
            
def test(epoch):
    model.eval()
    test_loss=0
    with torch.no_grad():
        for i , (data,_) in enumerate(validation_loader):
            data=data.to(device)
            recon_batch, mu,logvar=model(data)
            test_loss +=nn.MSELoss(recon_batch , data, mu,logvar).item()
            if i==0:
                n=min(data.size(0),8)
                comparison=torch.cat([data[:n],recon_batch.view(batch_size,3,100,100)[:n]])
                save_image(comparison.cpu(),'/home/pegah/Desktop/results/reconstruction_'+str(epoch)+'.png',nrow=n)
                test_loss/=len(validation_loader)
                print('===>test set loss:{:.4f}'.format(test_loss))
                val_losses.append(test_loss)
                
for epoch in range(1,epochs+1):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample=torch.randn(64,2048).to(device)
        sample=model.decode(sample).cpu()
        save_image(sample.view(64,3,100,100),'/home/pegah/Desktop/results/sample_'+str(epoch)+'.png')

plt.figure(figsize=(15,10))
plt.plot(range(len(train_losses)),train_losses)
plt.plot(range(len(val_losses)),val_losses)
plt.title("validation loss and loss per epoch",fontsize=18)
plt.xlabel("epoch",fontsize=18)
plt.ylabel("loss",fontsize=18)
plt.legend(['training loss','validation loss'],fontsize=14)
plt.show()                

