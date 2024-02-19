import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist , cifar10
from torchvision import models
torch.manual_seed(42)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))
for i in range(9):	
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()
BATCH_SIZE = 256

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) 

torch_X_train = torch_X_train.view(-1,1,28,28).float()
torch_X_train.shape
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

torch_X_test = torch_X_test.view(-1,1,28,28).float()
# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
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
        return F.log_softmax(x)
    
cnn = CNN()
print(cnn)
if torch.cuda.is_available():
    cnn = cnn.cuda()
it = iter(train_loader)
X_batch, y_batch = next(it)
print(X_batch.shape)
print(y_batch.shape)

if torch.cuda.is_available():
    X_batch = X_batch.cuda()
    y_batch = y_batch.cuda()
    
print(cnn.forward(X_batch).shape)
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx == 0:
        print(data.shape)
        print(target.shape)
        
learning_rate = 0.01
momentum = 0.5
n_epochs = 7

optimizer = optim.SGD(cnn.parameters(), lr=learning_rate,momentum=momentum)
train_losses = []
train_counter = []
test_losses = []
def train(epoch):
    cnn.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = cnn(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 25 == 0: #every 25 * batchsize sample we print results
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
      
        train_loss += loss.item()

    train_losses.append(train_loss)
    train_counter.append(epoch)

    torch.save(cnn.state_dict(), 'model.pth')
    torch.save(cnn.state_dict(), 'optimizer.pth')

def test():
  cnn.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = cnn(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset))) 

for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

plt.plot(train_counter, train_losses, color='blue')
plt.plot(range(1,len(test_losses)+1,1), test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('negative log likelihood loss')
#pretrain model
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))
y_train = y_train[:,0]
y_test = y_test[:,0]
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(y_test.shape))
for i in range(9):	
    plt.subplot(330 + 1 + i)
    plt.imshow(X_train[i])
plt.show()
BATCH_SIZE = 256

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor) 

torch_X_train = torch_X_train.reshape(-1,3,32,32).float()
torch_X_train.shape
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)

torch_X_test = torch_X_test.reshape(-1,3,32,32).float()
torch_X_test.shape
# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)
# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx == 0:
        print(data.shape)
        print(target.shape)
        
train_losses = []
train_counter = []
test_losses = []
model = models.vgg16_bn(pretrained=True)
print(model)
class CNN(nn.Module):
    """CNN."""

    def __init__(self):
        """CNN Builder."""
        super(CNN, self).__init__()

        self.conv_layer = model.features

        self.avg_layer = model.avgpool

        self.fc_layer = nn.Sequential(
            nn.Linear(25088, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )


    def forward(self, x):
        """Perform forward."""
        
        # conv layers
        x = self.conv_layer(x)

        # avg layers
        x = self.avg_layer(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        # fc layer
        x = self.fc_layer(x)

        return F.log_softmax(x)

model = CNN()
print(model)
# Freeze model weights
for param in model.conv_layer.parameters():
    param.requires_grad = False

# Freeze model weights
for param in model.avg_layer.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.fc_layer.parameters(), lr=0.000001)
train_losses = []
train_counter = []
test_losses = []
if torch.cuda.is_available():
    model = model.cuda()

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 25 == 0: #every 25 * batchsize sample we print results
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
      
        train_loss += loss.item()

    train_losses.append(train_loss)
    train_counter.append(epoch)

    torch.save(cnn.state_dict(), 'model.pth')
    torch.save(cnn.state_dict(), 'optimizer.pth')

def test():
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

n_epochs = 15
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
plt.plot(train_counter, train_losses, color='blue')
plt.plot(range(1,len(test_losses)+1,1), test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('negative log likelihood loss')
        
       