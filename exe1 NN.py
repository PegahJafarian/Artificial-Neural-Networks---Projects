import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split
Xdata=pd.read_csv(r'/home/pegah/Desktop/x.csv')
Xdata.head()
Ydata=pd.read_csv(r'/home/pegah/Desktop/y.csv')
Ydata.info()
#sigmoid and its derivation
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

#normalizing the data
for cols in Xdata.columns:
    if Xdata[cols].dtype=='float64':
        Xdata[cols]=((Xdata[cols]-Xdata[cols].mean())/(Xdata[cols].std()))
        
#for cols in Ydata.columns:
    #if Ydata[cols].dtype=='float64':
        #Ydata[cols]=((Ydata[cols]-Ydata[cols].mean())/(Ydata[cols].std()))
        

#split data to train and test
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#weights
w0=2*np.random.random((3526,15))-1
w1=2*np.random.random((15,3525))-1
#learning rate
n=0.1
errors=[]
#train
for i in range(100000):
    #feed forward
    layer0=X_train
    layer1=sigmoid(np.dot(layer0,w0))
    layer2=sigmoid(np.dot(layer1,w1))
#backpropagation using gradient decsent
    layer2_error=y_train-layer2
    layer2_delta=layer2_error*sigmoid_deriv(layer2)
    layer1_error=layer2_delta.dot(w1.T)
    layer1_delta=layer1_error*sigmoid_deriv(layer1)
    w1 +=layer1.T.dot(layer2_delta)*n
    w0 +=layer0.T.dot(layer1_delta)*n
    error=np.mean(np.abs(layer2-error))
    errors.append(error)
    accuracy=(1-error)*100 
#plot the accuracy chart
plt.plot(errors)
plt.xlabel('training')
plt.ylabel('error')
plt.show()
print("training accuracy" +str(round(accuracy,2)) + "%")

#validate
layer0=X_test
layer1=sigmoid(np.dot(layer0,w0))
layer2=sigmoid(np.dot(layer1,w1))   
layer2_error=y_test-layer2
error=np.mean(np.abs(layer2-error))
accuracy=(1-error)*100
print("validation accuracy" +str(round(accuracy,2)) + "%")

