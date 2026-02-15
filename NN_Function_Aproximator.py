import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


'''
Function Aproximator
Marco Mas Sempre
'''


'''
Function to approximate
'''

def func(x):
    return torch.sin(torch.pi*x)


n = 10                      #Number of intermediate neurons



'''
Define the network
'''

class Red(nn.Module):
    
    def __init__(self, n_in):
        super(Red, self).__init__()
        self.linear1 = nn.Linear(n_in, n)
        self.linear2 = nn.Linear(n, 1)
        
    def forward(self, inputs):
        pred_1 = torch.sigmoid(input=self.linear1(inputs))
        pred_f = self.linear2(pred_1)
        return pred_f
        

'''
Define the training parameters and train the network
'''

lr = 0.01
epochs = 1500
estatus_print = 10
losses = np.zeros(epochs)
E = np.zeros(epochs)
loss_fun = nn.L1Loss()
model = Red(1)
optimizer = torch.optim.Adam(params = model.parameters(), lr = lr)
X = torch.rand(30,1)

'''
Training loop
'''

for epoch in range(1,epochs +1):
    if epoch % estatus_print == 0:
        X = torch.rand(30,1)
    Y = func(X)
    Y_pred = model(X)
    
    loss = loss_fun(Y,Y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if epoch % estatus_print == 0:
        print("Epoch {} = {}".format(epoch,loss))
    
    with torch.no_grad():
        if epoch == epochs:
            #X = torch.rand(100,1)
            X = torch.rand(100,1)
            for x in range(100):
                X[x][0]= torch.linspace(start = 0, end = 1,steps= 100)[x]
            Y = func(X).numpy()
            Y_pred=model(X).numpy() 
            X = X.numpy()
            
            plt.figure()
            plt.plot(X,Y, "o",label = "Exacto")
            plt.plot(X,Y_pred, ".",label = "Usando la red")
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('$f(x) = sen(\pi x)$')
            plt.show()
            
    E[epoch-1] = epoch
    losses[epoch-1] = loss.item()
    
    
'''
Plot the loss function
'''

plt.figure()
plt.plot(E,losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

