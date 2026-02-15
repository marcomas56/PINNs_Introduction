import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable


'''
PINN to solve the forced harmonic oscillator with
mixed methods (Adam + LBFGS) and Forzed CC. 
Study of the colapse of neurons and its relation with the loss.
Marco Mas Sempre
'''


'''
EDO parameters
'''

w = 6*np.pi
A = 1
g = 2



x0 = 0
x1 = w*A


'''
Definition of the analytic solution
'''

def sol(t):
    return A*np.exp(-g*t)*np.sin(np.sqrt(w**2-g**2)*t)



'''
Definition of the network structure
'''

class Red(nn.Module):
    
    def __init__(self,n):
        super(Red, self).__init__()
        self.Linear1 =  nn.Linear(1, n)
        self.Linear2 = nn.Linear(n, 1)
        
    def forward(self, t):
        pred_1 = torch.tanh(input=self.Linear1(t))

        return self.Linear2(pred_1)
    
    
'''
Definition of the function to initialize the weights
'''

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)



'''
Forcing the network to satisfy the necessary conditions(Forced CC)
'''

def fx(t):
    Nt = net(t)
    return x0 + x1*t + Nt *t*t



'''
Definition of the loss function
'''

def compute_loss(net,t):
    x = fx(t)
    f_out = f(x,t,g,w)
    zero = Variable(torch.zeros(N), requires_grad = True).view(-1, 1)
    return loss_fun(f_out,zero)



'''
Auxiliary function for LBFGS
'''


def closure():
    optimizer.zero_grad()
    loss = compute_loss(net,t)
    loss.backward()

    return loss



'''
Definition of the ODE
'''


def f(x,t,g,w):
    dxdt = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    dxdt2 = torch.autograd.grad(dxdt, t, grad_outputs=torch.ones_like(dxdt), create_graph=True)[0]
    return dxdt2 + 2*g*dxdt + (w**2)*x


'''
Function to calculate the colapse of neurons
'''

def colap(b,w):
    n = len(b)
    ind = []
    for i in range(n):
        if np.abs(np.abs(b[i]) + np.abs(w[i])) > 6 and np.abs(b[i]) > 6 and ( (-b[i]/w[i]) > 1 or (-b[i]/w[i]) < 0):
            ind.append(i)
    return ind



'''
Necessary lists to save the values we need
'''

colaps = []
L = []



'''
Training loop for different number of neurons.
'''

ns = [40,60,80,100,120,140,160,180,200]
for n in ns:
    

    '''
    Initializing the NN for n neurons
    '''

    print(n)
    
    net = Red(n)
    net.apply(init_weights)
    loss_fun = nn.MSELoss(reduction = 'mean')
    
    
    
    
    '''
    Defining more necesary lists
    '''
    Ki1 = []
    Bi1 = []
    Ki2 = []
    Bi2 = []
    Ki3 = []
    Bi3 = []
    losses = []
    E = []

    

    '''
    First training with ADAM
    '''
    

    lr = 0.01
    epochsAdam =1000
    changebach = 100
    estatus_print = 50
    N = 2000
    losscup = 25
    
    optimizer = optim.Adam(params = net.parameters(), lr = lr)
    
    t0 = torch.linspace(0,1,N)
    t = Variable(t0, requires_grad = True).view(-1, 1)
    

    
    for epoch in range(1,epochsAdam +1):
        if epoch % changebach == 0:
            t0 = torch.rand(N)
            t0[0] = 0
            t = Variable(t0, requires_grad = True).view(-1, 1)
            
        optimizer.zero_grad()
        loss = compute_loss(net,t)
        loss.backward()
        optimizer.step()
        
        E.append(epoch)
        losses.append(loss.item())
        
        
        
        kernels = net.Linear1.weight.cpu().detach().clone().numpy()
        Ki1.append(kernels)
        
        kernels = net.Linear1.bias.cpu().detach().clone().numpy()
        Bi1.append(kernels)
        
        kernels = net.Linear2.weight.cpu().detach().clone().numpy()
        Ki2.append(kernels)
        
        kernels = net.Linear2.bias.cpu().detach().clone().numpy()
        Bi2.append(kernels)
        
    
    
    Ultepoch = epoch
    
    
    
    '''
    Second training with LBFGS
    '''
    
    lr = 1
    epochsLBFGS =1000
    changebach = 100
    estatus_print = 50
    N = 2000
    losscup2 = 0
    
    optimizer = optim.LBFGS(params = net.parameters(), lr = lr,line_search_fn="strong_wolfe")
    
    t0 = torch.linspace(0,1,N)
    t = Variable(t0, requires_grad = True).view(-1, 1)
      
    for epoch in range(Ultepoch + 1,epochsLBFGS + Ultepoch +1):
        if epoch % changebach == 0:
            t0 = torch.rand(N)
            t0[0] = 0
            t = Variable(t0, requires_grad = True).view(-1, 1)
            
        optimizer.step(closure)
    
        loss = compute_loss(net,t)
        E.append(epoch)
        losses.append(loss.item())
        
        
        
        kernels = net.Linear1.weight.cpu().detach().clone().numpy()
        Ki1.append(kernels)
        
        kernels = net.Linear1.bias.cpu().detach().clone().numpy()
        Bi1.append(kernels)
        
        kernels = net.Linear2.weight.cpu().detach().clone().numpy()
        Ki2.append(kernels)
        
        kernels = net.Linear2.bias.cpu().detach().clone().numpy()
        Bi2.append(kernels)
    
    
    

    
    '''
    Last calculations and plot of the loss function
    '''
    
    Ki1 = np.array(Ki1)
    Ki2 = np.array(Ki2)
    Bi1 = np.array([Bi1])
    Bi2 = np.array([Bi2])
    

    colaps.append(len(colap(Bi1[0,-1,:], Ki1[-1,:,0])))
    
    losses = np.array(losses)/(w**4)
    
    
    L.append(losses[-1])
    
    plt.figure()
    plt.title('n = {}'.format(n)) 
    plt.plot(E,losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss normalizado')


'''
Last plots of the relation between the number of neurons,
 the number of colapsed neurons and the loss function
'''

L = np.array(L)
ns = np.array(ns)
colaps = np.array(colaps)

plt.figure()
plt.plot(ns,L)
plt.yscale('log')
plt.ylabel('Loss normalizada')
plt.xlabel('nº de neruonas')
plt.show()

plt.figure()
plt.plot(ns,colaps)
plt.ylabel('nº de neuronas colapsadas')
plt.xlabel('nº de neruonas')
plt.show()

plt.figure()
plt.plot(colaps,L)
plt.yscale('log')
plt.ylabel('Loss normalizada')
plt.xlabel('nº de neruonas colapsadas')
plt.show()
    
    
