import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from matplotlib import animation

'''
PINN to solve the forced harmonic oscillator with
mixed methods (Adam + LBFGS) and Forzed CC. 
Animations of the weights and biases are included(Must uncomment the animation wanted).
Marco Mas Sempre
'''


n = 30              #Number of intermediate neurons

'''
EDO parameters
'''

w = 6*np.pi
A = 1
g = 5


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
Function to animate the weights and biases
'''

def animaPesos(M, tit, direct):
    M = np.array(M)

    fig, ax = plt.subplots()
    fig.suptitle(tit)

    im = ax.imshow(M[0, :, :], interpolation="none")
    cbar = fig.colorbar(im)

    def plotsim(i):
        im.set_array(M[i, :, :])
        im.set_clim(vmin=M[i, :, :].min(), vmax=M[i, :, :].max())
        cbar.update_normal(im)
        return [im]

    ani = animation.FuncAnimation(fig, plotsim, frames=len(M[:, 0, 0]), interval = 10, repeat_delay=2000, blit=True)
    ani.save(direct)
    plt.show()
    

def animaBias(M, tit, direct):
    M = np.array(M)

    fig, ax = plt.subplots()
    fig.suptitle(tit)

    im = ax.imshow(M[:,0, :], interpolation="none")
    cbar = fig.colorbar(im)

    def plotsim(i):
        im.set_array(M[:,i, :])
        im.set_clim(vmin=M[:,i, :].min(), vmax=M[:,i, :].max())
        cbar.update_normal(im)
        return [im]

    ani = animation.FuncAnimation(fig, plotsim, frames=len(M[0,:, 0]), interval = 10, repeat_delay=2000, blit=True)
    ani.save(direct)
    plt.show()




'''
Defining some necesary lists
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
Initializing the NN
'''


net = Red(n)
net.apply(init_weights)
loss_fun = nn.MSELoss(reduction = 'mean')





'''
First training with ADAM
'''

lr = 0.01
epochsAdam =1000
changebach = 100
estatus_print = 100
N = 2000
losscup = 25

optimizer = optim.Adam(params = net.parameters(), lr = lr)

t0 = torch.linspace(0,1,N)
t = Variable(t0, requires_grad = True).view(-1, 1)


with torch.no_grad():
    tn = torch.linspace(start = 0, end = 1,steps= 100).view(-1, 1)
    
    Y_pred=fx(tn).numpy() 
    tn = tn.numpy()
    plt.figure()
    plt.title('Epoch 0')
    plt.plot(tn,Y_pred,label = "Usando la red")
    plt.plot(tn,sol(tn),label = 'Solución analítica')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('$x(t)$')
            
    plt.show()





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
    
      
    

    if epoch % estatus_print == 0:
        print("Epoch {} = {}".format(epoch,loss.item()))
        
        
        
    with torch.no_grad():
        if epoch % (10*estatus_print) == 0:

            tn = torch.linspace(start = 0, end = 1,steps= 100).view(-1, 1)

            Y_pred=fx(tn).numpy() 
            tn = tn.numpy()
            plt.figure()
            plt.title('Epoch {}'.format(epoch))
            plt.plot(tn,Y_pred,label = "PINN")
            plt.plot(tn,sol(tn),label = 'Analytic solution')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('$x(t)$')
        
            plt.show()

    if loss.item() < losscup and  epoch > 10*estatus_print:
        break


Ultepoch = epoch



'''
Second training with LBFGS
'''

lr = 1
epochsLBFGS =1000
changebach = 100
estatus_print = 100
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
    
    
    
    
    
    if epoch % estatus_print == 0:
        print("Epoch {} = {}".format(epoch,loss.item()))
        
        
        
    with torch.no_grad():
        if epoch % (5*estatus_print) == 0:

            tn = torch.linspace(start = 0, end = 1,steps= 100).view(-1, 1)

            Y_pred=fx(tn).numpy() 
            tn = tn.numpy()
            plt.figure()
            plt.title('Epoch {}'.format(epoch))
            plt.plot(tn,Y_pred,label = "PINN")
            plt.plot(tn,sol(tn),label = 'Analytic solution')
            plt.legend()
            plt.xlabel('t')
            plt.ylabel('$x(t)$')
        
            plt.show()
    if loss.item() < losscup2:
        break
    
with torch.no_grad():
    tn = torch.linspace(start = 0, end = 1,steps= 100).view(-1, 1)
    
    Y_pred=fx(tn).numpy() 
    tn = tn.numpy()
    plt.figure()
    plt.title('Epoch {}'.format(epoch))
    plt.plot(tn,Y_pred,label = "PINN")
    plt.plot(tn,sol(tn),label = 'Analytic solution')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('$x(t)$')
    plt.show()






'''
Some animations, if want to see them, just uncomment the following lines
'''



Ki1 = np.array(Ki1)
Ki2 = np.array(Ki2)
Bi1 = np.array([Bi1])
Bi2 = np.array([Bi2])

'''
animaPesos(Ki1, 'Pesos capa 1', 'Prueba\Pesos_Capa_1_n={}.mp4'.format(n))
animaPesos(Ki2, 'Pesos capa 2', 'Prueba\Pesos_Capa_2_n={}.mp4'.format(n))
animaBias(Bi1, 'Bias capa 1', 'Prueba\Bias_Capa_1_n={}.mp4'.format(n))
animaBias(Bi2, 'Bias capa 2', 'Prueba\Bias_Capa_2_n={}.mp4'.format(n))
'''



'''
Last plot about the loss function, we normalize it by w^4 to make
it more comparable with other cases
'''


losses = np.array(losses)/(w**4)


plt.figure()
plt.title('n = {}'.format(n)) 
plt.plot(E,losses)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Normalized Loss')
plt.show()


