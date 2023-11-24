import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

# Spatial Grid and Initial Condition:
N = 128
x = 32*np.pi*np.arange(1,N+1)/N
u = np.cos(x/16)*(1+np.sin(x/16))
v = fft(u)

# Precompute ETDRK4 scalar quantities
h = 1/4
k = np.concatenate([np.arange(0,N/2),np.array([0.]),np.arange(-N/2+1,0)],0)/16
L = k**2 - k**4 
E = np.exp(h*L)
E2 = np.exp(h*L/2)
M = 16
r = (1j*np.pi*(np.arange(1,M+1)-.5)/M)
LR = np.repeat(h*L[:,None],M,1) + np.repeat(r[None,:],N,0)
Q = h*np.mean(((np.exp(LR/2)-1)/LR),1).real
f1 = h*np.mean(((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3),1).real
f2 = h*np.mean(((2+LR+np.exp(LR)*(-2+LR))/LR**3),1).real
f3 = h*np.mean(((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3),1).real

# Timestepping
uu = [u]
tt = [0]
tmax = 150
nmax = int(tmax/h)
nplt = int((tmax/100)/h)
g = -.5j*k

for n in range(1,nmax+1):
    t = n*h
    Nv = g * fft(ifft(v).real**2)
    a = E2*v + Q*Nv
    Na = g * fft(ifft(a).real**2)
    b = E2*v + Q*Na
    Nb = g * fft(ifft(b).real**2)
    c = E2*a + Q*(2*Nb-Nv)
    Nc = g * fft(ifft(c).real**2)
    v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
    if n%nplt==0:
        u = ifft(v).real
        uu.append(u)
        tt.append(t) 

uu = np.stack(uu)
tt = np.array(tt)

print(uu.shape, tt.shape)

plt.imshow(uu)
plt.show()