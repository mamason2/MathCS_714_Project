# Michael Mason
# Math/CS 714 - Semester Project - CN method CONJUGATE GRADIENT

# importing the main libraries
import numpy as np
import scipy.sparse as spsp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import math as m
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from numba import jit

# Function CONJUGATE GRADIENT
def cg(A, x0, b, tol=1e-5, maxiter = 100):
    op_count = 0 #negates ~single operations (interested in order)

    #length of x0
    refLen = len(x0)

    #initial step
    k = 0

    #norm_r = 1e6
    r = b - A(x0)
    op_count = op_count + refLen*refLen
    p_k = np.copy(r)
    norm_r = (np.max(np.abs(r)))
    r_normInd = [norm_r]
    #print(norm_r)

    x_k = x0

    #iterations
    while k < maxiter and norm_r > tol:
        #print(r)
        k = k+1
        x_km1 = np.copy(x_k)
        r_km1 = np.copy(r)
        p_km1 = np.copy(p_k)
        w_km1 = A(p_km1)
        op_count = op_count + refLen*refLen
        alpha_km1 =  (np.transpose(r_km1).dot(r_km1))/(np.transpose(p_km1).dot(w_km1))
        x_k = x_km1 + alpha_km1*p_km1
        r = r_km1 - alpha_km1*w_km1

        norm_r = np.max(np.abs(r))
        r_normInd.append(norm_r)

        #if next loop will happen...
        if norm_r >= tol:
            beta_km1 = (np.transpose(r).dot(r))/(np.transpose(r_km1).dot(r_km1))
            p_k = r + beta_km1*p_km1
        #print(norm_r)
    
    x_star = x_k

    #print("res=",r)
    
    return x_star, op_count;
    #return x_star, r_normInd, op_count;


    
#initialize domain
N = int(51)
h = 1/(N-1)
dt = 0.0001
time = 0
timeFinal = 1.1
alpha = 0.1
mu = alpha*dt/(h*h)

baseTemp = 300
iniTemp = 450

#setup spatial domains
xx = np.zeros(N)
yy = np.zeros(N)
for ii in range(0,N):
    xx[ii] = ii*h
    yy[ii] = ii*h

#setup LHS matrix
ones = np.ones(N)
data = np.array([(-mu/2)*ones,0.5*(1+2*mu)*ones,(-mu/2)*ones])
diags = np.array([-1, 0, 1])

Anp1D = spsp.spdiags(data, diags, N,N, format="csr")
Anp = spsp.kron(Anp1D, spsp.eye(N)) + spsp.kron(spsp.eye(N), Anp1D)

#setup RHS matrix
data = np.array([(mu/2)*ones,0.5*(1-2*mu)*ones,(mu/2)*ones])
diags = np.array([-1, 0, 1])

An1D = spsp.spdiags(data, diags, N,N, format="csr")
An = spsp.kron(An1D, spsp.eye(N)) + spsp.kron(spsp.eye(N), An1D)

#BCs to add to RHS
f = np.zeros(N*N)
for jj in range(0,N):
    for ii in range(0,N):
        index = ii*N+jj
        if ii == 0 or ii == N-1:
            f[index] = f[index] + 2*baseTemp*(mu/2)
        if jj == 0:
            f[index] = f[index] + 2*(baseTemp+(baseTemp/2)*m.sin((ii/(N-1))*2*m.pi))*(mu/2)
        if jj == N-1:
            f[index] = f[index] + 2*(baseTemp-(baseTemp/2)*m.sin((ii/(N-1))*2*m.pi))*(mu/2)


#initial 1D representation (x-major order)
u_n = iniTemp*np.ones(N*N)
u2D = np.zeros([N,N]) #for plotting

#print(Anp.todense())

#for printing result plots
figPrint = 1
figPrintCount = 0

#plot initial condition

#2D reframe
for ii in range(0,N):
    for jj in range(0,N):
        index = ii*N+jj
        u2D[ii,jj] = u_n[index]

x_loc, y_loc = np.meshgrid(xx, yy)
fig = plt.figure(0)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x_loc, y_loc, u2D, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('Temp (K)', labelpad=4)
ax.set_title('Time = 0')

print(f)
print(An.dot(u_n))

#print(An.dot(u_n))

#for operation complexity statistics
tallySetup = 0
tallyOp = 0
tallyTotal = 0

#Run time loop
for i in range(1,int(timeFinal/dt)+1):
    time = time + dt
    print("time = ",time)

    u_p1, op_count = cg(lambda x: Anp.dot(x),u_n,An.dot(u_n)+f)
    setup_count = N*N*N
    #print(op_count)
    total_count = op_count+ setup_count
    #print("up1 = ",u_p1)
    u_n = np.copy(u_p1)
    #print(u_n)
    figPrintCount = figPrintCount + 1

    #For averaging
    tallySetup = tallySetup + setup_count
    tallyOp = tallyOp + op_count
    tallyTotal = tallyTotal + total_count

    #If chosen time step, plot current result
    if figPrintCount > int(0.1*timeFinal/dt):  
        for ii in range(0,N):
            for jj in range(0,N):
                index = ii*N+jj
                u2D[ii,jj] = u_p1[index]
        x_loc, y_loc = np.meshgrid(xx, yy)
        fig = plt.figure(figPrint)
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(x_loc, y_loc, u2D, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Temp (K)', labelpad=4)
        ax.set_title('Time = %f s' %np.round(time,4))
        figPrintCount = 0
        figPrint = figPrint + 1

plt.show()

print("Average per-iteration setup complexity: ",tallySetup/int(timeFinal/dt)+1)
print("Average per-iteration operations complexity: ",tallyOp/int(timeFinal/dt)+1)
print("Average per-iteration total complexity: ",tallyTotal/int(timeFinal/dt)+1)