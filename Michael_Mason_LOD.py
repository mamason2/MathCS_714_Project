# Michael Mason
# Math/CS 714 - Semester Project - LOD method

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

#  Function - LOD Cycle (dir. BCs)
@jit(nopython=True)
def LOD_dir(u, h, N, dt, alpha, thresh):
    #initialize t+1/2 and t+1 u matrix ("next" time step)
    mu = alpha*dt/(h*h)
    u_h1 = np.copy(u) #first half
    u_p1 = np.copy(u) #second half

    uNorm = 10 
    countIt = 0

    # First half calc (x-oriented)
    while uNorm >= thresh:
        countIt = countIt + 1
        u_h1_old = np.copy(u_h1)

        for ii in range(1,N-1):
            for jj in range(1,N-1):
                u_h1[ii,jj] = ((mu/2)*(u_h1[ii-1,jj]+u_h1[ii+1,jj]) + (1-mu)*u[ii,jj]+(mu/2)*(u[ii-1,jj]+u[ii+1,jj]))/(mu + 1)

        uNorm = np.max(np.absolute(u_h1-u_h1_old))
        #print(uNorm) 

    uNorm = 10 
    countIt = 0

    # Second half calc (y-oriented)
    while uNorm >= thresh:
        countIt = countIt + 1
        u_p1_old = np.copy(u_p1)

        for ii in range(1,N-1):
            for jj in range(1,N-1):
                u_p1[ii,jj] = ((mu/2)*(u_p1[ii,jj-1]+u_p1[ii,jj+1]) + (1-mu)*u_h1[ii,jj]+(mu/2)*(u_h1[ii,jj-1]+u_h1[ii,jj+1]))/(mu + 1)

        uNorm = np.max(np.absolute(u_p1-u_p1_old))
        #print(uNorm) 

    #print(countIt)
    return u_p1;



#initialize domain
N = int(51)
h = 1/(N-1)
dt = 0.0001
time = 0
timeFinal = 1

baseTemp = 300
iniTemp = 450

#Initial field and BCs
u_ini = iniTemp*np.ones([N,N])
for i in range(0,N):
    u_ini[0,i] = baseTemp+(baseTemp/2)*m.sin((i/(N-1))*2*m.pi)
    u_ini[N-1,i] = baseTemp-(baseTemp/2)*m.sin((i/(N-1))*2*m.pi)
    #u_ini[0,i] = baseTemp
    #u_ini[N-1,i] = baseTemp
    u_ini[i,0] = baseTemp
    u_ini[i,N-1] = baseTemp

#setup spatial domains
xx = np.zeros(N)
yy = np.zeros(N)
for ii in range(0,N):
    xx[ii] = ii*h
    yy[ii] = ii*h

#for printing result plots
figPrint = 1
figPrintCount = 0

#plot initial condition
x_loc, y_loc = np.meshgrid(xx, yy)
fig = plt.figure(0)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x_loc, y_loc, u_ini, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('Temp (K)', labelpad=4)
ax.set_title('Time = 0')

#Run time loop
for i in range(1,int(timeFinal/dt)+1):
    time = time + dt
    print("time = ",time)

    u_p1_check = LOD_dir(u_ini,h,N,dt,0.1,0.000001)
    u_ini = np.copy(u_p1_check)
    figPrintCount = figPrintCount + 1


    #If chosen time step, plot current result
    if figPrintCount > int(0.1*timeFinal/dt):  
        x_loc, y_loc = np.meshgrid(xx, yy)
        fig = plt.figure(figPrint)
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(x_loc, y_loc, u_p1_check, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Temp (K)', labelpad=4)
        ax.set_title('Time = %f s' %np.round(time,4))
        figPrintCount = 0
        figPrint = figPrint + 1

plt.show()