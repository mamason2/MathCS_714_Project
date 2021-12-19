# Michael Mason
# Math/CS 714 - Semester Project
# Comparing ERROR (not complexity) scaling (naive implementations)

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

#  Function - C-N Cycle (dir. BCs)
@jit(nopython=True)
def CN_dir(u, h, N, dt, alpha, thresh):
    #initialize t+1 u matrix ("next" time step)
    mu = alpha*dt/(h*h)
    u_p1 = np.copy(u)

    uNorm = 10 
    countIt = 0

    while uNorm >= thresh:
        countIt = countIt + 1
        u_p1_old = np.copy(u_p1)

        # Function takes in vector u and h, N, dt, alpha, and returns next u
        for ii in range(1,N-1):
            for jj in range(1,N-1):
                u_p1[ii,jj] = ((mu/2)*(u_p1[ii-1,jj]+u_p1[ii+1,jj]+u_p1[ii,jj-1]+u_p1[ii,jj+1]) + (1-2*mu)*u[ii,jj]+(mu/2)*(u[ii-1,jj]+u[ii+1,jj]+u[ii,jj-1]+u[ii,jj+1]))/(2*mu + 1)

        uNorm = np.max(np.absolute(u_p1-u_p1_old))
        #print(uNorm) 

    #print(countIt)
    return u_p1;



#  Function - ADI Cycle (dir. BCs)
@jit(nopython=True)
def ADI_dir(u, h, N, dt, alpha, thresh):
    #initialize t+1/2 and t+1 u matrix ("next" time step)
    mu = alpha*dt/(h*h)
    u_h1 = np.copy(u) #first half
    u_p1 = np.copy(u) #second half

    uNorm = 10 
    countIt = 0

    # First half calc (x-oriented for advancing half)
    while uNorm >= thresh:
        countIt = countIt + 1
        u_h1_old = np.copy(u_h1)

        for ii in range(1,N-1):
            for jj in range(1,N-1):
                u_h1[ii,jj] = ((mu/2)*(u_h1[ii-1,jj]+u_h1[ii+1,jj]) + (1-mu)*u[ii,jj]+(mu/2)*(u[ii,jj-1]+u[ii,jj+1]))/(mu + 1)

        uNorm = np.max(np.absolute(u_h1-u_h1_old))
        #print(uNorm) 

    uNorm = 10 
    countIt = 0

    # Second half calc (y-oriented for advancing half)
    while uNorm >= thresh:
        countIt = countIt + 1
        u_p1_old = np.copy(u_p1)

        for ii in range(1,N-1):
            for jj in range(1,N-1):
                u_p1[ii,jj] = ((mu/2)*(u_p1[ii,jj-1]+u_p1[ii,jj+1]) + (1-mu)*u_h1[ii,jj]+(mu/2)*(u_h1[ii-1,jj]+u_h1[ii+1,jj]))/(mu + 1)

        uNorm = np.max(np.absolute(u_p1-u_p1_old))
        #print(uNorm) 

    #print(countIt)
    return u_p1;



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




#MAIN CODE

N_index = [129,65,33,17]
N_spacing = [8,16,32,64]
h_ind = np.zeros(4)
maxNormsDifCN = np.zeros(4)
maxNormsDifADI = np.zeros(4)
maxNormsDifLOD = np.zeros(4)


tol = 0.00001
timeFinal = 1

baseTemp = 300 #K
iniTemp = 450 #K

#Initially run high-fidelity CN case for comparisons

#initialize domain
N = int(1025)
h = 1/(N-1)
dt = 0.000005
timeCN = 0

#Initial field and BCs
u_ini = iniTemp*np.ones([N,N])
for i in range(0,N):
    u_ini[0,i] = baseTemp+(baseTemp/2)*m.sin((i/(N-1))*2*m.pi)
    u_ini[N-1,i] = baseTemp-(baseTemp/2)*m.sin((i/(N-1))*2*m.pi)
    u_ini[i,0] = baseTemp
    u_ini[i,N-1] = baseTemp

#setup spatial domains
xx = np.zeros(N)
yy = np.zeros(N)
for ii in range(0,N):
    xx[ii] = ii*h
    yy[ii] = ii*h

#Run time loop
for i in range(1,int(timeFinal/dt)+1):
    timeCN = timeCN + dt
    print("time = ",timeCN)

    u_p1_check = CN_dir(u_ini,h,N,dt,0.1,tol)
    u_ini = np.copy(u_p1_check)

u_cn_base = np.copy(u_p1_check)



#Run comparison tests

for iii in range(0,4):
    ind = iii

    print(N_index[iii])

    #initialize domain
    N = N_index[iii]
    h = 1/(N-1)
    h_ind[ind] = h
    dt = 0.000005
    timeCN = 0
    timeLOD = 0
    timeADI = 0

    #Initial field and BCs
    u_iniCN = iniTemp*np.ones([N,N])

    for i in range(0,N):
        u_iniCN[0,i] = baseTemp+(baseTemp/2)*m.sin((i/(N-1))*2*m.pi)
        u_iniCN[N-1,i] = baseTemp-(baseTemp/2)*m.sin((i/(N-1))*2*m.pi)
        u_iniCN[i,0] = baseTemp
        u_iniCN[i,N-1] = baseTemp

    u_iniLOD = np.copy(u_iniCN)
    u_iniADI = np.copy(u_iniCN)

    #setup spatial domains
    xx = np.zeros(N)
    yy = np.zeros(N)
    for ii in range(0,N):
        xx[ii] = ii*h
        yy[ii] = ii*h

    #Run time loop - CN
    for i in range(1,int(timeFinal/dt)+1):
        timeCN = timeCN + dt
        print("time = ",timeCN)

        u_p1_checkCN = CN_dir(u_iniCN,h,N,dt,0.1,tol)
        u_iniCN = np.copy(u_p1_checkCN)

    #Run time loop - LOD
    for i in range(1,int(timeFinal/dt)+1):
        timeLOD = timeLOD + dt
        print("time = ",timeLOD)

        u_p1_checkLOD = LOD_dir(u_iniLOD,h,N,dt,0.1,tol)
        u_iniLOD = np.copy(u_p1_checkLOD)

    #Run time loop - ADI
    for i in range(1,int(timeFinal/dt)+1):
        timeADI = timeADI + dt
        print("time = ",timeADI)

        u_p1_checkADI = ADI_dir(u_iniADI,h,N,dt,0.1,tol)
        u_iniADI = np.copy(u_p1_checkADI)

    #Compare to high-resolution CN

    NspaceCur = N_spacing[ind]
    for ii in range(0,N):
        for jj in range(0,N):
            #print(ii*NspaceCur)
            if(np.absolute(u_p1_checkCN[ii,jj]-u_cn_base[ii*NspaceCur,jj*NspaceCur])>maxNormsDifCN[ind]):
                maxNormsDifCN[ind] = np.absolute(u_p1_checkCN[ii,jj]-u_cn_base[ii*NspaceCur,jj*NspaceCur])

            if(np.absolute(u_p1_checkLOD[ii,jj]-u_cn_base[ii*NspaceCur,jj*NspaceCur])>maxNormsDifLOD[ind]):
                maxNormsDifLOD[ind] = np.absolute(u_p1_checkLOD[ii,jj]-u_cn_base[ii*NspaceCur,jj*NspaceCur])

            if(np.absolute(u_p1_checkADI[ii,jj]-u_cn_base[ii*NspaceCur,jj*NspaceCur])>maxNormsDifADI[ind]):
                maxNormsDifADI[ind] = np.absolute(u_p1_checkADI[ii,jj]-u_cn_base[ii*NspaceCur,jj*NspaceCur])

print("maxCN (fine) = ",np.min(u_cn_base))
print("maxCN = ",np.min(u_p1_checkCN))
print("maxLOD = ",np.min(u_p1_checkLOD))
print("maxADI = ",np.min(u_p1_checkADI))


plt.figure(1)
plt.plot(h_ind,maxNormsDifCN, 'm-o') 
plt.plot(h_ind,maxNormsDifLOD, 'b-o') 
plt.plot(h_ind,maxNormsDifADI, 'r-o') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("h")
plt.ylabel("Maximum error norm",rotation="vertical",labelpad=2)
plt.legend(['CN','LOD','ADI'])

x_loc, y_loc = np.meshgrid(xx, yy)

plt.figure(2)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x_loc, y_loc, u_p1_checkCN, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('Temp (K)', labelpad=4)
ax.set_title('Time = 0')

plt.figure(3)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x_loc, y_loc, u_p1_checkLOD, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('Temp (K)', labelpad=4)
ax.set_title('Time = 0')

plt.figure(4)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x_loc, y_loc, u_p1_checkADI, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('Temp (K)', labelpad=4)
ax.set_title('Time = 0')


plt.figure(5)
plt.plot(h_ind,maxNormsDifCN, 'b-o') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("h")
plt.ylabel("Maximum error norm",rotation="vertical",labelpad=2)


plt.figure(6)
plt.plot(h_ind,maxNormsDifLOD, 'b-o') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("h")
plt.ylabel("Maximum error norm",rotation="vertical",labelpad=2)


plt.figure(7)
plt.plot(h_ind,maxNormsDifADI, 'b-o') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("h")
plt.ylabel("Maximum error norm",rotation="vertical",labelpad=2)






plt.show()