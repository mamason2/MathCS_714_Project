# Michael Mason
# Math/CS 714 - Semester Project - ADI - 1D tridiagonal method

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

#Thomas alg
def thomas(a, b, c, d, N):
    #tally operations
    ops = 0.0

    #map creation
    ac, bc, cc, dc = map(np.array, (a, b, c, d))

    #first sweep
    for i in range(1, N):
        mc = ac[i-1]/bc[i-1]
        bc[i] = bc[i] - mc*cc[i-1] 
        dc[i] = dc[i] - mc*dc[i-1]
        ops = ops + 3
        	    
    x = bc
    x[-1] = dc[-1]/bc[-1]
    ops = ops + N*N

    #back sweep
    for i in range(N-2, -1, -1):
        x[i] = (dc[i]-cc[i]*x[i+1])/bc[i]
        ops = ops + 1

    return x, ops


#ADI tridiag function (Thomas)
def ADItri(N, neigh, cent, u, A, B, C, f_x, f_y):
    op_count = 0
    setup_count = 0
    RHS = np.zeros(N)

    u_star = np.zeros(N*N)
    u_p1 = np.zeros(N*N)

    #first half (x)
    for i in range(0,N):
        f = f_x[i,:]
        f2 = f_y[i,:]
        #print(f)
        u1x = u[i*N:((i+1)*N)]
        #print(u1x)
        for iii in range(0,N):
            count = 0
            RHS[iii] = cent*u1x[iii] + 0.5*f[iii]+ 0.5*f2[iii]
            if i > 0:
                RHS[iii] = RHS[iii] + neigh*u[i*N+iii-N]
                count = count + 1
            if i < N-1:
                RHS[iii] = RHS[iii] + neigh*u[i*N+iii+N]
                count = count + 1
            if count < 2 and f[iii] == 0:
                print("hit")
        
        setup_count = setup_count + N*N

        u1px,ops = thomas(A,B,C,RHS, N)
        op_count = op_count + ops
        #print("x",u1px)

        #implement in u_star matrix
        for iii in range (0,N):
            u_star[i*N + iii] = u1px[iii]

    #second half (y)
    for i in range(0,N):
        f = f_y[:,i]
        f2 = f_x[:,i]
        #print(f)
        u1y = u_star[i::N]
        #print(u1y)
        for iii in range(0,N):
            count = 0
            RHS[iii] = cent*u1y[iii] + 0.5*f[iii]+ 0.5*f2[iii]
            if i > 0:
                RHS[iii] = RHS[iii] + neigh*u_star[i+iii*N-1]
                count = count + 1
            if i < N-1:
                RHS[iii] = RHS[iii] + neigh*u_star[i+iii*N+1]
                count = count + 1
            if count < 2 and f[iii] == 0:
                print("hit")


        setup_count = setup_count + N*N

        u1py,ops = thomas(A,B,C,RHS, N)
        op_count = op_count + ops
        #print("y",u1py)
        for iii in range (0,N):
            u_p1[iii*N+i] = u1py[iii]
    
    #print(f_x)
    #print(f_y)

    return u_p1, op_count, setup_count;



#Main code

#initialize domain
N = int(21)
h = 1/(N-1)
dt = 0.0001
time = 0
timeFinal = 0.5
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

#setup LHS lines for tridiagonal setup
A = (-mu/2)*np.ones(N)
B = (1+mu)*np.ones(N)
C = (-mu/2)*np.ones(N)

#setup boundary-imposing f vectors
f_x = np.zeros([N,N])
f_y = np.zeros([N,N])
for ii in range(0,N):
    for jj in range(0,N):
        if ii == 0 or ii == N-1:
            f_x[ii,jj] = f_x[ii,jj] + 2*baseTemp*(mu/2)
        if jj == 0:
            f_y[ii,jj] = f_y[ii,jj] + 2*(baseTemp+(baseTemp/2)*m.sin((ii/(N-1))*2*m.pi))*(mu/2)
        if jj == N-1:
            f_y[ii,jj] = f_y[ii,jj] + 2*(baseTemp-(baseTemp/2)*m.sin((ii/(N-1))*2*m.pi))*(mu/2)

#initial 1D representation (x-major order)
u_n = iniTemp*np.ones(N*N)
u2D = np.zeros([N,N]) #for plotting

#print(Anp.todense())

#for printing result plots
figPrint = 1
figPrintCount = 0

#2D reframe
for ii in range(0,N):
    for jj in range(0,N):
        index = ii*N+jj
        u2D[ii,jj] = u_n[index]
        #u_n[index] = index

x_loc, y_loc = np.meshgrid(xx, yy)
fig = plt.figure(0)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(x_loc, y_loc, u2D, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel('Temp (K)', labelpad=4)
ax.set_title('Time = 0')


#for operation complexity statistics
tallySetup = 0
tallyOp = 0
tallyTotal = 0

neigh = (mu/2)
cent = (1-mu)

#Run time loop
for i in range(1,int(timeFinal/dt)+1):
    time = time + dt
    print("time = ",time)

    u_p1, op_count, setup_count = ADItri(N, neigh, cent, u_n, A, B, C, f_x, f_y)

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