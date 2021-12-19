# Michael Mason
# Math/CS 714 - Semester Project - Comparing Complextiy

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
        ops = ops + 1
        	    
    x = bc
    x[-1] = dc[-1]/bc[-1]
    ops = ops + N

    #back sweep
    for i in range(N-2, -1, -1):
        x[i] = (dc[i]-cc[i]*x[i+1])/bc[i]
        ops = ops + 1

    return x, ops


#LOD tridiag function (Thomas)
def LODtri(N, u, An1D, A, B, C, f_x, f_y):
    op_count = 0
    setup_count = 0

    u_star = np.zeros(N*N)
    u_p1 = np.zeros(N*N)

    #first half (x)
    for i in range(0,N):
        f = f_x[:,i]
        u1x = u[i*N:((i+1)*N)]
        #print(u1x)
        RHS = An1D.dot(u1x) + f
        setup_count = setup_count + N*N

        u1px,ops = thomas(A,B,C,RHS, N)
        op_count = op_count + ops
        #print("x",u1px)

        #implement in u_star matrix
        for iii in range (0,N):
            u_star[i*N + iii] = u1px[iii]



    #second half (y)
    for i in range(0,N):
        f = f_y[i,:]
        u1y = u_star[i::N]
        #print(u1y)
        RHS = An1D.dot(u1y) + f
        setup_count = setup_count + N*N

        u1py,ops = thomas(A,B,C,RHS, N)
        op_count = op_count + ops
        #print("y",u1py)
        for iii in range (0,N):
            u_p1[iii*N+i] = u1py[iii]

    return u_p1, op_count, setup_count;



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


#MAIN CODE


#Loop over different Ns - CG
N_index = [65, 129, 257, 513, 1025]
cn_ave_complexitySetup = np.zeros(5)
lod_ave_complexitySetup = np.zeros(5)
adi_ave_complexitySetup = np.zeros(5)
cn_ave_complexityOp = np.zeros(5)
lod_ave_complexityOp = np.zeros(5)
adi_ave_complexityOp = np.zeros(5)
cn_ave_complexityTotal = np.zeros(5)
lod_ave_complexityTotal = np.zeros(5)
adi_ave_complexityTotal = np.zeros(5)
cn_1it_ave_complexityOp =  np.zeros(5)
cn_1it_ave_complexityTotal =  np.zeros(5)

    
for iiii in  range(0,5):
    #initialize domain
    N = N_index[iiii]
    h = 1/(N-1)
    dt = 0.0001
    timeFinal = 0.1
    alpha = 0.1
    mu = alpha*dt/(h*h)

    baseTemp = 300
    iniTemp = 450



    # CRANK NICOLSON

    time = 0

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
        setup_count = N*N
        #print(op_count)
        total_count = op_count+ setup_count
        #print("up1 = ",u_p1)
        u_n = np.copy(u_p1)
        #print(u_n)

        #For averaging
        tallySetup = tallySetup + setup_count
        tallyOp = tallyOp + op_count
        tallyTotal = tallyTotal + total_count


    print("Average per-iteration setup complexity: ",tallySetup/int(timeFinal/dt)+1)
    print("Average per-iteration operations complexity: ",tallyOp/int(timeFinal/dt)+1)
    print("Average per-iteration total complexity: ",tallyTotal/int(timeFinal/dt)+1)

    cn_ave_complexitySetup[iiii] = tallySetup/int(timeFinal/dt)+1
    cn_ave_complexityOp[iiii] = tallyOp/int(timeFinal/dt)+1
    cn_ave_complexityTotal[iiii] = tallyTotal/int(timeFinal/dt)+1


    # CRANK NICOLSON - 1 Iteration (for comparison)

    time = 0

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

    #print(An.dot(u_n))

    #for operation complexity statistics
    tallySetup = 0
    tallyOp = 0
    tallyTotal = 0

    #Run time loop
    for i in range(1,int(timeFinal/dt)+1):
        time = time + dt
        print("time = ",time)

        u_p1, op_count = cg(lambda x: Anp.dot(x),u_n,An.dot(u_n)+f,tol=0.00001,maxiter=1)
        setup_count = N*N
        #print(op_count)
        total_count = op_count+ setup_count
        #print("up1 = ",u_p1)
        u_n = np.copy(u_p1)
        #print(u_n)

        #For averaging
        tallySetup = tallySetup + setup_count
        tallyOp = tallyOp + op_count
        tallyTotal = tallyTotal + total_count


    print("Average per-iteration operations complexity: ",tallyOp/int(timeFinal/dt)+1)
    print("Average per-iteration total complexity: ",tallyTotal/int(timeFinal/dt)+1)

    cn_1it_ave_complexityOp[iiii] = tallyOp/int(timeFinal/dt)+1
    cn_1it_ave_complexityTotal[iiii] = tallyTotal/int(timeFinal/dt)+1





    # LOD

    time = 0

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

    #setup RHS matrix
    data = np.array([(mu/2)*np.ones(N),(1-mu)*np.ones(N),(mu/2)*np.ones(N)])
    diags = np.array([-1, 0, 1])

    An1D = spsp.spdiags(data, diags, N,N, format="csr")

    #setup boundary-imposing f vectors
    f_x = np.zeros([N,N])
    f_y = np.zeros([N,N])
    for ii in range(0,N):
        for jj in range(0,N):
            if jj == 0 or jj == N-1:
                f_x[ii,jj] = f_x[ii,jj] + 2*baseTemp*(mu/2)
            if ii == 0:
                f_y[ii,jj] = f_y[ii,jj] + 2*(baseTemp+(baseTemp/2)*m.sin((jj/(N-1))*2*m.pi))*(mu/2)
            if ii == N-1:
                f_y[ii,jj] = f_y[ii,jj] + 2*(baseTemp-(baseTemp/2)*m.sin((jj/(N-1))*2*m.pi))*(mu/2)

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


    #for operation complexity statistics
    tallySetup = 0
    tallyOp = 0
    tallyTotal = 0

    #Run time loop
    for i in range(1,int(timeFinal/dt)+1):
        time = time + dt
        print("time = ",time)

        u_p1, op_count, setup_count = LODtri(N, u_n, An1D, A, B, C, f_x, f_y)

        total_count = op_count+ setup_count
        #print("up1 = ",u_p1)
        u_n = np.copy(u_p1)
        #print(u_n)
        figPrintCount = figPrintCount + 1

        #For averaging
        tallySetup = tallySetup + setup_count
        tallyOp = tallyOp + op_count
        tallyTotal = tallyTotal + total_count


    print("Average per-iteration setup complexity: ",tallySetup/int(timeFinal/dt)+1)
    print("Average per-iteration operations complexity: ",tallyOp/int(timeFinal/dt)+1)
    print("Average per-iteration total complexity: ",tallyTotal/int(timeFinal/dt)+1)

    lod_ave_complexitySetup[iiii] = tallySetup/int(timeFinal/dt)+1
    lod_ave_complexityOp[iiii] = tallyOp/int(timeFinal/dt)+1
    lod_ave_complexityTotal[iiii] = tallyTotal/int(timeFinal/dt)+1


    #ADI

    time = 0

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


    print("Average per-iteration setup complexity: ",tallySetup/int(timeFinal/dt)+1)
    print("Average per-iteration operations complexity: ",tallyOp/int(timeFinal/dt)+1)
    print("Average per-iteration total complexity: ",tallyTotal/int(timeFinal/dt)+1)

    adi_ave_complexitySetup[iiii] = tallySetup/int(timeFinal/dt)+1
    adi_ave_complexityOp[iiii] = tallyOp/int(timeFinal/dt)+1
    adi_ave_complexityTotal[iiii] = tallyTotal/int(timeFinal/dt)+1

N4 = np.zeros(5)
N3 = np.zeros(5)
N2 = np.zeros(5)

for i in range(0,5):
    N2[i] = N_index[i]*N_index[i]
    N3[i] = N_index[i]*N_index[i]*N_index[i]
    N4[i] = N_index[i]*N_index[i]*N_index[i]*N_index[i]

#plt.figure(1)
#plt.plot(N_index,cn_ave_complexitySetup, 'b-o') 
#plt.plot(N_index,lod_ave_complexitySetup, 'r-o') 
#plt.plot(N_index,adi_ave_complexitySetup, 'm-o') 
#plt.plot(N_index,N2, 'k--') 
#plt.plot(N_index,N4, 'k-') 
#plt.xscale("log") 
#plt.yscale("log") 
##plt.xlabel("N")
#plt.ylabel("Complexity (Approx. Setup Operations)",rotation="vertical",labelpad=2)
#plt.legend(["CN","LOD","ADI","N^3 line","N^4 line"])


plt.figure(2)
plt.plot(N_index,cn_ave_complexityOp, 'b-o') 
plt.plot(N_index,lod_ave_complexityOp, 'r-o') 
plt.plot(N_index,adi_ave_complexityOp, 'm-o') 
plt.plot(N_index,N2, 'k--') 
plt.plot(N_index,N3, 'k:') 
plt.plot(N_index,N4, 'k-') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("N")
plt.ylabel("Complexity (Approx. Solving Operations)",rotation="vertical",labelpad=2)
plt.legend(["CN","LOD","ADI","N^2 line","N^3 line","N^4 line"])




plt.figure(3)
plt.plot(N_index,cn_ave_complexityTotal, 'b-o') 
plt.plot(N_index,lod_ave_complexityTotal, 'r-o') 
plt.plot(N_index,adi_ave_complexityTotal, 'm-o') 
plt.plot(N_index,N2, 'k--') 
plt.plot(N_index,N3, 'k:') 
plt.plot(N_index,N4, 'k-') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("N")
plt.ylabel("Complexity (Approx. Total Operations)",rotation="vertical",labelpad=2)
plt.legend(["CN","LOD","ADI","N^2 line","N^3 line","N^4 line"])



plt.figure(4)
plt.plot(N_index,cn_ave_complexityOp, 'b-o') 
plt.plot(N_index,cn_1it_ave_complexityOp, 'r-o') 
plt.plot(N_index,N2, 'k--') 
plt.plot(N_index,N3, 'k:') 
plt.plot(N_index,N4, 'k-') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("N")
plt.ylabel("Complexity (Approx. Solving Operations)",rotation="vertical",labelpad=2)
plt.legend(["CN","CN - 1 iteration"])





plt.figure(5)
plt.plot(N_index,cn_ave_complexityTotal, 'b-o') 
plt.plot(N_index,cn_1it_ave_complexityTotal, 'r-o') 
plt.plot(N_index,N4, 'k-') 
plt.xscale("log") 
plt.yscale("log") 
plt.xlabel("N")
plt.ylabel("Complexity (Approx. Total Operations)",rotation="vertical",labelpad=2)
plt.legend(["CN","CN - 1 iteration","N^4 line"])

plt.show()