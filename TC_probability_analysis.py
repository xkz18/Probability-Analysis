#!/usr/bin/env python
# coding: utf-8



import os
print("current working directory:",os.getcwd())



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
#pd.set_option('display.max_rows', None)




def Maibaum_calc_Q(a,b,c,d,Ntot1,Ntot2,V,calc_Q=True):
    (conc1,conc2) = (0.089,0.142)
    deltaG = pd.DataFrame()
    deltaG["j"] = [i for i in range(0,Ntot1+1) for j in range(0,i+1)]
    deltaG["k"] = [j for i in range(0,Ntot2+1) for j in range(0,i+1)]
    deltaG["fac"] = np.log(factorial(deltaG["j"])/factorial(deltaG["j"]-deltaG["k"])/factorial(deltaG["k"]))
    deltaG["dG_conc"] = -a*(deltaG["j"]-1)+b*(deltaG["j"]**(2.0/3.0)-1)+ c*(deltaG["j"]**2-1) - d*deltaG["k"] - deltaG["fac"]
    deltaG["minus_dG"] = -(deltaG["dG_conc"]+ (deltaG["j"]-1)*np.log(conc1) + deltaG["k"]*np.log(conc2))
    deltaG["K_Q"] = np.exp(deltaG["minus_dG"])
    deltaG["q"] = deltaG["K_Q"]*V**(1-deltaG["j"]-deltaG["k"])

    deltaG.loc[0,"q"] = 1
    deltaG.loc[1,"q"] = 1
    return deltaG




def Partition(q,ntot1,ntot2):
    ### q [0:ntot1+1,0:ntot2+1]
    mat_b = q
    fac = np.zeros(ntot1+2)
    facinv = np.zeros(ntot1+2)
    deriv_b = np.zeros((ntot1+2,ntot2+2))
    deriv_d = np.zeros((ntot1+2,ntot2+2))
    deriv_x = np.zeros((ntot1+2,ntot2+2))
    mat_a = np.zeros((ntot1+2,ntot2+2))
    mat_x = np.zeros((ntot1+2,ntot2+2))
    mat_y = np.zeros((ntot1+2,ntot2+2))
    mat_m = np.zeros((ntot1+2,ntot2+2))


    fac[0] = 1.0
    fac[1] = 1.0
    facinv[0] = 1.0
    facinv[1] = 1.0

    for i in range(2,ntot1+1):
        fac[i] = fac[i-1]*i
        facinv[i] = 1.0/fac[i]

    for i in range(0,ntot1+1):
        for j in range(0,ntot2+1):
            deriv_b[i,j]= mat_b[i+1,j]*(i+1)
    for i in range(0,ntot1+1):
        for j in range(0,ntot2+1):
            deriv_d[i,j]= mat_b[i,j+1]*(j+1)

    mat_a[0,0] = mat_b[0,0]
    for j in range(0,ntot2+1):
        deriv_x[0,j] = mat_b[0,j+1]*(j+1)

    mat_a[0,1] = deriv_x[0,0]
    mat_x = deriv_x
    for j in range(2,ntot2+1):
        mat_y = calc2(ntot1,ntot2,deriv_d,mat_x)
        mat_a[0,j] = mat_y[0,0] * facinv[j]
        mat_x = np.zeros((ntot1+2,ntot2+2))
        mat_x = mat_y

    mat_m = np.zeros((ntot1+2,ntot2+2))
    for j in range(0,ntot2+1):
        for i in range(0,ntot1+1):
            mat_m[i,j] = mat_b[i+1,j]*(i+1)

    mat_a[1,0] = mat_m[0,0]
    mat_x = np.zeros((ntot1+2,ntot2+2))
    mat_x = mat_m
    for j in range(0,ntot2+1):
        mat_y = calc2(ntot1,ntot2,deriv_d,mat_x)
        mat_a[1,j]=mat_y[0,0]*facinv[j]
        mat_x = np.zeros((ntot1+2,ntot2+2))
        mat_x=mat_y


    for i in range(2,ntot1+1):
        mat_x = np.zeros((ntot1+2,ntot2+2))
        mat_x = calc1(ntot1,ntot2,deriv_b,mat_m)
        mat_a[i,0]=mat_x[0,0]*facinv[i]  
        mat_c = mat_x
        for j in range(1,ntot2+1):
            mat_y = np.zeros((ntot1+2,ntot2+2))
            mat_y = calc2(ntot1,ntot2,deriv_d,mat_x)
            mat_a[i,j]=mat_y[0,0]*facinv[i]*facinv[j]
            mat_x = np.zeros((ntot1+2,ntot2+2))
            mat_x=mat_y

        mat_m=mat_c
        mat_c = np.zeros((ntot1+2,ntot2+2))
    Qn = mat_y[0,0]*facinv[ntot1]*facinv[ntot2]
    C = np.zeros((ntot1+2,ntot2+2))
    nk = np.zeros((ntot1+2,ntot2+2))
    #Calculate <n_i,j>
    for i in range(0,ntot1+1):
        for j in range(0,ntot2+1):
            k1=ntot1-i
            k2=ntot2-j
            C[k1,k2]=mat_b[k1,k2]*mat_a[i,j]
            nk[k1,k2]=C[k1,k2]/Qn
    return Qn


def calc1(ntot1,ntot2,deriv_b,mat_m):

    mat_x = np.zeros((ntot1+2,ntot2+2))
    mat_u = mat_m
    deriv_u = np.zeros((ntot1+2,ntot2+2))
    mat_v = np.zeros((ntot1*2+2,ntot2*2+2))
    mat_w = np.zeros((ntot1+2,ntot2+2))
    i = 0
    for m in range(0,ntot2+1):
        for j in range(0,ntot1+1):
            i += 1
            deriv_u[j,m]=mat_u[j+1,m]*(j+1)


    for j in range(0,ntot1+1):
        for p in range(0,ntot1+1):
            for l in range(0,ntot2+1):
                for k in range(0,ntot2+1):
                    r = j+p
                    s = l+k
                    mat_v[r,s]=mat_v[r,s]+mat_u[j,l]*deriv_b[p,k]


    for l in range(0,ntot1+1):
        for m in range(0,ntot2+1):
            mat_w[l,m]=deriv_u[l,m]+mat_v[l,m]
    return mat_w

def calc2(ntot1,ntot2,deriv_d,mat_n):
    deriv_o = np.zeros((ntot1+2,ntot2+2))
    mat_p = np.zeros((ntot1*2+2,ntot2*2+2))
    mat_q = np.zeros((ntot1+2,ntot2+2))
    for j in range(0,ntot2+1):
        deriv_o[0,j]=mat_n[0,j+1]*(j+1)
    for l in range(0,ntot2+1):
        for k in range(0,ntot2+1):
            s=l+k
            mat_p[0,s]=mat_p[0,s]+mat_n[0,l]*deriv_d[0,k]
    for l in range(0,ntot2+1):
        mat_q[0,l]=deriv_o[0,l]+mat_p[0,l]

    return mat_q




def Probfunc(a,b,c,d,T=295):
    """The Probability function"""

    #### Read INPUT
    ## Read V
    f = open("INPUT_{0}K/inputV.txt".format(T),"r")
    ReadV = f.read()
    Vset = [float(V) for V in ReadV.split("\n") if V]
    Nsimset = len(Vset)
    ## Read Nsim
    READnsim = {}
    #print(READnsim)
    for Nset in range(0,Nsimset):
        #print("Nset",Nset)
        READnsim[Nset]=pd.read_csv("INPUT_{0}K/input{1}.txt".format(T,Nset), delimiter="\s+",header=None,skiprows=5)
        READnsim[Nset].columns=["N1","N2","nsim","sd"]
    ##### Read Ntot
    f = open("INPUT_{0}K/inputN.txt".format(T),"r")
    ReadN = f.read()
    f.close()
    Ntot_set = [int(N) for N in ReadN.split("\n") if N]


    ##Calc prob
    Prob_M = []
    for Nset in range(0,Nsimset):
    #for Nset in [0]:
        Ntot1 = Ntot_set[Nset]
        Ntot2 = Ntot_set[Nset]
        V = Vset[Nset]
	print("Nset",Nset,"Ntot1",Ntot1,"Ntot2",Ntot2)

        ### Create a matrix q from df for calculation of Q
        deltaG_Q = Maibaum_calc_Q(a,b,c,d,Ntot1,Ntot2,V,True) ###Here deltaG calculates the -deltaG = lnK
        q = np.zeros((Ntot1+2,Ntot2+2))

        for j in range(0,Ntot1+2):
            for k in range(0,Ntot2+2):
                value = deltaG_Q.loc[(deltaG_Q["j"]==j)&(deltaG_Q["k"]==k),"q"]
		q[j,k] =[0 if value.empty else value][0]
		if (j==0) & (k==1):
                    q[j,k] = 1.0

        #print("q",q)
        #np.savetxt("q_1.txt",q)
        Qn= Partition(q,Ntot1,Ntot2)
        #print("Nset",Nset,Qn)



        deltaG_prodq = Maibaum_calc_Q(a,b,c,d,Ntot1,Ntot2,V,False)
        prodvalue_log = []
        deltaG_prodq["nsim"] = 0
        for i in range(0,len(deltaG_prodq["nsim"])):

            try:
                value_nsim = float(READnsim[Nset].loc[(READnsim[Nset]['N1']==deltaG_prodq.loc[i,"j"])&(READnsim[Nset]['N2']==deltaG_prodq.loc[i,"k"]),"nsim"])
                deltaG_prodq.loc[i,"nsim"]= value_nsim
            except:
                pass

        deltaG_prodq["nsim_q"] = deltaG_prodq["nsim"]*np.log(deltaG_prodq["q"])
        prodvalue_log = np.asarray(deltaG_prodq["nsim_q"])

        prodvalue_log_sum = prodvalue_log.sum()
        print("prodvalue_log_sum",prodvalue_log_sum)
        Prob_temp = -np.log(Qn)+ prodvalue_log_sum
	print("Qn",Qn)
	print("log(Prob)",Prob_temp)
	print("\n")
        Prob_M.append(Prob_temp)


    Prob = np.sum(Prob_M)
    print("log(Prob)_array",Prob_M)
    print ("log(Prob)_sum",Prob)


    return Prob








#### create the object function
# T= 295K
## Answer key
(a,b,c,d) = [4.05676336e+00,1.39109565e+01, 1.07066919e-02, -5.00666186e-01]
#for i in range(0,1):
#a = a + 0.1
prob_1 = Probfunc(a,b,c,d,295)
print("a=",a,"b=",b,"c=",c,"d=",d,"Prob=",np.exp(prob_1))




