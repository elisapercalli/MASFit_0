from matplotlib import pyplot as plt
import numpy as np
import scipy.constants as sc
import scipy.signal as sci
import sys

#function for reading flux from input
def ReadFlux(filename, Nbin):
    TextFile = filename
    f = open(TextFile,"r")
    i=0
    v=[]
    line=f.readline()
    while line:
        v.append(float(line))
        i+1
        line=f.readline()
    f.close()
    if(np.size(v)!=Nbin):
        print("The dimension of flux input file differs from the numbers of bins")
        sys.exit(1)
    return np.array(v)

#Energy resolution function
def ResEn(e,xvis,A,B,C):
    Edep=np.array(e-0.8)
    deltaE=np.array(np.sqrt((A/(np.sqrt(Edep)))**2+B**2+(C/Edep)**2)*Edep)
    g=np.array(1/(np.sqrt(2*sc.pi)*deltaE)*np.exp(-0.5*(Edep-xvis)**2/(deltaE**2)))
    return g

# Asimov flux NO
def FluxNO_ResEn_teo(x, flux, T13, T12, M21, M3l, dist, N, A, B,C): #x in MeV
    #Theorical flux of anti-nu
    F=flux
    #IBD cross section
    Ee=x-1.293
    m_e=0.511
    pe=np.sqrt(Ee**2-m_e**2)
    s=10**(-43)*pe*Ee*(x**(-0.07056+0.02018*np.log(x)-0.001953*np.log(x)**3))
    #Oscillation probability
    hbar=sc.hbar/sc.e
    L=dist*10**3/(hbar*sc.c)
    y=x*10**6
    M31=M3l
    M32=M31-M21
    D21=M21*L/y/4
    D31=M31*L/y/4
    D32=M32*L/y/4
    P=1 - 4*(1-T13)**2*T12*(1-T12)*np.sin(D21)**2 - 4*(1-T12)*T13*(1-T13)*np.sin(D31)**2 - 4*T12*T13*(1-T13)*np.sin(D32)**2
    Y=np.array(F*s*P)
    #Normalized flux
    tot=np.sum(Y)
    Y*=N/tot
    #Energy resolution
    n=np.size(x)
    out=[]
    Evis=np.array(x-0.8)
    for i in range(0,n):
        gi=ResEn(x,Evis[i],A,B,C)
        gi/=np.sum(gi)
        conv=np.sum(gi*Y)
        out.append(conv)
    yconv=np.array(out)
    return yconv

# Asimov flux IO
def FluxIO_ResEn_teo(x, flux, T13, T12, M21, M3l, dist, N, A, B, C):
    #Theorical flux of anti-nu
    F=flux
    #IBD cross section
    Ee=x-1.293
    m_e=0.511
    pe=np.sqrt(Ee**2-m_e**2)
    s=10**(-43)*pe*Ee*(x**(-0.07056+0.02018*np.log(x)-0.001953*np.log(x)**3))
    #Oscillation probability
    hbar=sc.hbar/sc.e
    L=dist*10**3/(hbar*sc.c)
    y=x*10**6
    M32=M3l
    M31=M32+M21
    D21=M21*L/y/4
    D31=M31*L/y/4
    D32=M32*L/y/4
    P=1 - 4*(1-T13)**2*T12*(1-T12)*np.sin(D21)**2 - 4*(1-T12)*T13*(1-T13)*np.sin(D31)**2 - 4*T12*T13*(1-T13)*np.sin(D32)**2
    #Normalized flux
    Y=np.array(F*s*P)
    tot=np.sum(Y)
    Y*=N/tot
    #Energy resolution
    n=np.size(x)
    out=[]
    Evis=np.array(x-0.8)
    for i in range(0,n):
        gi=ResEn(x,Evis[i],A,B,C)
        gi/=np.sum(gi)
        conv=np.sum(gi*Y)
        out.append(conv)
    yconv=np.array(out)
    return yconv

#Flux NO with statistical fluctuations
def FluxNO_ResEn(x, flux, T13, T12, M21, M3l, dist, N, A, B, C):
    #Theorical flux of anti-nu
    F=flux
    #IBD cross section
    Ee=x-1.293
    m_e=0.511
    pe=np.sqrt(Ee**2-m_e**2)
    s=10**(-43)*pe*Ee*(x**(-0.07056+0.02018*np.log(x)-0.001953*np.log(x)**3))
    #Oscillation probability
    hbar=sc.hbar/sc.e
    L=dist*10**3/(hbar * sc.c)
    y=x*10**6
    M31=M3l
    M32=M31-M21
    D21=M21*L/y/4
    D31=M31*L/y/4
    D32=M32*L/y/4
    P=1 - 4*(1-T13)**2*T12*(1-T12)*np.sin(D21)**2 - 4*(1-T12)*T13*(1-T13)*np.sin(D31)**2 - 4*T12*T13*(1-T13)*np.sin(D32)**2
    Y=np.array(F*s*P)
    #Normalized flux
    tot=np.sum(Y)
    Y*=N/tot
    #Poisson fluctuations
    new=np.array(np.random.poisson(Y),dtype='float')
    #Energy resolution
    n=np.size(x)
    out=[]
    Evis=np.array(x-0.8)
    for i in range(0,n):
        gi=ResEn(x[i],Evis,A,B,C)
        gi/=np.sum(gi)
        conv=np.sum(gi*new)
        out.append(conv)
    yconv=np.array(out)
    return yconv

#Flux IO with statistical fluctuations
def FluxIO_ResEn(x, flux, T13, T12, M21, M3l, dist, N, A, B, C):
    #Theorical flux of anti-nu
    F=flux
    #IBD cross section
    Ee=x-1.293
    m_e=0.511
    pe=np.sqrt(Ee**2-m_e**2)
    s=10**(-43)*pe*Ee*(x**(-0.07056+0.02018*np.log(x)-0.001953*np.log(x)**3))
    #Oscillation probability
    hbar=sc.hbar/sc.e
    L=dist*10**3/(hbar * sc.c)
    y=x*10**6
    M32=M3l
    M31=M32+M21
    D21=M21*L/y/4
    D31=M31*L/y/4
    D32=M32*L/y/4
    P=1 - 4*(1-T13)**2*T12*(1-T12)*np.sin(D21)**2 - 4*(1-T12)*T13*(1-T13)*np.sin(D31)**2 - 4*T12*T13*(1-T13)*np.sin(D32)**2
    #Normalized flux
    Y=np.array(F*s*P)
    tot=np.sum(Y)
    Y*=N/tot
    #Poisson fluctuations
    new=np.array(np.random.poisson(Y),dtype='float')
    #Energy resolution
    n=np.size(x)
    out=[]
    Evis=np.array(x-0.8)
    for i in range(0,n):
        gi=ResEn(x,Evis[i],A,B,C)
        gi/=np.sum(gi)
        conv=np.sum(gi*new)
        out.append(conv)
    yconv=np.array(out)
    return yconv
