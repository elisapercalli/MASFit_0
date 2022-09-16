from matplotlib import pyplot as plt
import numpy as np
import scipy.constants as sc
from iminuit import cost, Minuit
from iminuit.cost import LeastSquares
import iminuit
import os
import sys
import MASFit_func_0 as fn

#Reading of parameters from file
InputTextFile = sys.argv[1]
InputFluxFile = sys.argv[2]
f = open(InputTextFile,"r")
i=0
v=[]
line=f.readline()
while line:
    sep = line.split()
    appo=float(sep[1])
    v.append(appo)
    i+1
    line=f.readline()
f.close()

#Energy and other parameters
Fluc=v[0]
if (Fluc==True):
    M=int(v[1])
else:
    M=1
N=int(v[2])
Emin=v[3]
Emax=v[4]
E=np.linspace(Emin,Emax,N)
N20=330000
N6=100000
N100=4000
Ntot=v[5]
#Anti-nu oscillations parameters
D=v[6]
Theta12=v[7]
Theta13_NO=v[8]
Theta13_IO=v[9]
DeltaM21=v[10]
DeltaM31_NO=v[11]
DeltaM32_IO=v[12]

#Energy resolution and systematic uncertainties
a=v[13]/100
b=v[14]/100
c=v[15]/100
sigma_a=v[16]/100
sigma_b=v[17]/100
sigma_c=v[18]/100
sigma_alphaC=v[19]/100
sigma_alphaD=v[20]/100
sigma_b2b=v[21]/100
sigma_alphaR=v[22]/100
sist=bool(v[23])
corr=bool(v[24])
#Fit parameters
Fix_M21=bool(v[25])
Fix_M3l=bool(v[26])
Fix_T13=bool(v[27])
Fix_T12=bool(v[28])
Fix_N=bool(v[29])
Fix_a=bool(v[30])
Fix_b=bool(v[31])
Fix_c=bool(v[32])

#Reading anti-nu flux from file
F=fn.ReadFlux(InputFluxFile, N)

#Creation of anti-nu NO flux
bin=(Emax-Emin)/N
E2=[]
E2.append(Emin)
for i in range(0,N-1):
    E2.append(Emin+bin*(i+0.5))
e=np.array(E2)
xe=e-0.8
D_CHI=[]
bins=int(1+3*np.log(M))

for i in range(0,M):
    if(i % 10==0 & i!=0):
        print(i)

    #without statistical fluctuations
    if(Fluc==False):
        yNO=fn.FluxNO_ResEn_teo(e, F, Theta13_NO, Theta12, DeltaM21, DeltaM31_NO, D, Ntot, a, b, c)

    #with statistical fluctuations
    else:
        yNO=fn.FluxNO_ResEn(e, F, Theta13_NO, Theta12, DeltaM21, DeltaM31_NO, D, Ntot, a, b, c)

    #Computing of errors
    err=np.sqrt(yNO)
    for i in range(0,N):
        if err[i]==0:
            err[i]=1

    #Fit NO su NO
    if(sist==False):
        def least_squares_NO(T13, T12, M21, M3l, dist, N, A, B, C):
            ym = fn.FluxNO_ResEn_teo(e, F, T13, T12, M21, M3l, dist, N, A, B, C)
            z=(yNO-ym)/err
            CHI=np.sum(z**2)
            #pull_N=((N-Ntot)/(Ntot*0.01))**2
            pull_a=((A-a)/sigma_a)**2
            pull_b=((B-b)/sigma_b)**2
            pull_c=((C-c)/sigma_c)**2
            return CHI+pull_a+pull_b+pull_c
        least_squares_NO.errordef = Minuit.LEAST_SQUARES
        m = Minuit(least_squares_NO, T13=Theta13_NO, T12=Theta12, M21=DeltaM21, M3l=DeltaM31_NO, dist=D, N=Ntot, A=a, B=b, C=c)

    #Fit NO su NO with systematics
    else:
        def least_squares_sist_NO(T13, T12, M21, M3l, dist, N, A, B, C,alpha_C,alpha_D,alpha_R):
            ym = fn.FluxNO_ResEn_teo(e, F, T13, T12, M21, M3l, dist, N, A, B, C)
            z=(yNO-ym*(1+alpha_C+alpha_D+alpha_R))**2/(yNO+(sigma_b2b*ym)**2)
            CHI=np.sum(z)
            pull_a=((A-a)/sigma_a)**2
            pull_b=((B-b)/sigma_b)**2
            pull_c=((C-c)/sigma_c)**2
            return CHI+pull_a+pull_b+pull_c+(alpha_C/sigma_alphaC)**2+(alpha_D/sigma_alphaD)**2+(alpha_R/sigma_alphaR)**2
        least_squares_sist_NO.errordef = Minuit.LEAST_SQUARES
        m = Minuit(least_squares_sist_NO, T13=Theta13_NO,T12=Theta12,M21=DeltaM21,M3l=DeltaM31_NO,dist=D,N=Ntot,A=a,B=b,C=c,alpha_C=0,alpha_D=0,alpha_R=0.1)

    #Fit parameters
    m.fixed["dist"]=True
    m.fixed["M21"]=Fix_M21
    m.fixed["M3l"]=Fix_M3l
    m.fixed["T13"]=Fix_T13
    m.fixed["T12"]=Fix_T12
    m.fixed["N"]=Fix_N
    m.fixed["A"]=Fix_a
    m.fixed["B"]=Fix_b
    m.fixed["C"]=Fix_c
    m.migrad()
    m.hesse()

    #Fit IO su NO
    if(sist==False):
        def least_squares_IO(T13, T12, M21, M3l, dist, N, A, B, C):
            ym = fn.FluxIO_ResEn_teo(e, F, T13, T12, M21, M3l, dist, N, A, B, C)
            z=(yNO-ym)/err
            CHI=np.sum(z**2)
            #pull_N=((N-Ntot)/(Ntot*0.01))**2
            pull_a=((A-a)/sigma_a)**2
            pull_b=((B-b)/sigma_b)**2
            pull_c=((C-c)/sigma_c)**2
            return CHI+pull_a+pull_b+pull_c
        least_squares_IO.errordef = Minuit.LEAST_SQUARES
        n = Minuit(least_squares_IO, T13=Theta13_IO, T12=Theta12, M21=DeltaM21, M3l=DeltaM32_IO, dist=D, N=Ntot, A=a, B=b, C=c)

    #Fit IO su NO with systematics
    else:
        def least_squares_sist_IO(T13, T12, M21, M3l, dist, N, A, B, C,alpha_C,alpha_D,alpha_R):
            ym = fn.FluxIO_ResEn_teo(e, F, T13, T12, M21, M3l, dist, N, A, B, C)
            z=(yNO-ym*(1+alpha_C+alpha_D+alpha_R))**2/(yNO+(sigma_b2b*ym)**2)
            CHI=np.sum(z)
            pull_a=((A-a)/sigma_a)**2
            pull_b=((B-b)/sigma_b)**2
            pull_c=((C-c)/sigma_c)**2
            return CHI+pull_a+pull_b+pull_c+(alpha_C/sigma_alphaC)**2+(alpha_D/sigma_alphaD)**2++(alpha_R/sigma_alphaR)**2
        least_squares_sist_IO.errordef = Minuit.LEAST_SQUARES
        n = Minuit(least_squares_sist_IO, T13=Theta13_IO,T12=Theta12,M21=DeltaM21,M3l=DeltaM32_IO,dist=D,N=Ntot,A=a,B=b,C=c,alpha_C=0,alpha_D=0,alpha_R=0)

    #Fit parameters
    n.fixed["dist"]=True
    n.fixed["M21"]=Fix_M21
    n.fixed["M3l"]=Fix_M3l
    n.fixed["T13"]=Fix_T13
    n.fixed["T12"]=Fix_T12
    n.fixed["N"]=Fix_N
    n.fixed["A"]=Fix_a
    n.fixed["B"]=Fix_b
    n.fixed["C"]=Fix_c
    n.migrad()
    n.hesse()

    D_CHI.append(n.fval-m.fval)

if(Fluc==False):
    #Plot parameters scan
    if(corr==True):
        fig_Z3,ax_Z3 = plt.subplots(1,1)
        xm,ym,rm = m.mnprofile('M3l',bound=5, subtract_min=False)
        xn,yn,rn = n.mnprofile('M3l',bound=5, subtract_min=False)
        ax_Z3.plot(xm, ym, label="NO", markersize=2, color='blue')
        ax_Z3.plot(np.abs(xn), yn, label="IO", markersize=2, color='red')
        leg_Z3 = ax_Z3.legend();
        ax_Z3.grid()
        plt.title(f"Scan $\\chi^2$")
        plt.tight_layout()
        plt.xlabel("$|\\Delta m^2_3l| [10^{-3} eV^2]$")
        plt.ylabel("$\\chi^2$")
        fig_Z3.set_size_inches(10, 8)
        plt.savefig("MASFit_scan_0.pdf")

    #Print on file of fit reconstructed parameters
    file_par = open("MASFit_parameters_0.txt", "w")
    file_par.write("Free parameters in fit NO")
    file_par.write("\nParameter\tInj_value\tRec_value\tError\tBias(%)")
    if(Fix_M21==False):
        file_par.write(f"\n\nDeltaM21\t{DeltaM21:.3e}\t{m.values[2]:.3e}\t{m.errors[2]:.3e}\t{(m.values[2]-DeltaM21)/DeltaM21*100:.3f}")
    if(Fix_M3l==False):
        file_par.write(f"\nDeltaM31\t{DeltaM31_NO:.3e}\t{m.values[3]:.3e}\t{m.errors[3]:.3e}\t{(m.values[3]-DeltaM31_NO)/DeltaM31_NO*100:.3f}")
    if(Fix_T13==False):
        file_par.write(f"\nSin^2(T_13)\t{Theta13_NO}\t{m.values[0]:.5f}\t{m.errors[0]:.5f}\t{(m.values[0]-Theta13_NO)/Theta13_NO*100:.3f}")
    if(Fix_T12==False):
        file_par.write(f"\nSin^2(T_12)\t{Theta12}\t{m.values[1]:.3f}\t{m.errors[1]:.3f}\t{(m.values[1]-Theta12)/Theta12*100:.3f}")
    if(Fix_N==False):
        file_par.write(f"\nN\t{Ntot}\t{m.values[5]}\t{m.errors[5]:.0f}\t{(m.values[5]-Ntot)/Ntot*100:.3f}")
    if(Fix_a==False):
        file_par.write(f"\na\t{a:.4f}\t{m.values[6]:.4f}\t{m.errors[6]:.4f}\t{(m.values[6]-a)/a*100:.3f}")
    if(Fix_b==False):
        file_par.write(f"\nb\t{b:.4f}\t{m.values[7]:.4f}\t{m.errors[7]:.4f}\t{(m.values[7]-b)/b*100:.3f}")
    if(Fix_c==False):
        file_par.write(f"\nc\t{c:.4f}\t{m.values[8]:.4f}\t{m.errors[8]:.4f}\t{(m.values[8]-c)/c*100:.3f}")
    file_par.write(f"\nChi^2 = {m.fval:.2f}")

    file_par.write("\n\nFree parameters in fit IO")
    file_par.write("\nParameter\tInj_value\tRec_value\tError\tBias(%)")
    if(Fix_M21==False):
        file_par.write(f"\n\nDeltaM21\t{DeltaM21:.3e}\t{n.values[2]:.3e}\t{n.errors[2]:.3e}\t{(n.values[2]-DeltaM21)/DeltaM21*100:.3f}")
    if(Fix_M3l==False):
        file_par.write(f"\nDeltaM32\t{DeltaM32_IO:.3e}\t{n.values[3]:.3e}\t{n.errors[3]:.3e}\t{(n.values[3]-DeltaM32_IO)/DeltaM32_IO*100:.3f}")
    if(Fix_T13==False):
        file_par.write(f"\nSin^2(T_13)\t{Theta13_IO}\t{n.values[0]:.5f}\t{n.errors[0]:.5f}\t{(n.values[0]-Theta13_IO)/Theta13_IO*100:.3f}")
    if(Fix_T12==False):
        file_par.write(f"\nSin^2(T_12)\t{Theta12}\t{n.values[1]:.3f}\t{n.errors[1]:.3f}\t{(n.values[1]-Theta12)/Theta12*100:.3f}")
    if(Fix_N==False):
        file_par.write(f"\nN\t{Ntot}\t{n.values[5]}\t{n.errors[5]:.0f}\t{(n.values[5]-Ntot)/Ntot*100:.3f}")
    if(Fix_a==False):
        file_par.write(f"\na\t{a:.4f}\t{n.values[6]:.4f}\t{n.errors[6]:.4f}\t{(n.values[6]-a)/a*100:.3f}")
    if(Fix_b==False):
        file_par.write(f"\nb\t{b:.4f}\t{n.values[7]:.4f}\t{n.errors[7]:.4f}\t{(n.values[7]-b)/b*100:.3f}")
    if(Fix_c==False):
        file_par.write(f"\nc\t{c:.4f}\t{n.values[8]:.4f}\t{n.errors[8]:.4f}\t{(n.values[8]-c)/c*100:.3f}")
    file_par.write(f"\nChi^2 = {n.fval:.2f}")
    file_par.write(f"\n\nDeltaChi^2 = {n.fval-m.fval:.2f}")
    file_par.close()

    #Plot
    plt.figure(figsize=(12, 8))
    plt.title("Fit su NO")
    plt.errorbar(xe, yNO, yerr=err, markersize=3, fmt="o", label="data", alpha=0.4)
    if(sist==True):
        valuesNO=np.delete(m.values,[9,10,11])
        valuesIO=np.delete(n.values,[9,10,11])
    else:
        valuesNO=m.values
        valuesIO=n.values
    plt.plot(xe, fn.FluxNO_ResEn_teo(e,F,*valuesNO), label="fit NO", color="seagreen")
    plt.plot(xe, fn.FluxIO_ResEn_teo(e,F,*valuesIO), label="fit IO", color="coral")

    # Legend and fit info
    CHI_NO=m.fval/(len(e)-m.nfit)
    CHI_IO=n.fval/(len(e)-n.nfit)
    DeltaCHI=n.fval-m.fval
    fit_info=[f"$\\Delta$ $\\chi^2$ = {DeltaCHI:.3f}"]
    fit_info.append(f"$\\chi^2$ / $n_\\mathrm{{dof}}$ NO = {CHI_NO:.3f}")
    for i in range(6,8):
        fit_info.append(f"{m.parameters[i]} = ${m.values[i]:.6f} \\pm {m.errors[i]:.6f}$")
    fit_info.append(f"$\\chi^2$ / $n_\\mathrm{{dof}}$ IO = {CHI_IO:.3f}")
    for i in range(6,8):
        fit_info.append(f"{n.parameters[i]} = ${n.values[i]:.6f} \\pm {n.errors[i]:.6f}$")
    plt.legend(title="\n".join(fit_info))
    plt.grid()
    plt.xlabel("E_vis [MeV]")
    plt.ylabel("N events")
    plt.savefig("MASFit_plot_0.pdf")

else:
    #Plot Chi squared
    plt.figure(figsize=(12, 8))
    plt.title("Fit su NO")
    plt.hist(D_CHI,bins)
    plt.grid()
    plt.xlabel(f"$\\Delta$ $\\chi^2$")
    plt.ylabel("N events")
    plt.savefig("MASFit_CHI.pdf")

if(v[33]==True):
    plt.show()
