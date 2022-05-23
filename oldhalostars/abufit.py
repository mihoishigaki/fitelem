import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import os, sys, re, glob




from scipy.interpolate import interp1d
from scipy import interpolate

import pymc3 as pm
import theano
import theano.tensor as tt
import arviz as az




#sys.path.insert(0, os.path.abspath('./'))
import oldhalostars.abudata as ad

def fmt0(x): 
    return '$%.0f$'%x

def fmt1(x):   
    return '$%.1f$'%x

def fmt2(x):   
    return '$%.2f$'%x


def fmt3(x):   
    return '$%.2f$'%x


def fmt4(x):   
    return '$%.4f$'%x



def fmt5(x):
    return '$%.5f$'%x

def fmtis(x):
    x=np.int(x)
    return '%s'%x

def fmts(x):
    return x

def fmts3(x):
    return '%3s'%x

def fmtsmr(x):
    if x=="":
        return '%s'%x
    else:
        return '\\multirow{5}{*}{%s}'%x


def fmte1(x):
    return '$%.1e$'%x

def fmte2(x):
    return '$%.2e$'%x

def fmti(x):
    return '%i'%x

def fmt2ors(x):
    if x=="-":
        return fmts(x)
    else:
        return '$%.2f$'%np.float64(x)
    



def fmt3ors(x):
    if x==-9.99:
        x="-"
        return fmts(x)
    else:
        return '$%.3f$'%np.float64(x)
    


def get_solarm(z):
    
    solarfile='/Users/ishigakimiho/AlphaStarMatching/ipcc.dat'
    
    zsuns,abusuns=np.loadtxt(solarfile,usecols=(1,5),unpack=True,skiprows=1)
    
    found=0
    for i,zsun in enumerate(zsuns):
        if zsun==z:
            
            if z==1:
                mass=10**abusuns[i]*(1.0*99.998/100+2.0*0.002/100)
            elif z==2:
                mass=10**abusuns[i]*(3.0*0.0166/100+4.0*99.9834/100)
            elif z==3:
                mass=10**abusuns[i]*(6.0*7.59/100+7.0*92.41/100)
            elif z==4:
                mass=10**abusuns[i]*(9.0*100./100)
            elif z==5:
                mass=10**abusuns[i]*(10.0*19.9/100+11.0*80.1/100)
            elif z==6:
                mass=10**abusuns[i]*(12.0*98.8938/100+13.0*1.1062/100)
            elif z==7:
                mass=10**abusuns[i]*(14.0*99.771/100+15.0*0.229/100)
            elif z==6.5:
                mass=10**abusuns[5]*(12.0*98.8938/100+13.0*1.1062/100)+\
                    10**abusuns[6]*(14.0*99.771/100+15.0*0.229/100)
            elif z==8:
                mass=10**abusuns[i]*(16.0*99.7621/100+17.0*0.0379/100+18.0*0.2/100)
            elif z==9:
                mass=10**abusuns[i]*(19.0*100./100)
        
            elif z==11:
                mass=10**abusuns[i]*(23.0*100./100)
            elif z==12:
                mass=10**abusuns[i]*(24.0*78.99/100+25.0*10.00/100+26.0*11.01/100)
            elif z==13:
                mass=10**abusuns[i]*(27.0*100./100)
            elif z==14:
                mass=10**abusuns[i]*(28.0*92.2297/100+29.0*4.6832/100+30.0*3.0872/100)
            elif z==15:
                mass=10**abusuns[i]*(31.0*100./100)
            elif z==16:
                mass=10**abusuns[i]*(32.0*94.93/100+33.0*0.76/100+34.0*4.29/100+36.0*0.02/100)
            elif z==17:
                mass=10**abusuns[i]*(35.0*75.78/100+37.0*24.22/100)

            elif z==19:
                mass=10**abusuns[i]*(39.0*93.132/100+40.0*0.147/100+41*6.721/100)
            elif z==20:
                mass=10**abusuns[i]*(40.0*96.941/100+42.0*0.647/100+43.0*0.135/100+\
                                 44.0*2.086/100+46.0*0.004/100+48.0*0.187/100)
            elif z==21:
                mass=10**abusuns[i]*(45.0*100./100)
            elif z==22:
                mass=10**abusuns[i]*(46.0*8.25/100+47.0*7.44/100+48.0*73.72/100+\
                                 49.0*5.41/100+50.0*5.18/100)
            elif z==23:
                mass=10**abusuns[i]*(50.0*0.25/100+51.0*99.750/100)
            elif z==24:
                mass=10**abusuns[i]*(50.0*4.345/100+52.0*83.789/100+\
                                  53.0*9.501/100+54.0*2.365/100)
            elif z==25:
                mass=10**abusuns[i]*(55.0*100./100)
            elif z==26:
                mass=10**abusuns[i]*(54.0*5.845/100+56.0*91.754/100+\
                                 57.0*2.119/100+58.0*0.282/100)
            elif z==27:
                mass=10**abusuns[i]*(59.0*100./100)
            elif z==28:
                mass=10**abusuns[i]*(58.0*68.0769/100+60.0*26.2231/100+\
                                  61.0*1.1399/100+62.0*3.6345/100+64.0*0.9256/100)
            elif z==29:
                mass=10**abusuns[i]*(63.0*69.17/100+65.0*30.83/100)
            elif z==30:
                mass=10**abusuns[i]*(64.0*48.63/100+66.0*27.90/100+\
                                  67.0*4.10/100+68.0*18.75/100+70.0*0.62/100)
            
            elif z==38:
            
                mass=10**abusuns[i]*88.
                
                #(64.0*48.63/100+66.0*27.90/100+\
                               #   67.0*4.10/100+68.0*18.75/100+70.0*0.62/100)
               
            
            found=1
            break
    if found==0:
        if z==10:
            abusun=7.93-12
            mass=10**abusun*(20.0*92.9431/100+21.0*0.2228/100+22.0*6.8341/100)
        elif z==18:
            abusun=6.40-12
            mass=10**abusun*(36.0*84.5946/100+38.0*15.3808/100+40.0*0.0246/100)
        elif z==6.5:
            mass=10**abusuns[5]*(12.0*98.8938/100+13.0*1.1062/100)+\
                10**abusuns[6]*(14.0*99.771/100+15.0*0.229/100)
        else:
            print('Solar abundance for Z=',z,' Not Found!')
            sys.exit()
    
    
    
    
    return(mass)





def read_cc_yieldtable():

    
        ## Pop II yield from Chiaki's table
    
    yieldfile='../yields/CCSNe/Chiaki_yield.dat'
    
    f=open(yieldfile)
    lines=f.readlines()
    f.close()
    
    no_of_Z_cc=4
    no_of_xiso=77
    no_of_mass_cc=7
    yields=np.float64(np.arange(no_of_mass_cc*no_of_Z_cc*no_of_xiso).\
                            reshape(no_of_mass_cc,no_of_Z_cc,no_of_xiso))
    yields.fill(0.0)
  
    j=0
    for i,line in enumerate(lines):
        if i<=24:
            continue
        data=line.split()
        import re
        if re.match("M_",data[1]):
            continue
        if np.float64(data[0])==0.0:
            ii=0
        elif np.float64(data[0])==0.001:
            ii=1
        elif np.float64(data[0])==0.004:
            ii=2
        elif np.float64(data[0])==0.02:
            ii=3
        

        yields[:,ii,j] = np.float64(data[2:9])

        
        if j==76:
            j=0
        else:
            j=j+1
   
    return(yields)



def calc_IMF_averaged_TypeIIyields(imfslope):
    
       
    # Read CCSN yield table
    yields = read_cc_yieldtable()
    
    
    expfac=-1.0*imfslope
    
    # Average over imfslopse
    no_of_Z_cc=4
    no_of_xiso=77
    no_of_cc_mass=7
    masses=np.array([13.,15.,18.,20.,25.,30.,40])
    yields_imf=np.float64(np.arange(no_of_Z_cc*no_of_xiso).\
                            reshape(no_of_Z_cc,no_of_xiso))
    yields_imf.fill(0.0)
    
    ## Calculate denominator
    integral=0.0
    for k in range(0,no_of_cc_mass-1):
        integral=integral+0.5*(masses[k]**expfac+masses[k+1]**expfac)*(masses[k+1]-masses[k])
    
    
    for ii in range(0,no_of_Z_cc):
        for j in range(0,no_of_xiso):
            for k in range(0,no_of_cc_mass-1):
                yields_imf[ii,j]=yields_imf[ii,j]+\
                0.5*(yields[k,ii,j]*masses[k]**expfac+yields[k+1,ii,j]*masses[k+1]**expfac)\
                *(masses[k+1]-masses[k])/integral
    
    
    # Check
    #x=np.arange(np.size(yields[0,0,:]))+1
    #y=yields[0,0,:]
    #plt.plot(x[x>10],y[x>10],linestyle=":")
    #y=yields[5,0,:]
    #plt.plot(x[x>10],y[x>10],linestyle=":")
    
    #y=yields_imf[0,:]
    #plt.plot(x[x>10],y[x>10],linestyle="-")
    #plt.show()
    
    #sys.exit()
    
    return(yields_imf)





def read_TypeIayield_subChandra_Solar(znum,zsc):
    

    
    elems=["c","n","o","f","ne","na","mg","al","si","p","s","cl","ar","k","ca",\
           "sc","ti","v","cr","mn","fe","co","ni","cu","zn"]

    yieldfile='../yields/TypeIaYields_Sep15_2020/yieldtable_M1.00_He0.05_sph_solar.dat' 
    
    
    
    zsun=0.015
    metals=np.array([0.0,0.1*zsun,0.5*zsun,1.0*zsun,2.0*zsun])
    df=pd.read_csv(yieldfile,header=None,names=["isonames","zsun0.0","zsun0.1","zsun0.5",\
                                                "zsun1.0","zsun2.0"],skiprows=1,delimiter='\s+',\
                   usecols=[0, 1, 2, 3, 4, 5])
    
    
    masses=np.zeros(np.size(znum),dtype=float)
    
    for i,elem in enumerate(elems):
        for j,iso in enumerate(df["isonames"]):
    
            isochar=re.sub(r'[0-9]+',"",iso)
            if elem==isochar:
                yield_metals=np.array([df["zsun0.0"][j],df["zsun0.1"][j],df["zsun0.5"][j],df["zsun1.0"][j],\
                                       df["zsun2.0"][j]])
                
                #plt.plot(metals,yield_metals,linestyle="-")
                
                
                f = interpolate.interp1d(metals, yield_metals)
                
                #plt.plot([zch],[f(zch)],marker="o",linestyle="",color="red")
                #plt.show()
                
                masses[i]=masses[i]+f(zsc)
    
        
    z_co=6.5
    type1ayield=np.hstack((masses[0],masses[0]+masses[1],masses[1:]))
    z=np.hstack((znum[0],[z_co],znum[1:]))

    
    return(z,type1ayield)





def read_TypeIayield_Chandra_Solar(znum,zch):
    
    import re
    
    
    elems=["c","n","o","f","ne","na","mg","al","si","p","s","cl","ar","k","ca",\
           "sc","ti","v","cr","mn","fe","co","ni","cu","zn"]

    yieldfile='../yields//TypeIaYields_Sep15_2020/yieldtable_Chand_M1.37_solar.dat' 

    metals=np.array([0.0,0.002,0.01,0.02,0.04,0.06,0.10])
    df=pd.read_csv(yieldfile,header=None,names=["isonames","z0","z0002",\
                                                "z001","z002","z004","z006","z010"],skiprows=7,delimiter='\s+')
    
    
    masses=np.zeros(np.size(znum),dtype=float)
    
    for i,elem in enumerate(elems):
        for j,iso in enumerate(df["isonames"]):
            isochar=re.sub(r'[0-9]+',"",iso)
            if elem==isochar:
                yield_metals=np.array([df["z0"][j],df["z0002"][j],df["z001"][j],\
                                       df["z002"][j],df["z004"][j],df["z006"][j],df["z010"][j]])
                
                
          
                
                f = interpolate.interp1d(metals, yield_metals)
                
           
                masses[i]=masses[i]+f(zch)
                

    z_co=6.5
    type1ayield=np.hstack((masses[0],masses[0]+masses[1],masses[1:]))
    z=np.hstack((znum[0],[z_co],znum[1:]))

    
    
    
    return(z,type1ayield)



def calc_yields_CC(theta):
    
    
    no_of_xiso = 77
    
    
    alpha, zcc = theta  
    
    
    # Fixed parameters
    
    yields_imf = calc_IMF_averaged_TypeIIyields(alpha)
  

    # Average over metallicity  (z_cc between 0.00 and 0.02)
    yields_imf_metal=np.zeros(no_of_xiso)
    yields_imf_metal.fill(0.0)
    metals=np.array([0.0,0.001,0.004,0.02])
    
    if zcc<0.0:
        zcc=metals[0]
    elif zcc>=0.02:
        zcc=metals[3]
    
    
    for j in range(0,no_of_xiso):
    
        func=interp1d(metals, yields_imf[:,j])
        yields_imf_metal[j]=func(zcc)
        
        
    nelem=25
    Type2yield_tmp=np.zeros(nelem)
    z_tmp=np.array([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

    for i,zz in enumerate(z_tmp):
        Type2yield_tmp[i] = ad.calc_ElemMass(zz,yields_imf_metal)
       
    # Check
    #xfe=np.zeros_like(Type2yield_tmp)
    #for i,zz in enumerate(z_tmp):
    #    solar_mass_ratio=get_solarm(zz)/get_solarm(26)
            
    #    xfe[i]=np.log10(Type2yield_tmp[i]/Type2yield_tmp[20])-np.log10(solar_mass_ratio)
 
    #plt.plot(z_tmp,xfe)
    #plt.show()
    

    z_co=6.5
    Type2yield=np.hstack((Type2yield_tmp[0], ad.calc_ElemMass(z_co,yields_imf_metal), \
                          Type2yield_tmp[1:]))
    z=np.hstack((z_tmp[0],[z_co],z_tmp[1:]))
 
   
    
    ## Mix PopIII, CC and Type Ia
    
    
    totalyield=Type2yield
    
    mass_fe=totalyield[z==26]

        
    nelem=np.size(z)
    xfe=np.zeros(nelem)
    for i in range(0,nelem):
            
        solar_mass_ratio=ad.get_solarm(z[i])/ad.get_solarm(26)
            
        xfe[i]=np.log10(totalyield[i]/mass_fe)-np.log10(solar_mass_ratio)
 
    #plt.plot(z,xfe)
    #plt.ylim(-1.0,1.0)
    #plt.show()
    return(z,xfe,mass_fe)    





def calc_yields_CC_Ia(theta, zIa, f_Ch):
    
    
    no_of_xiso = 77
    
    alpha,zcc,f_Ia = theta
    

    zch = zIa
    zsc = zIa
    
    #alpha,zcc,zch,zsc,f_Ia=theta  
 
    
    yields_imf = calc_IMF_averaged_TypeIIyields(alpha)
  

    # Average over metallicity  (z_cc between 0.00 and 0.02)
    yields_imf_metal=np.zeros(no_of_xiso)
    yields_imf_metal.fill(0.0)
    metals=np.array([0.0,0.001,0.004,0.02])
    
    if zcc<0.0:
        zcc=metals[0]
    elif zcc>=0.02:
        zcc=metals[3]
    
    
    
    for j in range(0,no_of_xiso):
    
        func=interp1d(metals, yields_imf[:,j])
        yields_imf_metal[j]=func(zcc)
        
        
    nelem=25
    Type2yield_tmp=np.zeros(nelem)
    z_tmp=np.array([6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])

    for i,zz in enumerate(z_tmp):
        Type2yield_tmp[i] = ad.calc_ElemMass(zz,yields_imf_metal)
       
    # Check
    #xfe=np.zeros_like(Type2yield_tmp)
    #for i,zz in enumerate(z_tmp):
    #    solar_mass_ratio=get_solarm(zz)/get_solarm(26)
            
    #    xfe[i]=np.log10(Type2yield_tmp[i]/Type2yield_tmp[20])-np.log10(solar_mass_ratio)
 
    #plt.plot(z_tmp,xfe)
    #plt.show()
    

    z_co=6.5
    Type2yield=np.hstack((Type2yield_tmp[0], ad.calc_ElemMass(z_co,yields_imf_metal), \
                          Type2yield_tmp[1:]))
    z=np.hstack((z_tmp[0],[z_co],z_tmp[1:]))
 
    
    # Type Ia yields
    ## Subchandra
  
    #z1a,subchandrayield=read_TypeIayield_subChandra(z_tmp,zsc)
    z1a,subchandrayield=read_TypeIayield_subChandra_Solar(z_tmp,zsc)

    ## Chandra
    #z1a,chandrayield=read_TypeIayield_Chandra(z_tmp,zch)
    z1a,chandrayield=read_TypeIayield_Chandra_Solar(z_tmp,zch)

    
    ## Sub-Chandra + Chandra
    TypeIayield=f_Ch*chandrayield+(1.0-f_Ch)*subchandrayield
 
 
    
    
    ## Mix PopIII, CC and Type Ia
    
    totalyield_Type2=(1.0-f_Ia)*Type2yield
    totalyield_TypeIa=f_Ia*TypeIayield
    
    #totalyield=(1.0-f_Ia)*Type2yield+f_Ia*TypeIayield
    totalyield=totalyield_Type2+totalyield_TypeIa
    
    mass_fe=totalyield[z==26]

        
    nelem=np.size(z)
    xfe=np.zeros(nelem)
    for i in range(0,nelem):
            
        solar_mass_ratio=ad.get_solarm(z[i])/ad.get_solarm(26)
            
        xfe[i]=np.log10(totalyield[i]/mass_fe)-np.log10(solar_mass_ratio)
 
    #plt.plot(z,xfe)
    #plt.ylim(-1.0,1.0)
    #plt.show()
    return(z,xfe,mass_fe,totalyield_Type2,totalyield_TypeIa)    






def lnlike_CC(theta, x, y, yerr):


    
    alpha,zcc= theta
    #print(theta)
    
    #zsun=0.015
    #if zcc>=zsun or zcc<0.0 or zch<0.0 or zch>=zsun or zsc<=0.0 or zsc>=zsun\
    #    or f_ch<0.0 or f_ch>1.0 or f_Ia<0.0 or f_Ia>1.0:
    #    return -np.inf
    
    model_z0, model0, mass_fe = calc_yields_CC(theta)
    
 

    model_z=np.zeros_like(x)
    model=np.zeros_like(y)
    for j,xx in enumerate(x):
        model_z[j]=model_z0[model_z0==xx]
        model[j]=model0[model_z0==xx]    
   
    inv_sigma2 = 1.0/yerr**2

    
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))





def lnlike_CC_Ia(theta, x, y, yerr, zIa, f_Ch):

    #zcc,zch,zsc,f_Ch,f_Ia = theta
    #print(theta)
    # Free parameters zcc,f_Ia
    alpha, zcc, f_Ia = theta
  
    
    #if zch>zcc or zsc>zcc:
    #    return -np.inf
    
    #zsun=0.015
    #if zcc>=zsun or zcc<0.0 or zch<0.0 or zch>=zsun or zsc<=0.0 or zsc>=zsun\
    #    or f_ch<0.0 or f_ch>1.0 or f_Ia<0.0 or f_Ia>1.0:
    #    return -np.inf
    
    model_z0, model0, mass_fe, m_Type2, m_TypeIa \
        = calc_yields_CC_Ia(theta, zIa, f_Ch)
    
 

    model_z = np.zeros_like(x)
    model = np.zeros_like(y)
    for j,xx in enumerate(x):
        model_z[j] = model_z0[model_z0 == xx]
        model[j] = model0[model_z0 == xx]    
   
    inv_sigma2 = 1.0/yerr**2

    
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2 - np.log(inv_sigma2)))




# define a theano Op for our likelihood function
class LogLike(tt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma, zIa, f_Ch):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma
        self.zIa = zIa
        self.f_Ch = f_Ch
        

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        if self.zIa is np.nan:
            logl = self.likelihood(theta, self.x, self.data, self.sigma)
        else:
            logl = self.likelihood(theta, self.x, self.data, self.sigma, self.zIa, self.f_Ch)

        outputs[0][0] = np.array(logl)  # output the log-likelihood





def fit_abundances_MAP(starname, f_Ch, Ia = True, woSiCa = False, Zn_uplim = False):

    CC = True
    
    outdir = "../outputs/MAPresults_fCh%.2f_CC%1i_Ia%1i_woSiCa%1i_Znuplim%1i/"%\
        (f_Ch, int(CC == True), int(Ia == True), int(woSiCa == True), \
         int(Zn_uplim == True))
    
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

    elems = ["c", "o", "na", "mg", "al", "si", "ca", "v", "cr", "mn", "co", "ni", "cu", "zn"]
    #elems = ["o", "na", "mg", "al", "si", "ca", "cr", "mn", "ni", "zn"]


    x, data, sigma, feh, abuclass = ad.read_GALAH_xfe(starname, elems)


    
    zcc_max = ad.get_zfrac(feh)

    
    # create our Op

    if Ia == True:

        zsun=0.0152
        if zcc_max < 0.1*zsun:
            zIa = 0.0
        else:
            zIa = 0.1*zsun

        logl = LogLike(lnlike_CC_Ia, data, x, sigma, zIa, f_Ch)

    else:
        zIa = np.nan
        f_Ch = np.nan
        logl = LogLike(lnlike_CC, data, x, sigma, zIa, f_Ch)
        

        
    ndraws = 3000  # number of draws from the distribution
    nburn = 1000  # number of "burn-in points" (which we'll discard)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as model:

        # uniform priors on the parameters
        alpha = pm.Uniform("alpha", lower=-1., upper=3.0)
        zcc = pm.Uniform("zcc", lower=0.0, upper=zcc_max)


    
        # convert the parameters to a tensor vector
        if Ia == True: 
            f_Ia = pm.Uniform("f_Ia", lower=0.0, upper=1.0)
            theta = tt.as_tensor_variable([alpha, zcc, f_Ia])
        else:
            theta = tt.as_tensor_variable([alpha, zcc])

            
        # use a DensityDist (use a lamdba function to "call" the Op)
        pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})

    # Carryout MAP estimate
    ## See below for the meaning of the outputs
    ## https://stackoverflow.com/questions/42146962/what-does-the-find-map-output-mean-in-pymc3
    
    map_estimate = pm.find_MAP(model = model)


    ## Calculate lnlike corresponds to the map_estimate

    theta_final = [map_estimate['alpha'], map_estimate['zcc'], map_estimate['f_Ia']]
    lnlike = lnlike_CC_Ia(theta_final, x, data, sigma, zIa, f_Ch)


    results = {}

    results['starname'] = starname
    results['feh'] = feh
    results['abuclass'] = abuclass.values
    results['zIa'] = zIa
    results['f_Ch'] = f_Ch
    results['alpha'] = map_estimate['alpha']
    results['zcc'] = map_estimate['zcc']
    if Ia == True:
        results['f_Ia'] = map_estimate['f_Ia']
        theta_final = [map_estimate['alpha'], map_estimate['zcc'], map_estimate['f_Ia']]
        results['lnlike'] = lnlike_CC_Ia(theta_final, x, data, sigma, zIa, f_Ch)
    else:
        theta_final = [map_estimate['alpha'], map_estimate['zcc']]
        results['lnlike'] = lnlike_CC(theta_final, x, data, sigma)

    outname = outdir + starname+"_bestfitparams.csv"

    df = pd.DataFrame(results, index = [0])
    df.to_csv(outname, index = False)
    
    return()








def fit_abundances_op(starname, f_Ch, CC = True, Ia = True, woSiCa = False, Zn_uplim = False):
    
    #starlist="starlist_GALAH.txt"
    #starlist="starlist_GALAH_age12Gyr.txt"
    #f=open(starlist)
    #lines=f.readlines()
    #f.close()
    
    #outpath="GALAH_results_CC_Ia/"


    
    outdir = "../outputs/opresults_fCh%.2f_CC%1i_Ia%1i_woSiCa%1i_Znuplim%1i/"%\
        (f_Ch, int(CC == True), int(Ia == True), int(woSiCa == True), \
         int(Zn_uplim == True))

    if os.path.exists(outdir) == False:
        os.mkdir(outdir)


    elems = ["c", "o", "na", "mg", "al", "si", "ca", "v", "cr", "mn", "co", "ni", "cu", "zn"]

    #z = elem2Znum(elems)
        
    z, xfe, xfeerr, feh, abuclass = ad.read_GALAH_xfe(starname, elems)

   
    #    if feh>0.0:
    #        continue
    #    print(starname,feh)
     
    outname = outdir + starname+"_bestfitparams.txt"
    fout = open(outname, "w")
    
    zmax = ad.get_zfrac(feh)
        
    # Parammeters are alpha,zcc, f_Ia
    #    z_Ia is assumed to be 0 if Z<0.1Msun
    #                         0.1Msun if Z>=0.1Msun
        
    zsun=0.0152
    if zmax < 0.1*zsun:
        zIa = 0.0
    else:
        zIa = 0.1*zsun
    

    if Ia == True:
        
        theta0 = [2.35, 0.5*zmax, 0.5]
        print("The initial ghess: ",theta0)
        
        theta_fit,chi2,dof = \
            fit_ml_CC_Ia(z, xfe, xfeerr, feh, theta0, zIa, \
                         f_Ch, woSiCa, Zn_uplim)
            
        #    #alpha,zcc,f_Ia=theta_fit
        
        alpha, zcc, f_Ia = theta_fit
        
        outtext="%s,%f,%f,%f,%f,%f,%f,%f,%i,%s\n"%\
            (starname, feh, alpha, zcc, f_Ia, zIa, f_Ch, \
             chi2, dof, abuclass)
    
        fout.write(outtext)
            
        zmodel, xfemodel, mass_fe, mass_Type2, mass_TypeIa = \
            calc_yields_CC_Ia(theta_fit, zIa, f_Ch)
        a = np.array([zmodel, xfemodel])
        aa = a.T
         
        abuout = outdir + starname.replace(" ","") + \
            "_bestfitmodelabundance.txt"
        np.savetxt(abuout, aa, fmt = ["%.1f","%.3f"], \
                   delimiter = ",", header = "Z,XFe")
        fout.close()
    

        plot_bestfitmodel_CC_Ia(outdir + \
                                starname,z,xfe,xfeerr,\
                                feh,theta_fit,zIa,f_Ch,chi2,dof)
    
    elif Ia == False:

        theta0 = [2.35,0.5*zmax]
        theta_fit,chi2,dof = fit_ml_CC(z,xfe,xfeerr,feh,theta0)
            
        alpha,zcc = theta_fit
        
        outtext = "%s,%f,%f,%f,%f,%i,%s\n"%\
            (starname, feh, alpha, zcc, chi2, dof, abuclass)
    
        fout.write(outtext)
            
        zmodel,xfemodel,mass_fe = calc_yields_CC(theta_fit)
        a = np.array([zmodel,xfemodel])
        aa = a.T
            #starname.gsub(" ","")
        abuout = outdir + starname.replace(" ","")+"_bestfitmodelabundance.txt"
        np.savetxt(abuout, aa, fmt = ["%.1f","%.3f"], delimiter = ",", header = "Z,XFe")
        fout.close() 
        plot_bestfitmodel_CC(outdir + starname,z,xfe,xfeerr,feh,theta_fit,chi2,dof)
    
    return()



def fit_abundances_MCMC(starname, ncores, f_Ch, Ia = True, woSiCa = False, Zn_uplim = False):

    CC = True
    
    outdir = "../outputs/MCMCresults_fCh%.2f_CC%1i_Ia%1i_woSiCa%1i_Znuplim%1i/"%\
        (f_Ch, int(CC == True), int(Ia == True), int(woSiCa == True), \
         int(Zn_uplim == True))
    
    if os.path.exists(outdir) == False:
        os.mkdir(outdir)

    elems = ["c", "o", "na", "mg", "al", "si", "ca", "v", "cr", "mn", "co", "ni", "cu", "zn"]
    #elems = ["o", "na", "mg", "al", "si", "ca", "cr", "mn", "ni", "zn"]


    x, data, sigma, feh, abuclass = ad.read_GALAH_xfe(starname, elems)


    
    zcc_max = ad.get_zfrac(feh)

    
    # create our Op

    if Ia == True:

        zsun=0.0152
        if zcc_max < 0.1*zsun:
            zIa = 0.0
        else:
            zIa = 0.1*zsun

        logl = LogLike(lnlike_CC_Ia, data, x, sigma, zIa, f_Ch)

    else:
        zIa = np.nan
        f_Ch = np.nan
        logl = LogLike(lnlike_CC, data, x, sigma, zIa, f_Ch)
        

    outncname = outdir + starname + "_trace.nc"

    
    ndraws = 3000  # number of draws from the distribution
    nburn = 1000  # number of "burn-in points" (which we'll discard)

    # use PyMC3 to sampler from log-likelihood
    with pm.Model() as model:

        # uniform priors on the parameters
        alpha = pm.Uniform("alpha", lower=-1., upper=3.0)
        zcc = pm.Uniform("zcc", lower=0.0, upper=zcc_max)
        f_Ia = pm.Uniform("f_Ia", lower=0.0, upper=1.0)

    
        # convert the parameters to a tensor vector
        if Ia == True: 
            theta = tt.as_tensor_variable([alpha, zcc, f_Ia])
        else:
            theta = tt.as_tensor_variable([alpha, zcc])

            
        # use a DensityDist (use a lamdba function to "call" the Op)
        pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})

        #trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

        start = pm.find_MAP(model = model)
        step = pm.Slice()
        trace = pm.sample(ndraws, step = step,  start = start, \
                      cores = ncores, return_inferencedata = True, \
                      idata_kwargs = {"density_dist_obs": False})


        az.to_netcdf(trace, outncname)
    

        ax = az.plot_trace(trace)
        fig = ax.ravel()[0].figure
        fig.savefig(outdir + "trace_" + starname + ".png")

        df = az.summary(trace, kind="stats")

    outname = outdir + starname+"_bestfitparams.csv"
    df.to_csv(outname)

    return()

    







def get_bestparams(bestfitfile):
    
    f=open(bestfitfile)
    bestparams=f.readline()
    f.close()
    
    bestparamsdata=bestparams.split()
    chi2=np.float64(bestparamsdata[1])
    dof=np.float64(bestparamsdata[2])-5
    mass=np.float64(bestparamsdata[3])
    en=np.float64(bestparamsdata[4])
    mini=np.float64(bestparamsdata[5])
    mco=np.float64(bestparamsdata[6])
    x_mmix=np.float64(bestparamsdata[7])
    mmix=mini+(mco-mini)*x_mmix
    logf=np.float64(bestparamsdata[8])
    
    
    mni=np.float64(bestparamsdata[9])
    fehmodel=np.float64(bestparamsdata[10])
    logfehabu=fehmodel+(-4.5)   # Soloar logA for Fe
    fehabu=10**logfehabu
    fehabu_mass=fehabu*56
    h_mass=mni/fehabu_mass      
    fehobs=np.float64(bestparamsdata[11])
    
    mrem=mini+(1.0-10**logf)*(mmix-mini)
    
    #mass,en,h_mass,chi2,dof,fehobs
    
    
    return(mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem)



def get_bestparams_CC(bestfitfile):
    
    f=open(bestfitfile)
    bestparams=f.readline()
    f.close()
    
    nparams=6
    
    bestparamsdata=bestparams.split()
    chi2=np.float64(bestparamsdata[1])
    dof=np.float64(bestparamsdata[2])-nparams
    mass=np.float64(bestparamsdata[3])
    en=np.float64(bestparamsdata[4])
    mini=np.float64(bestparamsdata[5])
    mco=np.float64(bestparamsdata[6])
    x_mmix=np.float64(bestparamsdata[7])
    mmix=mini+(mco-mini)*x_mmix
    logf=np.float64(bestparamsdata[8])
    
    
    mni=np.float64(bestparamsdata[9])
    fehmodel=np.float64(bestparamsdata[10])
    logfehabu=fehmodel+(-4.5)   # Soloar logA for Fe
    fehabu=10**logfehabu
    fehabu_mass=fehabu*56
    h_mass=mni/fehabu_mass      
    fehobs=np.float64(bestparamsdata[11])
    ccfrac=np.float64(bestparamsdata[12])
    zcc=np.float64(bestparamsdata[13])
    
    mrem=mini+(1.0-10**logf)*(mmix-mini)
    
    #mass,en,h_mass,chi2,dof,fehobs
    
    
    return(mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem,ccfrac,zcc)


def calc_chi2(zmodel,xfemodel,fehmodel,mass_fe,zobs,xfeobs,errobs,flag,fehobs):

    # Hydrogen dilution mass
    mdil=mass_fe/(10**(fehmodel+np.log10(get_solarm(26)/get_solarm(1))))
       
    chi2=0
    ndata=0
    
    for i,zzobs in enumerate(zobs):
                            
        if (zzobs==11 or zzobs==13) :
            msig=0.5                       
        else:
            msig=0.0
    
        for j,zzmodel in enumerate(zmodel):
            
            if zzobs==zzmodel:
                mabu=xfemodel[j]
                    
                # Observational upper limit or theoretical lower limits
                if flag[i]==-1 or zzmodel==19 or zzmodel==22 or zzmodel==21:    
                    if(xfeobs[i]<mabu):
                        chi2=chi2+((xfeobs[i]+fehobs)-(mabu+fehmodel))**2/(0.1**2+msig**2)
                        ndata=ndata+1
                            
                # Observational lower limit
                elif(flag[i]==-2): 
                    if zzobs==6.5: # Special treatment for C+N
                        if(xfeobs[i]>mabu or mabu>4.0):  # [(C+N)/Fe] should be less than 4.0 
                            chi2=chi2+((xfeobs[i]+fehobs)-(mabu+fehmodel))**2/(0.1**2+msig**2)
                            ndata=ndata+1
                    else:
                        if(xfeobs[i]>mabu):
                            chi2=chi2+((xfeobs[i]+fehobs)-(mabu+fehmodel))**2/(0.1**2+msig**2)
                            ndata=ndata+1
                            
                elif(flag[i]>0):
                    chi2=chi2+((xfeobs[i]+fehobs)-(mabu+fehmodel))**2/(errobs[i]**2+msig**2)
                    ndata=ndata+1
                        
                else:
                    continue

            else:
                continue


    return(chi2,ndata,mdil)






def fit_ml_CC(z0, xfe0, xfeerr0, feh, theta0):
    
    
    zmax = ad.get_zfrac(feh)

    # Parameters are alpha, ZCC, fIa
    alpha_min = -1.0
    alpha_max = 3.0
    bnds = ((alpha_min,alpha_max),(0.00,zmax))
    
    
    filt= (z0!=23) & (z0!=21) & (z0!=22) & (z0!=26)
    
    #filt= (z0!=26)
    
    z = z0[filt]
    xfe = xfe0[filt]
    xfeerr = xfeerr0[filt]
    
    
    
    import scipy.optimize as op
    
    nll = lambda *args: -lnlike_CC(*args)
    #result = op.minimize(nll, theta0, args=(z,xfe,xfeerr),method='L-BFGS-B', bounds=bnds)
    result = op.minimize(nll, theta0, args=(z,xfe,xfeerr),method='SLSQP',bounds=bnds)
    theta = result["x"]
    
    #from scipy.optimize import least_squares
    
    #result=op.leastsq(nll,theta,args=(z,xfe,xfeerr))
    #result = op.minimize(nll, theta, args=(z,xfe,xfeerr),method='BFGS')
    #theta = result["x"]
    
    #print(result)
    #sys.exit()
    
    model_z0,model0,mass_fe = calc_yields_CC(theta)
    
    
    nd=np.size(z)
    dof=nd-np.size(theta0)

    model_z=np.zeros_like(z)
    model=np.zeros_like(xfe)
    for j,xx in enumerate(z):
        model_z[j]=model_z0[model_z0==xx]
        model[j]=model0[model_z0==xx]    
   
    inv_sigma2 = 1.0/xfeerr**2

    
    chi2= np.sum((xfe-model)**2*inv_sigma2)
    
    
    return theta,chi2,dof
    
  



def fit_ml_CC_Ia(z0, xfe0, xfeerr0, feh, theta0,zIa, f_Ch, woSiCa = False, Zn_uplim = False):
    
    
    
    zmax = ad.get_zfrac(feh)
    alpha_min=-1.0
    alpha_max=3.0
    
    # Parameters are alpha,Zcc, ZIa, f_Ia
    bnds = ((alpha_min,alpha_max),(0.00,zmax),(0.0,1.0))
    
    if woSiCa == True:
        filt= (z0!=23) & (z0!=21) & (z0!=22) & (z0!=26) & (z0!=14) & (z0!=20)
    else:
        filt= (z0!=23) & (z0!=21) & (z0!=22) & (z0!=26)
    
    #filt= (z0!=26)
    
    z=z0[filt]
    xfe=xfe0[filt]
    xfeerr=xfeerr0[filt]
    
    #zsun=0.0152
    #if zmax<0.1*zsun:
    #    zIa=0.0
    #else:
    #    zIa=0.1*zsun
    
    
    import scipy.optimize as op
    
    nll = lambda *args: -lnlike_CC_Ia(*args)
    result = op.minimize(nll, theta0, args=(z,xfe,xfeerr,zIa,f_Ch),method='SLSQP', bounds=bnds)
    theta = result["x"]
    
    model_z0,model0,mass_fe,m_Type2,m_TypeIa=calc_yields_CC_Ia(theta,zIa,f_Ch)
    
    
    nd=np.size(z)
    dof=nd-np.size(theta)

    model_z=np.zeros_like(z)
    model=np.zeros_like(xfe)
    for j,xx in enumerate(z):
        model_z[j]=model_z0[model_z0==xx]
        model[j]=model0[model_z0==xx]    
        
   
    inv_sigma2 = 1.0/xfeerr**2

    
    chi2 = np.sum((xfe - model)**2 * inv_sigma2)
    
    if Zn_uplim == 1:
        
        model_zn = model[np.where(z == 30)]
        znfe = xfe[np.where(z == 30)]
        
        if model_zn <= znfe:
            chi2 = np.sum((xfe[np.where(z != 30)] - model[np.where(z != 30)])**2 * inv_sigma2[z != 30])
        else:
            chi2 = np.sum((xfe - model)**2 * inv_sigma2)
    else:
        chi2 = np.sum((xfe - model)**2 * inv_sigma2)
    
    return theta,chi2,dof
    
  


def fit_apogee(starname, catalog, ncores, f_Ch, out_rootdir, Ia = True, CC = True, woSiCa = False, Zn_uplim = False):

    outdir = out_rootdir + "/MCMCresults_fCh%.2f_CC%1i_Ia%1i_woSiCa%1i_Znuplim%1i/"%\
        (f_Ch, int(CC == True), int(Ia == True), int(woSiCa == True), \
         int(Zn_uplim == True))

    if os.path.exists(outdir) == False:
        os.mkdir(outdir)



    elems = ["c", "n", "o", "na", "mg", "al", "si", "ca", "ti", "v", "cr", "mn", "co", "ni", "cu"]

    # Read observational data
    x, data, sigma, mh  = ad.read_APOGEE_xfe(starname, catalog, elems)

    # Convert [M/H] to the mass fraction of metal 
    zcc_max = ad.get_zfrac(mh)


    # create our Op

    if Ia == True:

        zsun=0.0152
        if zcc_max < 0.1*zsun:
            zIa = 0.0
        else:
            zIa = 0.1*zsun


    else:
        zIa = np.nan
        f_Ch = np.nan
        logl = LogLike(lnlike_CC, data, x, sigma, zIa, f_Ch)


    outncname = outdir + starname + "_trace.nc"


    ndraws = 3000  # number of draws from the distribution
    nburn = 1000  # number of "burn-in points" (which we'll discard)

    # use PyMC3 to sample from log-likelihood
    with pm.Model() as model:

        # uniform priors on the parameters
        alpha = pm.Uniform("alpha", lower=-1., upper=3.0)
        zcc = pm.Uniform("zcc", lower=0.0, upper=zcc_max)
        f_Ia = pm.Uniform("f_Ia", lower=0.0, upper=1.0)


        # convert the parameters to a tensor vector
        if Ia == True:
            theta = tt.as_tensor_variable([alpha, zcc, f_Ia])
        else:
            theta = tt.as_tensor_variable([alpha, zcc])


        # use a DensityDist (use a lamdba function to "call" the Op)
        pm.DensityDist("likelihood", lambda v: logl(v), observed={"v": theta})

        #trace = pm.sample(ndraws, tune=nburn, discard_tuned_samples=True)

        start = pm.find_MAP(model = model)
        step = pm.Slice()
        trace = pm.sample(ndraws, step = step,  start = start, \
                      cores = ncores, return_inferencedata = True, \
                      idata_kwargs = {"density_dist_obs": False})


        az.to_netcdf(trace, outncname)


        ax = az.plot_trace(trace)
        fig = ax.ravel()[0].figure
        fig.savefig(outdir + "trace_" + starname + ".png")

        df = az.summary(trace, kind="stats")

    outname = outdir + starname+"_bestfitparams.csv"
    df.to_csv(outname)

    return




if __name__ == "__main__":

    starname = "2M00024677+0127542"
    catalog = "../../../../APOGEE_sample/APOGEE_halo-sample.csv"

    f_Ch = 0.2
    ncores = 4, 
    out_rootdir = "../../../../APOGEE_sample/outputs"
    fit_apogee(starname, catalog, ncores, f_Ch, out_rootdir)







    
