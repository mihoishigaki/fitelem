import os, sys, glob

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib


from astropy.coordinates import SkyCoord, Distance
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import ICRS, Galactic, Galactocentric

import galpy
from galpy.potential import MWPotential2014, McMillan17
from galpy.orbit import Orbit
from galpy.util.bovy_conversion import get_physical

    
from astropy.coordinates import galactocentric_frame_defaults


from astroquery.gaia import Gaia




def calc_ElemMass(z,xmass):

    if z==1:  # H
        mass=np.sum(xmass[0]+xmass[1])
    elif z==2: # He
        mass=np.sum(xmass[2]+xmass[3])
    elif z==3: # Li
        mass=np.sum(xmass[4]+xmass[5])
    elif z==4: # Be
        mass=np.sum(xmass[6])
    elif z==5: # B 
        mass=np.sum(xmass[7]+xmass[8])
    elif z==6: # C
        mass=np.sum(xmass[9]+xmass[10])
    elif z==7: # N
        mass=np.sum(xmass[11]+xmass[12])
    elif z==6.5: # C+N
        mass=np.sum(xmass[9]+xmass[10]+xmass[11]+xmass[12])
    elif z==8: # O
        mass=np.sum(xmass[13]+xmass[14]+xmass[15])
    elif z==9: # F 
        mass=np.sum(xmass[16])
    elif z==10: # Ne
        mass=np.sum(xmass[17]+xmass[18]+xmass[19])
    elif z==11: # Na
        mass=np.sum(xmass[20])

    elif z==12: # Mg
        mass=np.sum(xmass[21]+xmass[22]+xmass[23])    
    elif z==13: # Al
        mass=np.sum(xmass[24])
        
    elif z==14: # Si
        mass=np.sum(xmass[25]+xmass[26]+xmass[27]) 
    elif z==15: # P
        mass=np.sum(xmass[28])
    elif z==16: # S
        mass=np.sum(xmass[29]+xmass[30]+xmass[31]+xmass[32])
    elif z==17: # Cl 
        mass=np.sum(xmass[33]+xmass[34])
    elif z==18: # Ar
        mass=np.sum(xmass[35]+xmass[36]+xmass[37])
    elif z==19: # K 
        mass=np.sum(xmass[38]+xmass[39]+xmass[40])
    elif z==20: # Ca
        mass=np.sum(xmass[41]+xmass[42]+xmass[43]+\
                     xmass[44]+xmass[45]+xmass[46])
    elif z==21: # Sc
        mass=np.sum(xmass[47])
    elif z==22: # Ti
        mass=np.sum(xmass[48]+xmass[49]+xmass[50]+\
                     xmass[51]+xmass[52])    
    elif z==23: # V
        mass=np.sum(xmass[53]+xmass[54])
    elif z==24: # Cr
        mass=np.sum(xmass[55]+xmass[56]+\
                      xmass[57]+xmass[58])
    elif z==25: # Mn
        mass=np.sum(xmass[59])
    elif z==26: # Fe
        mass=np.sum(xmass[60]+xmass[61]+\
                      xmass[62]+xmass[63])
    elif z==27: # Co
        mass=np.sum(xmass[64])
    elif z==28: # Ni
        mass=np.sum(xmass[65]+xmass[66]+xmass[67]+\
                     xmass[68]+xmass[69])  
    
    elif z==29: # Cu
        mass=np.sum(xmass[70]+xmass[71])  
    
    elif z==30: # Zn
        mass=np.sum(xmass[72]+xmass[73]+xmass[74]+\
                     xmass[75]+xmass[76])  
    
    return(mass)


def get_solarm(z):
    
    solarfile='../utility/solarabund/ipcc.dat'
    
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



def select_ohs(catalog_csv, plot = False):


    df0 = pd.read_csv(catalog_csv, dtype = {'wg4_field': str, 'irfm_ebv_ref': str})

    print("GALAHDR3_flagsp0_SNR_MSTO x GaiaDR3 with good abundance: %i"\
          %(len(df0.index)))



    # Select stars with good abundance estimates
    #df_goodabund = select_goodabund(df0)
    
    #print("+ Good abundance: %i"%(len(df_goodabund)))

    
    # Select MSTO stars with good parameter estimate
    #filt = df0['e_age_bstep'] / df0['age_bstep'] < 0.2

    
    #filt = ( df0['e_teff'] < 80. ) & (df0['e_logg'] < 0.2) & (df0['e_fe_h'] < 0.1) \
    #    & (df0['parallax_error_gdr2']/df0['parallax_gdr2'] < 0.1)

    # Select stars with parallax from DR2 and EDR3 are consistent
    #filt = np.abs(df0['parallax_gdr2'] - df0['parallax']) < \
    #    df0['parallax_error']

    #filt = df0['snr_c2_iraf'] > 40.

    #filt = (df0['logg'] > 3.2) & (df0['logg'] < 4.1) & \
    #    (df0['teff'] > 5000.) & (df0['teff'] < 7000.)

    
    #df = df0[filt]
    #print("+ High-quality parameter estimates: %i"%(len(df.index)))


    # Select stars with high-quality astrometry from GaiaEDR3 
    filt = (df0['ruwe'] < 1.4) & \
        (df0['parallax'] > 0.0) & (df0['parallax_error']/df0['parallax'] < 0.1) 

    #print("Negative parallax: ", len(df0['parallax'][df0['parallax'] < 0.0]))

    #print("more than 10% errors: ", len(df0['parallax'][df0['parallax_error']/df0['parallax'] >= 0.1]))


    print("+ RUWE<1.4 and good parallax: %i"%(len(df0[filt].index)))
    

    #df = calc_kinematics(df0[filt], solar_motion = "Sharma20", random = False)
    df = calc_kinematics(df0[filt], random = True)


   

    filt = (df['age_bstep'] > 12.0) # & (df['e_age_bstep'] / df['age_bstep'] < 0.1)
    df_age = df[filt]
    print("+ Age > 12Gyrs: %i"%(len(df_age.index)))
    print("Age uncertainties of all sample: %.2f"%(np.median(df['e_age_bstep'].dropna())))
    print("Age uncertainties of all sample: %.2f"%(np.median(df_age['e_age_bstep'].dropna())))
    

    # Select stars with halo-like kinematics
    df_ohs = select_kinematics(df_age, selection_method = "vtot")

    print("+ Halo kinematics: %i"%(len(df_ohs.index)))

    # Select stars with good abundance estimates
    #df_ohs_goodabund = select_goodabund(df_ohs)
    
    #print("+ Good abundance: %i"%(len(df_ohs_goodabund.index)))



    abuclasses = get_abuclass(df_ohs["fe_h"].values, df_ohs["mg_fe"].values)


    df_ohs["abuclass"] = abuclasses
    
    if plot == True:

        plot_distribution(df, df_age, df_ohs, axislabelcolor = False)
        #plot_lzltan(df)
        #plot_feh(df)


    
    return(df, df_age, df_ohs)




def select_kinematics(df0, selection_method = "vtot", sanity_check = False):


    catalog_dir = "../../data/GALAH_DR3/"
    
    # selection_method
    ##   1. vtot
    ##   2. sharma

    if selection_method == "vtot":


        v_X = df0['v_X']
        v_Y = df0['v_Y']
        v_Z = df0['v_Z']
        e_v_X = df0['e_v_X']
        e_v_Y = df0['e_v_Y']
        e_v_Z = df0['e_v_Z']
        
        
        vtot, e_vtot = get_vtot(v_X, v_Y, v_Z, e_v_X, e_v_Y, e_v_Z)
        df0['vtot'] = vtot
        df0['e_vtot'] = e_vtot
        #filt = vtot + e_vtot > 150. * (u.km / u.s)
        filt = vtot > 150. * (u.km / u.s)
        df_selected = df0[filt]

        

    elif selection_method == "sharma":

    
        # Note that df0 should contain velocities relative to the Galactic center
        # calculated with a set of solar parameters of Sharma+20.


        omega_sun = 30.24 # Reid & Brunthaler 2004
        r_sun = 8.0 # Reid 1993
        Lz_sun = omega_sun * r_sun**2



        print("Minimum [Fe/H]=", np.min(df0['fe_h'].values))

    

        #df= calc_kinematics(df0, random = True)

        df = pd.DataFrame()
    
        for feh in [-1.5, -1.0, -0.5, 0.0]:

            df_Vdisp = pd.read_csv(catalog_dir + \
                                   "Vdisp_Age12.0_FeH%.1f.csv"%(feh))

            if feh == -1.5:
                filt = df0['fe_h'] <= feh + 0.25 # To include stars with [Fe/H] ~ -1.9
            else:
                filt = (df0['fe_h'] > feh - 0.25) & (df0['fe_h'] <= feh + 0.25)

            
            df_feh = (df0[filt]).reset_index(drop = True)
            Lzs = np.round(df_feh['Lz_cyl'].values/Lz_sun/1000., 1)
            Zs = np.round(np.abs(df_feh['Z'].values/1000.), 1)
            v_X = df_feh['v_X'].values
            v_Z = df_feh['v_Z'].values
        
        
            halokin = np.zeros(len(df_feh), dtype = bool)
            sigRs = np.zeros(len(df_feh))
            sigZs = np.zeros(len(df_feh))
        
            for k in range(0, len(df_feh)):


                if Lzs[k] < 0.75:
                    halokin[k] = True
                    continue
            
                filt = (df_Vdisp['Lz_Lsun'].values == Lzs[k]) & (df_Vdisp['Z'].values == Zs[k])
                sigR = df_Vdisp['sigma_R'][filt].values
                sigZ = df_Vdisp['sigma_Z'][filt].values


                if len(sigR)==0:
                    print("The value of Lz = %.1f and Z = %.1f not found in the data."%(Lzs[k], Zs[k]))
                elif len(sigR)>1:
                    print("More than 1 values are found for Lz = %.1f and Z = %.1f."%(Lzs[k], Zs[k]))
                else:

                    if (np.abs(v_X[k] - 1.8) > sigR) & (np.abs(v_Z[k] - 0.58) > sigZ) :

                        halokin[k] = True

                    sigRs[k] = sigR
                    sigZs[k] = sigZ
                    
            
            df_feh['sigR'] = sigRs
            df_feh['sigZ'] = sigZs
            df_feh['halokin'] = halokin


            df = df.append(df_feh)


    
        df.reset_index(inplace = True, drop = True)


        filt = df['halokin'] == True
        df_selected = df[filt]

        
    
    if sanity_check == True:
        df_noerr = calc_kinematics(df_good, random = False)

        
        for ii in range(0, len(df_noerr)):
            if (abs(df_noerr['v_X'][ii] - df_noerr['v_X'][ii]) / \
                abs(df_noerr['v_X'][ii]) > 0.01):
                print(df_noerr['v_X'][ii], df_noerr['v_X'][ii], df_noerr['e_v_X'][ii])

    





    #filt0 = e_vtot/vtot > 0.2
    #print("N stars with large errors = ", len(df_good[filt0]))
    
    #filt = vtot + e_vtot > 150.*u.km/u.s

    

    

    #df  = df_good_plx[filt]


    
    #o= Orbit(c)

    #MWpotential=="MWPotential2014":
    #ts= np.linspace(0.,100.,2001)
    #o.integrate(ts,MWPotential2014)
    
    #print(o.rap, o.rperi, o.e, o.zmax)


    
    return(df_selected)



def select_goodabund(catalog_csv, goodabund_csv):


    df0 = pd.read_csv(catalog_csv)

    

    elems = ["o", "na", "mg", "al", "si", "ca", "sc", "ti", \
             "v", "cr", "mn", "co", "ni", "cu", "zn"]

    df0.loc['N_flag0'] = 0

    df1 = pd.DataFrame(index = [], columns = df0.columns)

    for i in range(0, len(df0.index)):
        
        dfrow = df0.iloc[i]

        ct_flag0 = 0
        for elem in elems:
            if elem == "sc" or elem == "ti":
                continue
            if dfrow["flag_" + elem + "_fe"] == 0:
                ct_flag0 = ct_flag0 + 1

        dfrow.loc['N_flag0'] = ct_flag0
        df1 = df1.append(dfrow, ignore_index = True)
        
        print(" %i/%i stars done"%(i, len(df0.index)))

    filt = df1['N_flag0'] > 5

    df = df1[filt]
    df.to_csv(goodabund_csv)
    
    return()



def write_abundance_data(df, outfilename):


    a = np.zeros_like(np.array(df["fe_h"]))
    a.fill(-9.99)
    aa = np.zeros_like(np.array(df["flag_c_fe"]),dtype=int)
    aa.fill(99)


    

    df["n_fe"] = pd.Series(a,index = df.index)
    df["e_n_fe"] = pd.Series(np.zeros_like(np.array(df["fe_h"])), index = df.index)
    df["flag_n_fe"] = pd.Series(aa,index = df.index)

    #df["ti2_fe"]=pd.Series(a,index=df2.index)
    #df["e_ti2_fe"]=pd.Series(np.zeros_like(np.array(df1_kin["fe_h"])),index=df2.index)
    #df["flag_ti2_fe"]=pd.Series(aa,index=df2.index)

    df["cr2_fe"] = pd.Series(a,index = df.index)
    df["e_cr2_fe"] = pd.Series(np.zeros_like(np.array(df["fe_h"])),index=df.index)
    df["flag_cr2_fe"] = pd.Series(aa,index = df.index)


    df_out = df[["sobject_id_1", "fe_h", "e_fe_h", "flag_fe_h", \
                 "c_fe", "e_c_fe", "flag_c_fe", "n_fe", "e_n_fe", \
                 "flag_n_fe", "o_fe", "e_o_fe", "flag_o_fe", "na_fe", \
                 "e_na_fe", "flag_na_fe", "mg_fe", "e_mg_fe", \
                 "flag_mg_fe", "al_fe", "e_al_fe", "flag_al_fe", \
                 "si_fe", "e_si_fe", "flag_si_fe", "ca_fe", "e_ca_fe", \
                 "flag_ca_fe", "sc_fe", "e_sc_fe", "flag_sc_fe", \
                 "ti_fe", "e_ti_fe", "flag_ti_fe", "ti2_fe", "e_ti2_fe", \
                 "flag_ti2_fe", "v_fe", "e_v_fe", "flag_v_fe", "cr_fe", \
                 "e_cr_fe", "flag_cr_fe", "cr2_fe", "e_cr2_fe", "flag_cr2_fe", \
                 "mn_fe", "e_mn_fe", "flag_mn_fe", "co_fe", "e_co_fe", \
                  "flag_co_fe", "ni_fe", "e_ni_fe", "flag_ni_fe", "cu_fe", \
                 "e_cu_fe", "flag_cu_fe", "zn_fe", "e_zn_fe", "flag_zn_fe", \
                 "y_fe", "e_y_fe", "flag_y_fe", "ba_fe", "e_ba_fe", \
                 "flag_ba_fe", "ra", "dec" , "abuclass"]]
        


    df_out.to_csv(outfilename, index=False, na_rep=-9.99)
    
    return()





def plot_xfe_feh(df, axislabelcolor = False, plot_all_elems = False):


    # Plot settinigs
    plt.rcParams["font.size"] = 16
    cmap = plt.get_cmap("tab10")

    alc = "cornsilk"  # FFF8DC in colorcode

    if axislabelcolor==True:
        matplotlib.rc('axes', edgecolor = alc)
    else:
        matplotlib.rc('axes', edgecolor = 'k')
        
    
    # Elements to be plotted

    if plot_all_elems == True:
        elems = ["c", "o", "na", "mg", "al", "si", "k", "ca", \
                 "sc", "ti", "v", "cr", "mn", "co", "ni", "cu", \
                 "zn", "y", "ba", "la", "eu"]
        iymax = 5
    else:
        
        elems = ["o", "na", "mg", "al", "si", "k", "ca", \
                 "sc", "ti", "v", "cr", "mn", "ni", "cu", \
                 "zn" ]

        iymax = 3
        
    nelems = np.size(elems)


    

    # Initialize a figure

    fig2, bx = plt.subplots(1, 1, figsize = (8, 6))
    
    fig, ax = plt.subplots(iymax + 1, 4, figsize = (20, 13))
    
    ix = 0
    iy = 0

    ymins = np.zeros(nelems)
    ymaxs = np.zeros(nelems)
    

    
    for k, elem in enumerate(elems): 

        if elem == "c" or elem == "y" or elem == "ba" or elem == "la" or elem == "eu":

            ymins[k] = -1.0
            ymaxs[k] = 2.0
            

        elif elem == "o":
            ymins[k] = -0.1
            ymaxs[k] = 1.7
            
        elif elem == "al" or elem == "na" or elem == "mn":
            ymins[k] = -0.5
            ymaxs[k] = 1.3
            
        else:
            ymins[k] = -0.4
            ymaxs[k] = 1.4
    


        textlabel="[" + elem.capitalize() + "/Fe]"
        
        key = elem + "_fe"
        ekey = "e_" + elem + "_fe"
        flagkey = "flag_" + elem + "_fe"


            

        fehs0 = df["fe_h"]
        xfes0 = df[key]
        exfes0 = df[ekey]
        flags0 = df[flagkey]
        classes0 = df['abuclass']
        
        fehs=fehs0[flags0==0]
        xfes=xfes0[flags0==0]
        exfes=exfes0[flags0==0]
        flags=flags0[flags0==0]
        classes = classes0[flags0==0] 


        for kk in range(0, 3):
            if kk==0:
                
                filt = classes == "high-alpha"
                lab = r"High-$\alpha$"
                mk = "x"
    
            elif kk==1:
                filt = classes == "low-alpha"
                lab = r"Low-$\alpha$"
                mk = "^" 
            elif kk==2:
                filt = classes == "metal-poor"
                lab = "Metal-poor"
                mk = "o"

                
            ax[iy,ix].errorbar(fehs[filt], xfes[filt], marker=mk,linestyle="", color =cmap(kk), \
                               ms=10, label = lab, alpha = 0.8)
            if elem == "mg":
                bx.errorbar(fehs[filt], xfes[filt], marker=mk,linestyle="", color = cmap(kk), \
                            ms=10, label = lab)

        



                x = np.arange(-1.8, 0.0, 0.1)
                bx.plot(x, abuclass_func(x), \
                        marker = '', linestyle = ":", \
                        color = 'k', lw = 3)

            
        ax[iy,ix].set_ylim(ymins[k], ymaxs[k])
        ax[iy,ix].set_xlim(-2.5, -0.1)
        ax[iy,ix].text(-2.4, ymaxs[k] - (ymaxs[k] - ymins[k]) * 0.13, textlabel)
    


        

        ticks=np.arange(-2.5, 0.0, 0.5)

        
        if plot_all_elems == True:


            
            if (iy == iymax and ix == 0) or (iy == iymax-1 and ix > 0):
                ax[iy,ix].set_xlabel("[Fe/H]")

                ax[iy,ix].set_xticks(ticks) 
                ax[iy,ix].set_xticklabels(ticks, fontsize=13)
            else:
                ax[iy,ix].xaxis.set_visible(False)
 
            for ii in range(1,4):
                ax[iymax, ii].set_axis_off()
                
        else:
            if (iy == iymax and ix < 3) or (iy == iymax -1 and ix == 3):
                ax[iy,ix].set_xlabel("[Fe/H]")
                
                ax[iy,ix].set_xticks(ticks) 
                ax[iy,ix].set_xticklabels(ticks, fontsize=16)
            else:
                ax[iy,ix].xaxis.set_visible(False)
            
            ax[iymax, 3].set_axis_off()
                
        
        if ix==0:
            ax[iy,ix].set_ylabel("[X/Fe]") 
            

        if axislabelcolor == True:

            ax[iy, ix].tick_params(axis = 'both', which = 'both', reset = True, \
                              color = alc, labelcolor = alc)
            ax[iy, ix].yaxis.label.set_color(alc)
            ax[iy, ix].xaxis.label.set_color(alc)

    
        if ix==3:
            iy=iy+1
            ix=0
        else:
            ix=ix+1

    #for ii in range(1,4):
    #ax[0, 3].set_axis_off()
    
    ax[3, 2].legend(loc='upper right',bbox_to_anchor=(1.8,0.5),prop={"size":17})

    
    plt.subplots_adjust(wspace = 0.17, hspace = 0.05)

    if axislabelcolor == True:
        fig.savefig("../figs/XFe_FeH_alc.png")
    else:
    
        fig.savefig("../figs/XFe_FeH.pdf")
        fig.savefig("../figs/XFe_FeH.png")

    return()


def plot_bay(df):


    # Plot settinigs
    plt.rcParams["font.size"] = 16
    cmap = plt.get_cmap("tab10")


    
    fig, ax = plt.subplots(2, 1, sharex = True, figsize = (6, 8))

    ymins = np.zeros(2)
    ymaxs = np.zeros(2)
    
    elems = ["y", "ba"]

    for i, elem in enumerate(elems):

        if i == 0:
            elem == "y"
            ymins[i] = -0.7
            ymaxs[i] = 1.0
        else:
            elem == "ba"
            ymins[i] = -0.7
            ymaxs[i] = 1.0
            
        textlabel="[" + elem.capitalize() + "/Fe]"
        
        key = elem + "_fe"
        ekey = "e_" + elem + "_fe"
        flagkey = "flag_" + elem + "_fe"

        fehs0 = df["fe_h"]
        xfes0 = df[key]
        exfes0 = df[ekey]
        flags0 = df[flagkey]
        classes0 = df['abuclass']
        
        fehs=fehs0[flags0==0]
        xfes=xfes0[flags0==0]
        exfes=exfes0[flags0==0]
        flags=flags0[flags0==0]
        classes = classes0[flags0==0] 

        for kk in range(0, 3):
            if kk==0:
                
                filt = classes == "high-alpha"
                lab = "high-alpha"
                mk = "x"
    
            elif kk==1:
                filt = classes == "low-alpha"
                lab = "Low-alpha"
                mk = "^" 
            elif kk==2:
                filt = classes == "metal-poor"
                lab = "Metal-poor"
                mk = "o"

                
            ax[i].errorbar(fehs[filt], xfes[filt], marker=mk,linestyle="", color =cmap(kk), \
                               ms=10, label = lab, alpha = 0.8)

        ax[i].set_ylim(ymins[i], ymaxs[i])
        ax[i].set_xlim(-2.5, -0.1)
        ax[i].text(-2.4, ymaxs[i] - (ymaxs[i] - ymins[i]) * 0.13, textlabel)
        ax[i].set_ylabel("[X/Fe]")

        
    ax[1].set_xlabel("[Fe/H]")
    #plt.subplots_adjust(hspace = 0.0, wspace = 0.1)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    plt.savefig('../figs/Ba_Y.pdf')
    


    return()






def plot_xfe_feh_old(df, df_age, df_ohs, plot_all_elems = False):


    # Plot settinigs
    plt.rcParams["font.size"] = 11
    cmap = plt.get_cmap("tab10")


    
    # Elements to be plotted

    if plot_all_elems == True:
        elems = ["c", "o", "na", "mg", "al", "si", "k", "ca", \
                 "sc", "ti", "v", "cr", "mn", "co", "ni", "cu", \
                 "zn", "y", "ba", "la", "eu"]
        iymax = 5
    else:
        
        elems = ["o", "na", "mg", "al", "si", "k", "ca", \
                 "sc", "ti", "v", "cr", "mn", "ni", "cu", \
                 "zn" ]

        iymax = 3
        
    nelems = np.size(elems)


    

    # Initialize a figure

    fig, bx = plt.subplots(1, 1, figsize = (8, 6))
    
    fig, ax = plt.subplots(iymax + 1, 4, figsize = (15,12))
    
    ix = 0
    iy = 0

    ymins = np.zeros(nelems)
    ymaxs = np.zeros(nelems)
    

    
    for k, elem in enumerate(elems): 

        if elem == "c" or elem == "y" or elem == "ba" or elem == "la" or elem == "eu":

            ymins[k] = -1.0
            ymaxs[k] = 2.0
            

        elif elem == "o":
            ymins[k] = -0.1
            ymaxs[k] = 1.7
            
        elif elem == "al" or elem == "na" or elem == "mn":
            ymins[k] = -0.5
            ymaxs[k] = 1.3
            
        else:
            ymins[k] = -0.4
            ymaxs[k] = 1.4
    


        textlabel="[" + elem.capitalize() + "/Fe]"
        
        key = elem + "_fe"
        ekey = "e_" + elem + "_fe"
        flagkey = "flag_" + elem + "_fe"


        for kk in range(0, 3):

            if kk == 0:

                df = df

            elif kk == 1:

                df = df_age

            elif kk == 2:

                df = df_ohs

            

            fehs0 = df["fe_h"]
            xfes0 = df[key]
            exfes0 = df[ekey]
            flags0 = df[flagkey]

            fehs=fehs0[flags0==0]
            xfes=xfes0[flags0==0]
            exfes=exfes0[flags0==0]
            flags=flags0[flags0==0]

            #if kk == 0:
                
            #    counts, xedges, yedges, Image = ax[iy,ix].hist2d(fehs, xfes, bins=60, cmap='Greys')
            #    levels = np.linspace(10.0, np.max(counts), 6)
            #    ax[iy,ix].contour(counts.transpose(), extent = \
            #                      [xedges.min(), xedges.max(), yedges.min(), yedges.max()], \
            #                      linewidth = 1, levels = levels, colors = '#606060')
    
            if kk == 0:

                ax[iy,ix].errorbar(fehs, xfes, marker='.', \
                                   linestyle="", color =cmap(0), \
                               ms=2, zorder=7,label="All MSTO, >5 elements, GEDR3")

                if elem == "mg":
                    bx.errorbar(fehs, xfes, marker='.',linestyle="", \
                                color =cmap(0), \
                               ms=2, zorder=7,label="All MSTO, >5 elements, GEDR3")

                
            elif kk == 1:


                ax[iy,ix].errorbar(fehs, xfes, \
                           marker = 'o', linestyle = "", mfc = cmap(1), mec = cmap(1), \
                           ecolor = cmap(1), ms = 3, zorder = 9, label = " + Age>12Gyrs")

                if elem == "mg":
                    bx.errorbar(fehs, xfes, \
                           marker = 'o', linestyle = "", mfc = cmap(1), mec = cmap(1), \
                           ecolor = cmap(1), ms = 3, zorder = 9, label = " + Age>12Gyrs")
                    
                

            elif kk == 2:

                ax[iy,ix].errorbar(fehs, xfes, yerr = exfes,\
                           marker = 's', linestyle = "", mfc = "none", mec = cmap(3),\
                           ecolor = cmap(3),ms = 6,zorder = 11,label = " + halo kinematics")
        
        

                if elem == "mg":
                    bx.errorbar(fehs, xfes, yerr = exfes,\
                                marker = 's', linestyle = "", mfc = "none", mec = cmap(3),\
                                ecolor = cmap(3),ms = 6,zorder = 11,label = " + halo kinematics")
        
                    
                
                
                
                
        #ax[iy,ix].errorbar(fehs, xfes, yerr = exfes, marker='o', \
        #                   linestyle = "", mfc = cmap(0), mec = cmap(0),\
        #                   ecolor = cmap(0), ms = 2, elinewidth = 1, \
        #                   zorder = 7, label = "Age>12Gyrs")


        if elem == "mg":
            #ax[iy, ix].plot([-1.2, -1.2], [-1., 2.], \
            #                marker = '', linestyle = ":", \
            #                color = cmap(4), lw = 2)

            x = np.arange(-1.8, 0.0, 0.1)
            bx.plot(x, abuclass_func(x), \
                            marker = '', linestyle = ":", \
                            color = cmap(4), lw = 2)

            
        ax[iy,ix].set_ylim(ymins[k], ymaxs[k])
        ax[iy,ix].set_xlim(-2.5, -0.1)
        ax[iy,ix].text(-2.4, ymaxs[k] - (ymaxs[k] - ymins[k]) * 0.13, textlabel)
    


        

        ticks=np.arange(-2.5, 0.0, 0.5)

        if plot_all_elems == True:


            
            if (iy == iymax and ix == 0) or (iy == iymax-1 and ix > 0):
                ax[iy,ix].set_xlabel("[Fe/H]")

                ax[iy,ix].set_xticks(ticks) 
                ax[iy,ix].set_xticklabels(ticks, fontsize=11)
            else:
                ax[iy,ix].xaxis.set_visible(False)
 
            for ii in range(1,4):
                ax[iymax, ii].set_axis_off()
                
        else:
            if (iy == iymax and ix < 3) or (iy == iymax -1 and ix == 3):
                ax[iy,ix].set_xlabel("[Fe/H]")
                
                ax[iy,ix].set_xticks(ticks) 
                ax[iy,ix].set_xticklabels(ticks, fontsize=11)
            else:
                ax[iy,ix].xaxis.set_visible(False)
            
            ax[iymax, 3].set_axis_off()
                
        
        if ix==0:
            ax[iy,ix].set_ylabel("[X/Fe]") 
            
    
    
        if ix==3:
            iy=iy+1
            ix=0
        else:
            ix=ix+1

    

    
    plt.show()

    return()



def plot_toomre(df_all, df_age, df_ohs, df_good_abund, axislabelcolor = False):


    # Figure setting
    plt.rcParams["font.size"] = 12
    cmap = plt.get_cmap("tab10")
    alc = "cornsilk"  # FFF8DC in colorcode



    #fig = plt.figure(figsize=(7, 3))


    
    for k in range(0, 4):

        if k == 0:
            df = df_all

        elif k == 1:
            df = df_age

        elif k == 2:
            df = df_ohs

        elif k == 3:
            df = df_good_abund

        X, Y, Z, v_X, v_Y, v_Z = calc_kinematics(df)


        nstars = np.size(X)
        
        vtan = np.sqrt(v_X**2 + v_Z**2)
        
        if k == 0:

            counts, xedges, yedges, Image = \
                plt.hist2d(np.array(v_Y), np.array(vtan), bins=80, \
                           cmap='Greys', density=False)


            levels = np.linspace(10.0, np.max(counts), 6)
            plt.contour(counts.transpose(), extent = \
                        [xedges.min(),xedges.max(),yedges.min(),yedges.max()], \
                        linewidth = 1, levels = levels, colors = '#606060')
    

        elif k == 1:

            plt.plot(v_Y, vtan, marker = "o", linestyle = "", mfc = cmap(0), mec = cmap(0), \
                     ms = 3, label = "Age>12Gyrs: %i"%(nstars), linewidth = 2)

        elif k == 2:

            plt.plot(v_Y, vtan, marker = "o", linestyle = "", mfc = "none", \
                     mec = cmap(1), ms = 8,label = " + halo kinematics: %i"%(nstars), \
                     linewidth = 2)

        elif k == 3:

            plt.plot(v_Y, vtan, marker = "s", linestyle = "", linewidth = 1, \
                     mec = cmap(3), mfc = "none", ms = 11, label = " + >5 elements: 5")


    plt.show()
            
    return()



def plot_cmd(df_all, df_age, df_ohs, isochrone, axislabelcolor = False):


    # Plot setting:
    plt.rcParams["font.size"] = 16
    cmap = plt.get_cmap("tab10")
    alc = "cornsilk"  # FFF8DC in colorcode

    # Prepare for the isochrones: 
    cols = ['k', 'k', 'k', 'k']
    lss = ["-.", "--", ":", "-"]

    if isochrone == "Padova":
        labs = ["5.3Gyr", "10.0Gyr", "11.5Gyr", "13.2Gyr"]
        isoages = [ 5.25e+9, 1.00e+10, 1.15e+10, 1.32e+10 ]
        isopath =  "../utility/isochrone/Padova_logt9.00-10.13/"
        isofiles = ["output_Z0.00048.dat","output_Z0.00152.dat","output_Z0.00481.dat"]
    elif isochrone == "DM":
        labs = ["5Gyr", "10Gyr", "12Gyr", "14Gyr"]
        labsalpha = ["5Gyr", "10Gyr", "12Gyr", "14Gyr"]
        isoages = [5, 10, 12, 14]
        isopath = "../utility/isochrone/Dartmouth/"


        
    
    # Plot setting for the sample
    ecols = [cmap(7), cmap(9), cmap(3)]
    fcols = [cmap(7), cmap(9), 'none']
    mss = [2, 6, 10]
    mks = ['.', 'x', 'o']
    samplelabs = ['All selected', '+ Age > 12Gyrs', '+ Halo kinematics']


    
    fehbins = [-1.5, -1.0, -0.5]
    fehw = 0.25



    fig = plt.figure(constrained_layout = True, \
                     figsize = (np.size(fehbins) * 5, 10))
    gs = fig.add_gridspec(ncols = 3, nrows = 2, \
                          width_ratios = [1., 1., 1.], \
                          height_ratios = [1., 7.])
    
    #fig,ax = plt.subplots(1, np.size(fehbins), sharey=True, \
    #                    figsize=(np.size(fehbins) * 5, 8))


    if axislabelcolor == True:

        matplotlib.rc('axes', edgecolor = alc)
        outfig = "../figs/CMD_axiscolor.pdf"
    else:
        outfig = "../figs/CMD.pdf"
    
    
    for i, fehbin in enumerate(fehbins):

        ax = fig.add_subplot(gs[1, i])
        # Plot isochrones
        
        if isochrone == "Padova":
            
            isofile = isopath + isofiles[i]     
            isodata = pd.read_csv(isofile,sep = '\s+', comment = '#', header = 0)

            for k in range(0,4):

                filt = isodata["Age"] == isoages[k]
                iso = isodata[filt]

                iso_teff = 10**iso["logTe"]
                iso_logg = iso["logg"]
                ax.plot(iso_teff, iso_logg, marker = '', linestyle = lss[k],\
                           color = cols[k], label = labs[k])
   
        elif isochrone == "DM":

            for k in range(0, 4):

                isofile = glob.glob(isopath + "feh%.1f/alpha0.0/a%02i000*.iso"%(fehbin, isoages[k]))

                liso_teff, iso_logg = np.loadtxt(isofile[0], usecols = (2, 3), unpack = True)
                iso_teff = 10**liso_teff
                ax.plot(iso_teff, iso_logg, marker = '', linestyle = lss[k],\
                           color = cols[k], lw = 2, label = labs[k])

                isofile = glob.glob(isopath + "feh%.1f/alpha0.4/a%02i000*.iso"%(fehbin, isoages[k]))
                
                liso_teff, iso_logg = np.loadtxt(isofile[0], usecols = (2, 3), unpack = True)
                iso_teff = 10**liso_teff
                ax.plot(iso_teff, iso_logg, marker = '', linestyle = lss[k],\
                           color = cmap(2), lw = 2, label = labsalpha[k])
            
        
        fehmax = fehbin + fehw
        fehmin = fehbin - fehw


        for kk in range(0, 4):


            if kk == 0:
                df = df_all
                
            elif kk == 1:
                df = df_age
                
            elif kk == 2:
                df = df_ohs
                
            #elif kk == 3:
                #df = df_good_abund


            fehfilt = (df["fe_h"] > fehmin) & (df["fe_h"] <= fehmax)
     

            if kk == 0:
            
                if i >= 2:
                    counts, xedges, yedges, Image = \
                        ax.hist2d(df["teff"][fehfilt], df["logg"][fehfilt], \
                                     bins = 40, cmap = 'Greys')
                    levels = np.linspace(10.0, np.max(counts), 5)
                    ax.contour(counts.transpose(), extent = \
                                  [xedges.min(), xedges.max(), yedges.min(), yedges.max()], \
                                  linewidth = 1,levels = levels, colors = '#606060')
    
                else:
                    ax.plot(df["teff"][fehfilt], df["logg"][fehfilt], linestyle = '',\
                               marker = "o", mfc = cmap(7), mec = cmap(7), ms = 3, \
                               label = samplelabs[kk])

            elif kk == 1:
            
                ax.plot(df["teff"][fehfilt], df["logg"][fehfilt],\
                               linestyle = '', marker = mks[kk], \
                           mec = ecols[kk], mfc = fcols[kk], ms = mss[kk],
                              label = samplelabs[kk])
                e_teff_med = np.median(df["e_teff"][fehfilt])
                e_logg_med = np.median(df["e_logg"][fehfilt])


            elif kk ==2:
                ax.plot(df["teff"][fehfilt], df["logg"][fehfilt], \
                           linestyle = '', marker = mks[kk], \
                           mec = ecols[kk], mfc = fcols[kk], ms = mss[kk],
                               label = samplelabs[kk])

              

                
            
        xmax = 7800.
        xmin = 4950.
        ax.text(xmax - (xmax - xmin) * 0.05, 3.25, \
                   "$%.1f<$[Fe/H]$<%.1f$"%(fehmin, fehmax))
        ax.set_xlim(7800, xmin)
        ax.set_ylim(4.1, 3.2)
        ax.set_xlabel(r"$T_{eff}$ [K]")
        ax.errorbar([7600.], [3.9], xerr = e_teff_med, yerr = e_logg_med, \
                       ecolor = ecols[1], marker = mks[1], mfc = fcols[1], \
                       mec = ecols[1], ms = mss[1])

        
        
        if i == 0:
            hans, labs = ax.get_legend_handles_labels()
            ax.set_ylabel(r"$\log g$ [dex]")
        else:
            ax.axes.yaxis.set_ticklabels([])


        if axislabelcolor == True:

            ax.tick_params(axis='both', which = 'both', colors = alc)
            ax.set_xlabel(r"$T_{eff}$ [K]", color = alc)
            if i == 0:
                ax.set_ylabel(r"$\log g$ [dex]", color = alc)
                

    print(hans, labs)
    bx = fig.add_subplot(gs[0, 0])
    bx.legend(handles = [hans[0], hans[2], hans[4], hans[6]], \
                 labels = [labs[0], labs[2], labs[4], labs[6]], \
                title = r"$\alpha$-solar", loc = 'upper left', ncol = 2, frameon = False)
    bx.axis('off')
    bx = fig.add_subplot(gs[0, 1])
    bx.legend(handles = [hans[1], hans[3], hans[5], hans[7]], \
              labels = [labs[1], labs[3], labs[5], labs[7]], \
              title = r"$\alpha$-enhanced", loc = 'upper left', ncol = 2, frameon = False)
    bx.axis('off')
    bx = fig.add_subplot(gs[0, 2])
    bx.legend(handles=hans[8:], labels=labs[8:], loc = 'upper left', ncol = 1, frameon = False)
    bx.axis('off')


    plt.subplots_adjust(wspace = 0.05)
    plt.savefig(outfig)
    plt.show()

    
    return()






def load_isochrones(fehbins, isotype = "Padova"):


    if isotype == "Padova":


        for feh in [-0.6, -1.0, -1.4, -1.9, -2.3]:
            Z = ohs.get_zfrac(feh)
            isofile = "output_GaiaEDR3_Z%.5f"%(Z)

            
            print(" %.5f"%(Z))
        
    #    isopath = "../../isochrone/Padova_logt9.00-10.13/"
        



    #    isofiles[i]     
    #    isodata = pd.read_csv(isofile,sep = '\s+', comment = '#', header = 0)

        


        
    #    elif isotype == "Dartmath":

    #        elif isotype == "YY":

    return()


def register_Sharma20():

    state = galactocentric_frame_defaults.get_from_registry("v4.0")
    rsun = 8.0 * u.kpc
    omega_sun = 30.24 * (u.km / u.s / u.kpc)
    usun = 10.96 * (u.km / u.s)
    wsun = 7.53 * (u.km / u.s)
    vsun = rsun * omega_sun 
    state["parameters"]["galcen_v_sun"] = (usun, vsun, wsun) 
    state["references"]["galcen_v_sun"] = "https://ui.adsabs.harvard.edu/abs/2020arXiv200406556S"
    state["parameters"]["galcen_distance"] = rsun
    state["references"]["galcen_distance"] = "https://ui.adsabs.harvard.edu/abs/2020arXiv200406556S"
    state["parameters"]["z_sun"] = 25.0 * (u.pc)
    state["references"]["z_sun"] = "https://ui.adsabs.harvard.edu/abs/2020arXiv200406556S"
    
    galactocentric_frame_defaults.register(name="Sharma20", **state)

    return(state)









def calc_orbit(df, plot, MWpotential):
    
 


    # Galactocentric frame definition
    
    _ = galactocentric_frame_defaults.set('v4.0') 
    
    
    epoch = df['ref_epoch'].values
  
    ra = df['ra'].values
    dec = df['dec'].values

    plx = df['parallax'].values 
    pmra = df['pmra'].values
    pmdec = df['pmdec'].values
    rv = df['rv_galah'].values


    c = SkyCoord(ra = ra*u.deg, dec = dec*u.deg, \
                 distance = Distance(parallax = plx * u.mas),
                 pm_ra_cosdec = pmra*u.mas/u.yr, \
                 pm_dec = pmdec*u.mas/u.yr,
                 radial_velocity = rv*u.km/u.s, \
                 obstime = Time(epoch, format = 'jyear'))
    

    
    #gc=c.transform_to(Galactocentric)
    
    
    #X=gc.x
    #Y=gc.y
    #Z=gc.z
    #v_X=gc.v_x
    #v_Y=gc.v_y
    #v_Z=gc.v_z
    
    #Lz=X*v_Y-Y*v_X
 

    o = Orbit(c)

    # For MW14
    if MWpotential=="MWPotential2014":
        ts= np.linspace(0.,100.,2001)
        o.integrate(ts,MWPotential2014)
    elif MWpotential=="McMillan17":
        ts= np.linspace(0,10.,10000)*u.Gyr
        o.integrate(ts,McMillan17)



    for i, indx in enumerate(df.index):

        starname = df.loc[indx, 'sobject_id_1']
    
        Lz = df.loc[indx, 'Lz']
        if plot==1:

 
            plt.rcParams["font.size"] = 13
            cmap = plt.get_cmap("tab10")

            o[i].plot([o.R()[i]],[o.z()[i]],'ro',label=[starname])
            plt.legend()
            plt.savefig('../figs/orbits/Orbit_'+np.str(starname)+'.png')
       
            xx=o.x(ts)[i]
            yy=o.y(ts)[i]
            zz=o.z(ts)[i]
            rr=o.R(ts)[i]
  
            fig,ax=plt.subplots(1,2) 

            ax[0].plot(xx,yy,linestyle='-',color=cmap(0),marker='')
            ax[0].plot([o.x()[i]],[o.y()[i]],marker='o',color=cmap(1))
            ax[1].plot(rr,zz,linestyle='-',color=cmap(0),marker='')
            ax[1].plot( [o.R()[i]],[o.z()[i]],marker='o',color=cmap(1))

            ax[0].set_xlim(-20.,20.)
            ax[0].set_ylim(-20.,20.)
            ax[0].set_xlabel("X [kpc]")
            ax[0].set_ylabel("Y [kpc]")
            ax[1].set_xlim(4.,20.)
            ax[1].set_ylim(-8.,8.)
            ax[1].set_xlabel("R [kpc]")
            ax[1].set_ylabel("Z [kpc]")
            ax[0].set_aspect('equal')
            ax[1].set_aspect('equal')
            plt.tight_layout()
            plt.savefig('../figs/orbits/XY_RZ_'+np.str(starname)+'.pdf')


    
        f = open('../outputs/Orbits/orbitalparams_'+np.str(starname)+'.txt','w')
        f.write("Rapo, Rapo_ana, Rperi, Rperi_ana, e, e_ana, zmax, zmax_ana, E, Lz\n")
        
        rapo=o.rap()[i]
        rapo_ana=o.rap(analytic=True)[i]
        rperi=o.rperi()[i]
        rperi_ana=o.rperi(analytic=True)[i]
        e=o.e()[i]
        e_ana=o.e(analytic=True)[i]
        zmax=o.zmax()[i]
        zmax_ana=o.zmax(analytic=True)[i]
        En=o.E()[i]

        outtext="%.3f, %.3f, %.3f, %.3f, %.3f, %.3f,%.3f, %.3f, %.4f,%.4f\n"%\
            (rapo,rapo_ana,rperi,rperi_ana,e,e_ana,zmax,zmax_ana,En,Lz)
        f.write(outtext)
        f.close()
    

    return()



def plot_E_Lz(df, axislabelcolor = False):

    #catalog="GALAH_Sanders18_GaiaDR2_ParallaxE10percent_rvmatch_MSTO_relage20percent_Age.csv"
    ##catalog="GALAH_Sanders18_GaiaDR2_ParallaxE10percent_rvmatch_MSTO_relage20percent_Kinematics.csv"
    
    plt.rcParams["font.size"] = 16
    cmap = plt.get_cmap("tab10")
    alc = "cornsilk"  # FFF8DC in colorcode

    if axislabelcolor==True:
        matplotlib.rc('axes', edgecolor = alc)

        
    fig, ax=plt.subplots(1,2,figsize=(12,6))

   
    

    
    
    
    Lzs0 = ()
    Es0 = ()
    rapos0 = ()
    zmaxs0 = ()
    abuclasses0 = ()

    for indx in df.index:

        starname = df.loc[indx, 'sobject_id_1']
        abuclass = df.loc[indx, 'abuclass']
    
        
        orbitfile='../outputs/Orbits/orbitalparams_'+np.str(starname)+'.txt'
        if os.path.isfile(orbitfile)==False:
            print('File not found!')
            break
            
        #f=open(orbitfile)
        #data=(f.readlines())[1].split(',')
        #f.close()

        df_o = pd.read_csv(orbitfile)

        if 'Lz' in df_o:
            lz = df_o['Lz']*(-1)/1000.
            en = df_o['E']

            
        else:
            lz = np.nan
            en = np.nan
            
        rapo = df_o['Rapo']
        zmax = df_o['zmax']
        
        #if np.size(data)<8:
        #    lz = np.nan
        #    en = np.nan
        #else:
        #    lz = np.float((data[9]).strip())*(-1)/1000.
        #    en = np.float(data[8])

        #rapo = np.float(data[0])
        #zmax = np.float(data[6])
            
            
        Lzs0 = np.append(Lzs0, lz)
        Es0 = np.append(Es0, en)
        rapos0 = np.append(rapos0, rapo)
        zmaxs0 = np.append(zmaxs0, zmax)
        abuclasses0 = np.append(abuclasses0, abuclass)

        
    filt= (~np.isnan(Lzs0))
    Lzs=Lzs0[filt]
    Es=Es0[filt]
    rapos=rapos0[filt]
    zmaxs=zmaxs0[filt]
    abuclasses = abuclasses0[filt]


    print(len(Lzs0))
    print(len(Lzs))

    
    abucs = ["high-alpha", "low-alpha", "metal-poor"]
    labs = [r"High-$\alpha$", r"Low-$\alpha$", r"Metal-poor"]
    mks = ["x", "^", "o"]
    
    for k, abuc in enumerate(abucs):

        filt = abuclasses == abuc
        
        ax[0].plot(Lzs[filt], Es[filt], marker = mks[k], linestyle = "", \
                   mfc = cmap(k), mec = cmap(k), ms = 10, linewidth = 2, \
                   alpha = 0.8, label = labs[k])
        ax[1].plot(zmaxs[filt],rapos[filt], marker=mks[k], linestyle="", \
                   mfc = cmap(k), mec = cmap(k), ms = 10, linewidth = 2, \
                   alpha = 0.8, label = labs[k])
    
    
  
    ax[0].set_xlim(-1800., 2900.)
    ax[0].set_ylim(-70000., 0.)
    ax[0].set_xlabel("$L_{z}$ [km kpc s$^{-1}$]")
    ax[0].set_ylabel("$E$ [km$^{2} s^{-2}$]")
    ax[1].legend(loc = 1)
    
    ax[1].set_xlabel("$Z_{max}$ [kpc]")
    ax[1].set_ylabel("$R_{apo}$ [kpc] ")


    if axislabelcolor == True:

        for i in range(0, 2):
            ax[i].tick_params(axis = 'both', which = 'both', reset = True, \
                              color = alc, labelcolor = alc)
            ax[i].yaxis.label.set_color(alc)
            ax[i].xaxis.label.set_color(alc)

        
        outfig = '../figs/E_Lz_Zmax_Rapo_alc.pdf'
    else:
        outfig = '../figs/E_Lz_Zmax_Rapo.pdf'
    
    plt.tight_layout()
    plt.savefig(outfig)
    
    return




def calc_kinematics(df, solar_motion = "default", random = False):


    if solar_motion == "Sharma20":
        
        _ = register_Sharma20()
        _ = galactocentric_frame_defaults.set('Sharma20') 

    else:
        _ = galactocentric_frame_defaults.set('v4.0') 
        
    

    ndata = len(df)
    
    epoch = df['ref_epoch'].values
  
    ra = df['ra'].values
    dec = df['dec'].values


    if random == False:
    
        plx = df['parallax'].values 
        pmra = df['pmra'].values
        pmdec = df['pmdec'].values
        rv = df['rv_galah'].values


        c = SkyCoord(ra = ra*u.deg, dec = dec*u.deg, \
                     distance = Distance(parallax = plx * u.mas),
                     pm_ra_cosdec = pmra*u.mas/u.yr, \
                     pm_dec = pmdec*u.mas/u.yr,
                     radial_velocity = rv*u.km/u.s, \
                     obstime = Time(epoch, format = 'jyear'))


        gc = c.transform_to(Galactocentric)

        X = gc.x
        Y = gc.y
        Z = gc.z
        v_X = gc.v_x
        v_Y = gc.v_y
        v_Z = gc.v_z

        e_X = np.zeros(ndata)
        e_Y = np.zeros(ndata)
        e_Z = np.zeros(ndata)
        e_v_X = np.zeros(ndata)
        e_v_Y = np.zeros(ndata)
        e_v_Z = np.zeros(ndata)

        

    else:
        
        N = 1000
        
        X0 = np.array([np.zeros(len(df))] * N)
        Y0 = np.array([np.zeros(len(df))] * N)
        Z0 = np.array([np.zeros(len(df))] * N)
        v_X0 = np.array([np.zeros(len(df))] * N)
        v_Y0 = np.array([np.zeros(len(df))] * N)
        v_Z0 = np.array([np.zeros(len(df))] * N)
        
        for i in range(0, N):
        
            plx = np.random.normal(df['parallax'].values, df['parallax_error'].values, ndata)
            pmra = np.random.normal(df['pmra'].values, df['pmra_error'].values, ndata)
            pmdec = np.random.normal(df['pmdec'].values, df['pmdec_error'].values, ndata)
            rv = np.random.normal(df['rv_galah'].values, df['e_rv_galah'].values, ndata)

        
            c = SkyCoord(ra = ra*u.deg, dec = dec*u.deg, \
                         distance = Distance(parallax = plx * u.mas),
                         pm_ra_cosdec = pmra*u.mas/u.yr, \
                         pm_dec = pmdec*u.mas/u.yr,
                         radial_velocity = rv*u.km/u.s, \
                         obstime = Time(epoch, format = 'jyear'))


            gc = c.transform_to(Galactocentric)

            X0[i, :] = gc.x
            Y0[i, :] = gc.y
            Z0[i, :] = gc.z
            v_X0[i, :] = gc.v_x
            v_Y0[i, :] = gc.v_y
            v_Z0[i, :] = gc.v_z

        X = np.mean(X0, axis = 0) *u.pc
        e_X = np.std(X0, axis = 0) *u.pc
        Y = np.mean(Y0, axis = 0) *u.pc
        e_Y = np.std(Y0, axis = 0) *u.pc
        Z = np.mean(Z0, axis = 0) *u.pc
        e_Z = np.std(Z0, axis = 0) *u.pc
        v_X = np.mean(v_X0, axis = 0) *u.km/u.s
        e_v_X = np.std(v_X0, axis = 0) *u.km/u.s
        v_Y = np.mean(v_Y0, axis = 0) *u.km/u.s
        e_v_Y = np.std(v_Y0, axis = 0) *u.km/u.s
        v_Z = np.mean(v_Z0, axis = 0) *u.km/u.s
        e_v_Z = np.std(v_Z0, axis = 0) *u.km/u.s



    df['X'] = X
    df['Y'] = Y
    df['Z'] = Z
    df['v_X'] = v_X
    df['v_Y'] = v_Y
    df['v_Z'] = v_Z
    df['e_X'] = e_X
    df['e_Y'] = e_Y
    df['e_Z'] = e_Z
    df['e_v_X'] = e_v_X
    df['e_v_Y'] = e_v_Y
    df['e_v_Z'] = e_v_Z


    df['Lx'] = Y * v_Z - Z * v_Y
    df['Ly'] = Z * v_X - X * v_Z
    
    df['Lz'] = X * v_Y - Y * v_X

    # Cylinderical coordinate
    df['R'] = np.sqrt( X**2 + Y**2 )
    df['Lz_cyl'] = -1. * df['Lz']
    
    return(df)



def elem2Znum(elems):

    # "elems" can contain both upper/lower case letters
    
    # Define a dictionary
    elemdict = { "c": 6, \
                 "n": 7, \
                 "o": 8, \
                 "na": 11, \
                 "mg": 12, \
                 "al": 13, \
                 "si": 14, \
                 "k": 19, \
                 "ca": 20, \
                 "sc": 21, \
                 "ti": 22, \
                 "v": 23, \
                 "cr": 24, \
                 "mn": 25, \
                 "fe": 26, \
                 "co": 27, \
                 "ni": 28, \
                 "cu": 29, \
                 "zn": 30, \
                 "y": 39, \
                 "ba": 56, \
                 "la": 57, \
                 "eu": 63 }


    # Format "elem" as a string containing only lower case letters
    Znum = np.zeros(np.size(elems))
    
    for i, elem in enumerate(elems):
        elem = elem.lower()

        Znum[i] = elemdict[elem]
        
    
    return(Znum)



def get_zfrac(feh):

    print(feh)

    
    # Get the solar value of Z_Fe (mass fraction of Fe) 
    
    logaFe_sun=-4.5 # Asplund+09
    
    # see http://stev.oapd.inaf.it/cgi-bin/cmd
    Z_sun=0.0152 
    Y_sun=0.2485+1.78*Z_sun
    
    zfe=(1.0-Y_sun-Z_sun)*56*10.**logaFe_sun
    
    
    # The solar value for Z_Fe/Z (mass fraction of Fe relative to the total metal )
    zfe_Z=zfe/Z_sun
    
    
    # Derive Z assuming that the star's Z_Fe/Z is the same as that of the Sun

    print(feh, logaFe_sun)    
    expfac=56*10**(feh+logaFe_sun)
    
    
    # with X=(1-Y-Z), where Y=0.2485+1.78Z, solve for Z gives following expression
    
    Z=(expfac*0.7515)/(zfe_Z+expfac*2.78)
    
    
    return(Z)



def read_GALAH_xfe(starname, elems):
#        if flag0[i]==0:
#            z=np.hstack((z,zz))
#            xfe=np.hstack((xfe,xfeobs0[i]))
#            xfeerr=np.hstack((xfeerr,xfeerr0[i]))
#        else:
#            continue
 
 
    #filt= (flag0!=99) 
    #z=znumobs0[filt]
    #xfe=xfeobs0[filt]
    #xfeerr=xfeerr0[filt]
  
 
    # Get Observed [Fe/H]
    
    df = pd.read_csv("../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat")
    filt = df["sobject_id_1"] == np.int(starname) 
    feh = np.float64(df["fe_h"][filt])
    abuclass = df["abuclass"][filt]



    
    elems_noflag = ()
    xfe = ()
    xfeerr = ()
    
    for elem in elems:

        
        if (df["flag_" + elem + "_fe"][filt]).values != 0:
            continue
            
        elems_noflag = np.append(elems_noflag, elem)
        xfe = np.append(xfe, df[elem + "_fe"][filt])
        xfeerr = np.append(xfeerr, df["e_" + elem + "_fe"][filt])



        
    z = elem2Znum(elems_noflag)

    
    
    return(z, xfe, xfeerr, feh, abuclass)


def read_APOGEE_xfe(starname, elems):

    df0 = pd.read_csv("../input_APOGEE/APOGEE_halo-sample.csv")

    filt = df0["apogee_id"] == starname
    df = df0[filt]


    mh = np.float64(df["m_h"])



    elems_noflag = ()
    xfe = ()
    xfeerr = ()

    for elem in elems:


        if (df[elem + "_fe_flag"]).values != 0:
            continue

        elems_noflag = np.append(elems_noflag, elem)
        xfe = np.append(xfe, df[elem + "_fe"])
        xfeerr = np.append(xfeerr, df[elem + "_fe_err"])


    z = elem2Znum(elems_noflag)


    return(z, xfe, xfeerr, mh)

                                       


def abuclass_func(x):

    
    x1 = -1.3
    y1 = 0.25
    x2 = -0.8
    y2 = 0.05

    
    y = (y2-y1)*(x - x1)/(x2-x1) + y1

    return(y)

def get_abuclass(fehs, mgfes):


    ys = abuclass_func(fehs)

    abuclasses = ()
    
    for i, feh in enumerate(fehs):

        if feh < -1.5:
            abuclass = "metal-poor"
        elif mgfes[i] < abuclass_func(feh):
            abuclass = "low-alpha"
        elif mgfes[i] >= abuclass_func(feh):
            abuclass = "high-alpha"

        abuclasses = np.append(abuclasses, abuclass)
        
    return(abuclasses)




def get_vtot(v_X, v_Y, v_Z, e_v_X, e_v_Y, e_v_Z):



    # Calculate velocity with respect to the Galactic center. See 
    # https://iopscience.iop.org/article/10.3847/2515-5172/aaef8b/pdf
    # for the definition
    state = galactocentric_frame_defaults.get_from_registry("v4.0")
    v_sun = [(state["parameters"]["galcen_v_sun"]).d_x, \
             (state["parameters"]["galcen_v_sun"]).d_y, \
             (state["parameters"]["galcen_v_sun"]).d_z ]

    e_v_sun = (3.0, 1.4, 0.09) *u.km/u.s
             
    vtot = np.sqrt((v_X-v_sun[0])**2 + (v_Y-v_sun[1])**2 + (v_Z-v_sun[2])**2)

    e_vtot = np.sqrt((1./vtot**2) * (
        (v_X - v_sun[0])**2 * (e_v_X**2 + e_v_sun[0]**2) +
        (v_Y - v_sun[1])**2 * (e_v_Y**2 + e_v_sun[1]**2) +
        (v_Z - v_sun[2])**2 * (e_v_Z**2 + e_v_sun[2]**2)
        ))
    
    return(vtot, e_vtot)




def cmatch_Gaia(table_name):

    # Full qualified table 
    full_qualified_table_name = 'user_mishigak.' + table_name


    query = ("select * from user_mishigak.galahdr3_flagsp0_snr_msto as gl "
             "inner join gaiaedr3.dr2_neighbourhood as dr "
             "on gl.source_id_gdr2 = dr.dr2_source_id "
             "inner join gaiaedr3.gaia_source as ga "
             "on dr.dr3_source_id = ga.source_id ")



    j = Gaia.launch_job_async(query = query, output_file = \
                              "GALAH_DR3_Main_x_Ages_flagsp0_SNR30_MSTO_GaiaEDR3.csv", \
                              dump_to_file = True, \
                              output_format = 'csv')

    r = j.get_results()

    return(r)


def plot_age_error(df0, df_age, df_ohs):

    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(1, 1)


    cols = [cmap(7), cmap(9), cmap(3)]
    lss = [":", "--", "-"]
    labs = ['All selected', '+ Age>12 Gyrs', '+ Halo kinematics']
    histt = ["stepfilled", "step", "step"] 
    
    bins = np.arange(0., 5.0, 0.4)
    
    for k in range(0, 3):

        if k == 0:
            df = df0
        elif k == 1:
            df = df_age
        elif k == 2:
            df = df_ohs
    
        #for i in (df.index):
        #    print(df['e_age_bstep'][i], \
        #          ((df['age_bstep'][i] - df['e16_age_bstep'][i]) + \
        #           (-df['age_bstep'][i] + df['e84_age_bstep'][i]))/2.)

    
        #filt = df['fe_h']>-1.0
        #metalrich = df['e_age_bstep'][filt]
        #filt = df['fe_h']<=-1.0
        #metalpoor = df['e_age_bstep'][filt]

    
        e_age = (df['e_age_bstep']).dropna()

        print("Median age error:", np.median(e_age))

    
        ax.hist(e_age, bins = bins, histtype = histt[k], density = True, \
                color = cols[k], \
                lw = 3, ls = lss[k], alpha = 0.7, label = labs[k])

        
        #ax.hist(metalpoor, bins = 4, histtype = 'step', color = cmap(9), \
        #        lw = 3, ls = "--", alpha = 0.8, label = r"[Fe/H]$\leq -1$")
    ax.legend(loc = 1)
    ax.set_xlabel("Age uncertainty [Gyrs]")
    ax.set_ylabel("Normalized histogram")
    plt.tight_layout()
    plt.savefig("../figs/age_uncertainty.pdf")
    
    plt.show()
    
    return()


    
def plot_distribution(df, df_age, df_ohs, axislabelcolor = False):
    
    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")
    alc = "cornsilk"  # FFF8DC in colorcode


    if axislabelcolor==True:
        matplotlib.rc('axes', edgecolor = alc)
    
    fig, ax = plt.subplots(1, 3, figsize = (19, 6))

    mks = ['.', 'x', 'o']

    ecols = [cmap(7), cmap(9), cmap(3)]
    fcols = [cmap(7), cmap(9), 'none']
    mss = [2, 5, 9]
    labs = ['All selected', '+ Age>12 Gyrs', '+ Halo kinematics']
    for i in range(0, 3):

        if i == 0:
            df0 = df
        elif i == 1:
            df0 = df_age
        elif i == 2:
            df0 = df_ohs


        df = df0.dropna(subset = ['R', 'Z', 'v_X', 'v_Y', 'v_Z'])

        if i == 0:
            
            for k in range(0, 2):

                if k == 0:
                    xhist = df["R"]/1000.
                    yhist = df["Z"]/1000.

                elif k == 1:
                    xhist = df['v_Y']
                    yhist = np.sqrt(df['v_X']**2 + df['v_Z']**2)

                    
                counts, xedges, yedges, Image = \
                    ax[k].hist2d(xhist, yhist, \
                                 bins = 40, cmap = 'Greys')
                levels = np.linspace(10.0, np.max(counts), 5)
                ax[k].contour(counts.transpose(), \
                              extent = [xedges.min(), xedges.max(), \
                                        yedges.min(), yedges.max()], \
                              linewidths = 2,levels = levels, colors='#606060', alpha = 0.8)
                x_LC = ()
                y_LC = ()
                for ii, xe in enumerate(xedges[:-1]):
                    for jj, ye in enumerate(yedges[:-1]):
                        if counts[ii,jj] < 5:

                            filt = (xhist >= xe) & (xhist < xedges[ii + 1]) & \
                                (yhist >= ye) & (yhist < yedges[jj + 1])
                            x_LC = np.hstack((x_LC, xhist[filt]))
                            y_LC = np.hstack((y_LC, yhist[filt]))


                ax[k].plot(x_LC, y_LC, marker = '.', linestyle = '', ms = mss[i], \
                           mfc = cmap(7), mec = cmap(7), label = labs[i])

                    
                
                #ax[k].hist(df['fe_h'], bins = 20, alpha = 0.5)


        else:

    
            ax[0].plot(df['R']/1000., df['Z']/1000., linestyle = '', marker = mks[i], \
                       mec = ecols[i], mfc = fcols[i], ms = mss[i], alpha = 0.8, label = labs[i])
            ax[1].plot(df['v_Y'], np.sqrt(df['v_X']**2 + df['v_Z']**2), linestyle = '', \
                       marker = mks[i], mec = ecols[i], mfc = fcols[i], ms = mss[i], alpha = 0.8)
            if i == 2:
                ax[2].hist(df['fe_h'], bins = 10, alpha = 0.8, \
                           linewidth = 2, color = ecols[i], histtype = 'step', label = labs[i])


                    

    ax[0].set_xlabel("R [kpc]")
    ax[0].set_ylabel("Z [kpc]")
    ax[0].set_xlim(5.5, 10.5)
    ax[0].set_ylim(-1.9, 1.9)
    ax[0].legend(loc = 2)
    ax[1].set_xlabel("$V_{Y}$ [km/s]")
    ax[1].set_ylabel("$\sqrt{V_{X}^2+V_{Z}^2}$ [km/s]")
    ax[1].set_xlim(-170., 360.)
    ax[1].set_ylim(0.0, 400.)
    ax[2].set_xlabel("[Fe/H]")
    ax[2].set_ylabel("N")

    plt.subplots_adjust(wspace = 0.25)
    plt.savefig("../figs/RZ_V_FeH.pdf")
    plt.savefig("../figs/RZ_V_FeH.png")


    if axislabelcolor == True:


        for i in range(0, 3):
            ax[i].tick_params(axis = 'both', which = 'both', reset = True, color = alc, labelcolor = alc)
            ax[i].yaxis.label.set_color(alc)
            ax[i].xaxis.label.set_color(alc)
            
        #ax[0].set_xlabel("R [kpc]", color = alc)
        #ax[0].set_ylabel("Z [kpc]", color = alc)
        #ax[1].set_xlabel("$V_{Y}$ [km/s]", color = alc)
        #ax[1].set_ylabel("$\sqrt{V_{X}^2+V_{Z}^2}$ [km/s]", color = alc)
        #ax[2].set_xlabel("[Fe/H]", color = alc)
        #ax[2].set_ylabel("N", color = alc)
            
        plt.savefig("../figs/RZ_V_FeH_alc.png")
    plt.show()


    return()








#df = pd.read_csv("GALAH_DR3_Main_x_Ages_flagsp0_SNR40_MSTO_GaiaEDR3.csv")

#select_kinematics(df)
