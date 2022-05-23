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




sys.path.insert(0, os.path.abspath('../'))
import oldhalostars.abudata as ad
import oldhalostars.abufit as af


def plot_bestfitmodel_CC(starname, abuclass, z, xfe, xfeerr, feh, \
                         theta0, chi2, dof, modelid, bottom = [0.0, 0.0, 0.0], \
                         top = [0.0, 0.0, 0.0]):
    
    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")  
    
    figids=["alpha","Zcc"]
    diffs=np.array([2.0,0.003])
    cols=[cmap(1),cmap(2)]
    lines=["--",":"]
    
    zmax = ad.get_zfrac(feh)
    print("zmax=",zmax)
    alpha_max = top[0]
    alpha_min = bottom[0]
    zcc_max = top[1]
    zcc_min = bottom[1]
    
    #figname=starname+"_bestfit.eps"
    
    #string=starname.split('/')
    
   

    print(theta0)
    
    fig,ax = plt.subplots(1,2,sharey=True,figsize=(13,4))

    outpath = "../figs/abupatterns_Ia/" + modelid + "/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    
    figname = outpath + starname.strip(" ") + \
        "_bestfit_alpha_Zcc.pdf"
    
    
    for k in range(0,np.size(theta0)):
        
    
        if k==0:
            lab=r"$\alpha_{\rm IMF}=%.2f$"%(theta0[0])
            lab_p=r" $%.2f$"%(alpha_max)
            lab_n=r" $%.2f$"%(alpha_min)
        else:
            zsun=0.0152
            lab=r"$Z_{\rm CC}/Z_\odot=%.2f$"%(theta0[1]/zsun)
            lab_p=r" $%.2f$"%(zcc_max/zsun)
            lab_n=r" $%.2f$"%(zcc_min/zsun)
            
            
        zmodel,xfemodel,mass_fe=calc_yields_CC(theta0)   
            
        # Elements that have not been taken into account
        melemlows=[21,22]
        for melemlow in melemlows:
            mlow=xfemodel[zmodel==melemlow]
            xx1=melemlow-0.1
            xx2=melemlow+0.1
            #yy1=mlow
            yy1 = -5.0
            yy2=5.0
            ax[k].fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')
        
   
        
        ax[k].errorbar(z,xfe,yerr=xfeerr,linestyle="",marker="o",color=cmap(0),ms=10)
        ax[k].plot(zmodel,xfemodel,linestyle="-",marker="",color='k',linewidth=2,label=lab)
    
        
        #figname=starname.strip(" ")+"_bestfit_"+figids[k]+".eps"
        theta_p=theta0.copy()
        if k==0:
            theta_p[k]=alpha_max
            
        else:
            theta_p[k]=zcc_max
            
            
        zmodel,xfemodel,mass_fe=calc_yields_CC(theta_p)
        ax[k].plot(zmodel,xfemodel,linestyle="--",marker="",color=cols[k],linewidth=2,label=lab_p)
        print(theta0)
        theta_n=theta0.copy()
        if k==0:
            theta_n[k]=alpha_min
        else:
            theta_n[k]=zcc_min
        
 
    
        zmodel,xfemodel,mass_fe=calc_yields_CC(theta_n)
        ax[k].plot(zmodel,xfemodel,linestyle=":",marker="",color=cols[k],linewidth=2,label=lab_n)
        print(theta0)
    
        #theta_n=theta0
        #theta_n[1]=theta0[1]-0.0001
    
    
        ax[k].set_xlim(5.0,31.)
        ax[k].set_ylim(-0.9,1.2)
        
     
        ax[k].set_xlabel("Z")
        ax[k].legend(labelspacing=0.25,prop={"size":13}, loc = 1)
        
    #ax[0].text(5.3,0.9,"%s\n[Fe/H]$=%.2f$, $\chi^2/DoF=$%.1f/%i"%(string[1],feh,chi2,dof))
    ax[0].text(5.3,-0.8,r"%s, %s, [Fe/H]=$%.1f$"%(starname, abuclass, feh))
    
    ax[0].set_ylabel("[X/Fe]")
    plt.subplots_adjust(wspace = 0.01)
    plt.savefig(figname)
        
    plt.show()
        
    return
    
    

    
def plot_bestfitmodel_CC_Ia(starname, outpath, abuclass, z,xfe,xfeerr,feh,theta0, \
                            zIa,f_Ch,chi2,dof, modelid, \
                            bottom = [0.0, 0.0, 0.0], \
                            top = [0.0, 0.0, 0.0]):
    
    plt.rcParams["font.size"] = 14
    
    cmap = plt.get_cmap("tab10")  
    
    figids=["alpha","Zcc","fIa"]
    diffs=np.array([2.0,0.003,0.05])
    
    #figname=starname+"_bestfit.eps"
    
    #string=starname.split('/')
    

    
    cols=[cmap(1),cmap(2),cmap(3)]
    lines=["--",":","-:"]
    
    zmax = ad.get_zfrac(feh)
    print("zmax=",zmax)
    alpha_max=top[0]
    alpha_min=bottom[0]
    zcc_max = top[1]
    zcc_min = bottom[1]
    f_Ia_max = top[2]
    f_Ia_min = bottom[2]
    


    
    #figname=starname+"_bestfit.eps"
    


    
    fig,ax=plt.subplots(1,3,sharey=True,figsize=(18,4))

    #outpath = "../figs/abupatterns_Ia/" + modelid + "/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    figname= outpath + starname.strip(" ") + \
        "_bestfit_alpha_Zcc_fIa.pdf"
    
    #fig_b,bx=plt.subplots(1,1,sharey=True,figsize=(5.5,4))
    #figname_b=starname.strip(" ")+"_bestfit_fIa.eps"
    
    metalmassname=starname.strip(" ")+"_metalmass.txt"

    print(theta0)
    for k in range(0,np.size(theta0)):

        if k==0:
            lab=r"$\alpha_{\rm IMF}=%.2f$"%(theta0[0])
            lab_p=r" $%.2f$"%(alpha_max)
            lab_n=r" $%.2f$"%(alpha_min)
        elif k==1:
            zsun=0.0152
            lab=r"$Z_{\rm CC}/Z_\odot=%.2f$"%(theta0[1]/zsun)
            lab_p=r" $%.2f$"%(zcc_max/zsun)
            lab_n=r" $%.2f$"%(zcc_min/zsun)
          
        else:
            lab=r"$f_{\rm Ia}=%.2f$"%(theta0[2])
            lab_p=r" $%.2f$"%(f_Ia_max)
            lab_n=r" $%.2f$"%(f_Ia_min)
        

        # Write metal mass to a file
        zmodel,xfemodel,mass_fe,mass_Type2,mass_TypeIa=af.calc_yields_CC_Ia(theta0,zIa,f_Ch)

        data = {"Z": zmodel, "XFe": xfemodel, "Mass_CCSNe": mass_Type2, \
                "Mass_Ia": mass_TypeIa, \
                "Mfrac_Ia": mass_TypeIa/(mass_Type2 + mass_TypeIa), \
                "Mass_Fe": np.full(len(zmodel), mass_fe)}
        df = pd.DataFrame(data)
         
        #a=np.array([zmodel,xfemodel, mass_fe, mass_Type2,mass_TypeIa])
        #aa=a.T
        #np.savetxt(dir + "/" + metalmassname,aa)    


        #outpath_metal = "../outputs/metalmass/" + modelid + "/"



        #if not os.path.exists(outpath_metal):
        #    os.makedirs(outpath_metal)
            
        df.to_csv(outpath \
                  + metalmassname, index = False)
        
            
        # Elements that have not been taken into account
        melemlows=[21,22]
        for melemlow in melemlows:
            mlow=xfemodel[zmodel==melemlow]
            xx1=melemlow-0.1
            xx2=melemlow+0.1
            #yy1=mlow
            yy1 = -5.
            yy2=5.0
            #if k!=2:
            ax[k].fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')
            #else:
            #bx.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')
   
        #if k!=2:
        ax[k].errorbar(z,xfe,yerr=xfeerr,linestyle="",marker="o",color=cmap(0),ms=10)
        ax[k].plot(zmodel,xfemodel,linestyle="-",marker="",color='k',linewidth=2,label=lab)
    
        #else:
        #    bx.errorbar(z,xfe,yerr=xfeerr,linestyle="",marker="o",color=cmap(0),ms=10)
        #    bx.plot(zmodel,xfemodel,linestyle="-",marker="",color='k',linewidth=2,label="Bestfit")
        
     
        theta_p=theta0.copy()
        if k==0:
            theta_p[k] = alpha_max       
        
        elif k==1:
            theta_p[k] = zcc_max #zmax
        else:
            theta_p[k] = f_Ia_max  #theta0[k]+diffs[k]  
        
            
        zmodel,xfemodel,mass_fe,mass_Type2,mass_TypeIa=af.calc_yields_CC_Ia(theta_p,zIa,f_Ch)
        #if k!=2:
        ax[k].plot(zmodel,xfemodel,linestyle="--",marker="",color=cols[k],linewidth=2,label=lab_p)
        #else:
        #    bx.plot(zmodel,xfemodel,linestyle="--",marker="",color=cols[k],linewidth=2,label=lab_p)
        print(theta0)
        
        theta_n=theta0.copy()
        if k==0:
            theta_n[k] = alpha_min
        elif k==1:
            theta_n[k] = zcc_min # 0.0000000001
        
        else:
            theta_n[k] = f_Ia_min #theta0[k]-diffs[k]
    
        zmodel,xfemodel,mass_fe,mass_Type2,mass_TypeIa = af.calc_yields_CC_Ia(theta_n,zIa,f_Ch)
        #if k!=2:
        ax[k].plot(zmodel,xfemodel,linestyle=":",marker="",color=cols[k],linewidth=2,label=lab_n)
        #else:
        #    bx.plot(zmodel,xfemodel,linestyle=":",marker="",color=cols[k],linewidth=2,label=lab_n)
        print(theta0)
        
        
        #if k!=2:
        ax[k].set_xlim(5.0,31.)
        ax[k].set_ylim(-0.9,1.2)
        
     
        ax[k].set_xlabel("Z")
        ax[k].legend(labelspacing=0.3, prop={"size":12}, loc = 1)
        #else:
        #    bx.set_xlim(5.0,31.)
        #    bx.set_ylim(-0.9,1.2)
        
     
        #    bx.set_xlabel("Z")
        #    bx.legend()
        
    #ax[0].text(5.3,0.9,"%s\n[Fe/H]$=%.2f$, $\chi^2/DoF=$%.1f/%i"%(string[1],feh,chi2,dof))
    
    ax[0].text(5.3,-0.8,r"%s, %s, [Fe/H]=$%.1f$"%(starname, abuclass, feh))
    
    
    ax[0].set_ylabel("[X/Fe]")
    #bx.text(5.3,0.9,"%s\n[Fe/H]$=%.2f$, $\chi^2/DoF=$%.1f/%i"%(string[1],feh,chi2,dof))
    #bx.set_ylabel("[X/Fe]")
    #fig.tight_layout(w_pad=0.01)
    fig.subplots_adjust(wspace = 0.01)
    #fig_b.tight_layout(w_pad=0.02)
    fig.savefig(figname)
    #fig_b.savefig(figname_b) 
        
    return
    


def plot_metalmassfrac(modelid):

    # Plot settinigs
    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")


    # Get stellar data
    catalog = "../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat"
    dfcat = pd.read_csv(catalog)


    cls = ["high-alpha", "low-alpha", "metal-poor"]
    cllabs = [r"High-$\alpha$", r"Low-$\alpha$", "Metal-poor"]


    fig, ax = plt.subplots(3, 1, figsize = (12, 15))
    
    for k, cl in enumerate(cls):

        filt = dfcat['abuclass'] == cl
        df = dfcat[filt]

        
        mfrac_Ias = np.empty((len(df),26))


        for i, starname in enumerate(df['sobject_id_1']):
        
            metalmassname = "../outputs/metalmass/" + modelid + "/" + \
                str(starname) + "_metalmass.txt"

            dfmetal = pd.read_csv(metalmassname)
            mfrac_Ias[i, :]= dfmetal["Mfrac_Ia"]
            if i == 0:
                zelem0 = dfmetal["Z"]

        print(zelem0)
        print(mfrac_Ias)
        
        mfrac_Ia_mean0 = np.mean(mfrac_Ias, axis = 0)
        mfrac_Ia_std0 = np.std(mfrac_Ias, axis = 0)

        

        filt = (zelem0 != 6.5) 
        zelem = zelem0[filt]
        mfrac_Ia_mean = mfrac_Ia_mean0[filt]
        
        barindx = np.arange(0, len(zelem), 1)
        labels=["C","N", "O","F", "Ne", "Na","Mg","Al","Si","P", \
                "S", "Cl", "Ar", "K", "Ca","Sc",\
                "Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
        ax[k].bar(barindx, mfrac_Ia_mean, tick_label = labels, \
                  color = cmap(k), align = "center", label = cllabs[k], \
                  alpha = 0.8)
        
        ax[k].legend(loc = 2, prop = {"size": 20})


    plt.savefig("../figs/IaMetal_mass_frac.png")
    plt.show()
    
    return








def show_fitting_results(dir):


    # Plot settinigs
    plt.rcParams["font.size"] = 11
    cmap = plt.get_cmap("tab10")


    
    cls = ["high-alpha-MP", "high-alpha-MR", "low-alpha"]
    
    if np.int(((dir.split("Ia"))[1])[0]) == 1:
        TypeIa = True
        params = ["alpha", "zcc", "f_Ia", "rchi2"]
    else:
        TypeIa = False

        params = ["alpha", "zcc", "rchi2"]
        
    filelist = glob.glob(dir + "/*_bestfitparams.txt")

    alpha = ()
    lzcc = ()
    z_sun = 0.0152
    f_Ia = ()
    abuclass = ()
    rchi2 = ()
    
    for file in filelist:
        
        f = open(file)
        line = f.readline()
        f.close()
        data = line.split(",")
        
        
        alpha = np.append(alpha, np.float64(data[2]))
        lzcc = np.append(lzcc, np.log10(np.float64(data[3])/z_sun))
        if TypeIa == True:
            f_Ia = np.append(f_Ia, np.float64(data[4]))
            rchi2 = np.append(rchi2, np.float64(data[7])/(np.float64(data[8]) - 3.))
            
            abuclass = np.append(abuclass, data[9].strip())
        else:
            rchi2 = np.append(rchi2, np.float64(data[4])/(np.float64(data[5]) - 2.))
            abuclass = np.append(abuclass, data[6].strip())

    if TypeIa==True:
        paramdata = [alpha, lzcc, f_Ia, rchi2]
        bins = [np.arange(-1., 3., 0.2), np.arange(-5.0, 0.0, 0.5), \
                np.arange(0.0, 1.0, 0.05), np.arange(0.0, 100., 5.)]
    else:
        paramdata = [alpha, lzcc, rchi2]

        bins = [np.arange(-1., 3., 0.2), np.arange(-5.0, 0.0, 0.5), np.arange(0.0, 200., 5.)]

        
    fig, ax = plt.subplots(1, len(params), figsize = (18, 5))


    
    for k, vals in enumerate(paramdata):

        for j, cl in enumerate(cls):

            filt = abuclass == cl
        
            mean = np.mean(vals[filt])
            median = np.median(vals[filt])
            std = np.std(vals[filt])

            print(mean, median, std)

            ax[k].hist(vals[filt], bins = bins[k], label = cl, \
                       color = cmap(j), \
                       density = True, alpha = 0.5)

            ax[k].legend()
                      #                    bins = bins)
            #\
            #                              density = True)
            #ax[k].plot([median, median], [0.0, 10])

            #ax[k].set_ylim(0.0, np.max(vals[filt])*0.05)

            
            #ax[k].errorbar([j], median, yerr = std, \
            #               color = cmap(j), marker = 'o')

            #ax[k].errorbar([j], mean, yerr = std, \
            #               color = cmap(j), marker = 's')
            
        #ax[k].set_xticks([0, 1, 2])
        #ax[k].set_xticklabels([cls[0], cls[1], cls[2]], rotation=20)
        #ax[k].set_xlim(-1, 3)
    
    return()


def plot_abupatterns_MCMC(dir):

    modelid = (dir.split("results_"))[1]

    # Get stellar data
    catalog = "../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat"
    dfcat = pd.read_csv(catalog)


    if np.int(((dir.split("Ia"))[1])[0]) == 1:
        TypeIa = True

        f_Ch = np.float(((dir.split("fCh"))[1])[0:4])
        
        #params = [r"$\alpha$", r"$\log(Z_{\rm CC}/Z_{\odot})$", r"$f_{\rm Ia}$"]
    else:
        TypeIa = False

        #params = [r"$\alpha$", r"$\log(Z_{\rm CC}/Z_{\odot})$"]
        

    

    filelist = glob.glob(dir + "/*_trace.nc")

    for file in filelist:
        
        starname = ((file.split('/'))[-1].split('_'))[0]


        # Get observed abundances
        elems = ["c", "o", "na", "mg", "al", "si", "ca", "v", "cr", "mn", "co", "ni", "cu", "zn"]
        z, xfe, xfeerr, feh, abuclass = ad.read_GALAH_xfe(starname, elems)

        if len(abuclass)==0:
            continue
        
        zmax = ad.get_zfrac(feh)
        zsun=0.0152
        if zmax < 0.1*zsun:
            zIa = 0.0
        else:
            zIa = 0.1*zsun
    

        
        # Get best-fit parameters
        mean_params, bottom, top = read_netcdf(file, TypeIa)


        if (abuclass.values)[0] == "high-alpha":
            abuclasslab = r"High-$\alpha$"
        elif (abuclass.values)[0] == "low-alpha":
            abuclasslab = r"Low-$\alpha$"
        else:
            abuclasslab = r"Metal-poor"
        
        if TypeIa == True:
            chi2 = np.nan
            dof = np.nan
            plot_bestfitmodel_CC_Ia(starname,abuclasslab, z,xfe, \
                                    xfeerr,feh,mean_params, \
                                    zIa,f_Ch,chi2,dof, modelid, \
                                    bottom, top)
    

        else:
            chi2 = np.nan
            dof = np.nan
            plot_bestfitmodel_CC(starname,abuclasslab, z,xfe, \
                                    xfeerr,feh,mean_params, \
                                    chi2,dof, modelid, bottom, top)
    
            

    return




def show_MCMC_results(dir):

    
    # Plot settinigs
    plt.rcParams["font.size"] = 14
    cmap = plt.get_cmap("tab10")


    # Get stellar data
    catalog = "../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat"
    dfcat = pd.read_csv(catalog)

    
    cls = ["high-alpha", "low-alpha", "metal-poor"]
    cllabs = [r"High-$\alpha$", r"Low-$\alpha$", "Metal-poor"]
    ymaxs = [55., 17., 5.]
    
    if np.int(((dir.split("Ia"))[1])[0]) == 1:
        TypeIa = True
        params = [r"$\alpha_{\rm IMF}$", \
                  r"$\log(Z_{\rm CC}/Z_{\odot})$", r"$f_{\rm Ia}$"]
    else:
        TypeIa = False

        params = [r"$\alpha_{\rm IMF}$", r"$\log(Z_{\rm CC}/Z_{\odot})$"]
        
    filelist = glob.glob(dir + "/*_trace.nc")

    starnames = ()
    fehs = ()
    alpha = ()
    alpha_min = ()
    alpha_max = ()
    lzcc = ()
    zcc = ()
    zcc_min = ()
    zcc_max = ()
    z_sun = 0.0152
    f_Ia = ()
    f_Ia_min = ()
    f_Ia_max = ()
    abuclasses = ()
    
    
    for file in filelist:
        

        starname = ((file.split('/'))[-1].split('_'))[0]

        
        filt = dfcat['sobject_id_1'] == np.int64(starname)
        if len(dfcat[filt])==0:
            print("No data for ", starname)
            continue

        starnames = np.append(starnames, starname)

        
        feh = dfcat['fe_h'][filt].values[0]
        fehs = np.append(fehs, feh)
        
        abuclass = dfcat['abuclass'][filt].values[0]
        abuclasses = np.append(abuclasses, abuclass)
        
        mean_params, bottom, top = read_netcdf(file, TypeIa)

        
        
        #df = pd.read_csv(file, \
        #                 names = ("paramname", "mean", "sd", "hdi3", "hdi97"), \
        #                 header = 0)
        
        alpha = np.append(alpha, mean_params[0])
        alpha_min = np.append(alpha_min, bottom[0])
        alpha_max = np.append(alpha_max, top[0])
        
        zcc = np.append(zcc, mean_params[1])
        zcc_min = np.append(zcc_min, bottom[1])
        zcc_max = np.append(zcc_max, top[1])
        
        lzcc = np.append(lzcc, np.log10(mean_params[1]/z_sun))
        
        if TypeIa == True:
            
            f_Ia = np.append(f_Ia, mean_params[2])
            f_Ia_min = np.append(f_Ia_min, bottom[2])
            f_Ia_max = np.append(f_Ia_max, top[2])

    print("Number of data: ", len(alpha))


    if TypeIa==True:
        
        paramdata = [alpha, lzcc, f_Ia]
        bins = [np.arange(-1., 3.2, 0.2), np.arange(-4., 0.0, 0.2), \
                np.arange(0.0, 0.52, 0.02)]

        tabledata = {"Starname":starnames, "[Fe/H]":fehs, \
                     "Group": abuclasses, "alpha_IMF": alpha, \
                     "alpha_IMF_min": alpha_min, "alpha_IMF_max": alpha_max, \
                     "Z_CC": zcc, "Z_CC_min": zcc_min, "Z_CC_max": zcc_max, \
                     "f_Ia": f_Ia, "f_Ia_min": f_Ia_min, "f_Ia_max": f_Ia_max}

        columns=['Starname', '[Fe/H]', 'Group', 'alpha_IMF', 'alpha_IMF_min', \
                 'alpha_IMF_max', 'Z_CC', 'Z_CC_min', 'Z_CC_max', \
                 'f_Ia', 'f_Ia_min', 'f_Ia_max']
        head = ["GALAH DR3 ID", "[Fe/H]", "OHS subgroup", \
                "$\\alpha_{\\rm IMF}$", "$\\alpha_{\\rm IMF, min}$", "$\\alpha_{\\rm IMF, max}$", \
                "$Z_{\\rm CC}$", "$Z_{\\rm CC, min}$", "$Z_{\\rm CC, max}$",
                "$f_{\\rm Ia}$", "$f_{\\rm Ia, min}$", "$f_{\\rm Ia, max}$"]
        tablename = "../tabs/Table_A4.tex"
        csvtablename = "../tabs/Table_A4.csv"
        formatters = [fmtis,fmt2,fmts,fmt3,fmt3,fmt3,fmt5,fmt5,fmt5,\
                     fmt2,fmt2,fmt2]

        for j, cl in enumerate(cls):
            min_fIa = np.min(f_Ia[abuclasses == cl])
            max_fIa = np.max(f_Ia[abuclasses == cl])
            mean_fIa = np.mean(f_Ia[abuclasses == cl])
            print("f_Ia range for %s: min = %.2f, max = %.2f, mean = %.2f\n"%\
                  (cl, min_fIa, max_fIa, mean_fIa))
            #print("Zcc and f_Ia range: %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n"\
            #      %(np.min(zcc[fehs>-1.5]/z_sun), \
            #  np.max(zcc[fehs>-1.5]/z_sun), np.median(zcc[fehs>-1.5]/z_sun), \
            #  np.mean(zcc[fehs>-1.5]/z_sun), \
            #  np.min(f_Ia[fehs>-1.5]), np.max(f_Ia[fehs>-1.5]), \
            #                          np.median(f_Ia[fehs>-1.5]), np.mean(f_Ia[fehs>-1.5])))

        
    else:
        paramdata = [alpha, lzcc]

        bins = [np.arange(-1., 3.2, 0.2), np.arange(-4, 0.0, 0.2)]


        tabledata = {"Starname":starnames, "[Fe/H]":fehs, \
                     "Group": abuclasses, "alpha_IMF": alpha, \
                     "alpha_IMF_min": alpha_min, "alpha_IMF_max": alpha_max, \
                     "Z_CC": zcc, "Z_CC_min": zcc_min, "Z_CC_max": zcc_max}
 


        columns=['Starname', '[Fe/H]', 'Group', 'alpha_IMF', 'alpha_IMF_min', \
                 'alpha_IMF_max', 'Z_CC', 'Z_CC_min', 'Z_CC_max']
        head = ["GALAH DR3 ID", "[Fe/H]", "OHS subgroup", \
                "$\\alpha_{\\rm IMF}$", "$\\alpha_{\\rm IMF, min}$", "$\\alpha_{\\rm IMF, max}$", \
                "$Z_{\\rm CC}$", "$Z_{\\rm CC, min}$", "$Z_{\\rm CC, max}$"]
        

        tablename = "../tabs/Table_A3.tex"
        csvtablename = "../tabs/Table_A3.csv"
        formatters = [fmtis,fmt2,fmts,fmt3,fmt3,fmt3,fmt5,fmt5,fmt5]
        
    df_latex = pd.DataFrame(tabledata)
    df_latex.to_csv(csvtablename, index = False)

    df_latex[:5].to_latex(tablename, columns = columns, 
                          header = head, \
                      escape=False, index=False,\
                      formatters=formatters)
    

    
    for k, vals in enumerate(paramdata):

        fig, ax = plt.subplots(3, 1, figsize = (6, 7), sharex = True)

        
        for j, cl in enumerate(cls):

            filt = abuclasses == cl

            print("Number of stars with class = ", cl, "%i"%(len(vals[filt])))
            print(vals[filt])
          
            
            
            mean = np.mean(vals[filt])
            median = np.median(vals[filt])
            std = np.std(vals[filt])

            print(mean, median, std)

            ax[j].hist(vals[filt], bins = bins[k], label = cllabs[j], \
                       color = cmap(j), \
                       density = False, alpha = 0.5)

            if k == 1:

                fehmean = np.mean(fehs[filt])
                zcc_obs = np.log10(ad.get_zfrac(fehmean)/z_sun)
                ax[j].plot([zcc_obs, zcc_obs], [0., 100.], marker = '', \
                           ls = ":", lw = 3, label = "Mean obs.", \
                           color = cmap(7), alpha = 0.7)
                ax[j].set_ylim(0.0, ymaxs[j])


            
            ax[j].set_ylabel("N stars")


            
            
            if TypeIa==True and k == 2:
                ax[j].legend(loc = 1)
            else:
                ax[j].legend(loc = 2)


           
                
        ax[2].set_xlabel(params[k])

        plt.subplots_adjust(hspace = 0.06)
        dir_replaced = ".." + ''.join((dir.split("."))[2:])
        plt.savefig("../figs/hist_param%i"%(k) + "_" + '_'.join((dir_replaced.split("_"))[-5:]) + ".pdf")
        
    return()


#def compare_models(dirlist):





#    return()







def read_netcdf(netcdf_file, TypeIa):

    Inference_Data = az.from_netcdf(netcdf_file)

    trace = Inference_Data['posterior']

    df = az.summary(trace, round_to = 5)

    if TypeIa == True:
        
        params = [df['mean']['alpha'], df['mean']['zcc'], df['mean']['f_Ia']]
        bottom = [df['hdi_3%']['alpha'], df['hdi_3%']['zcc'], df['hdi_3%']['f_Ia']]
        top = [df['hdi_97%']['alpha'], df['hdi_97%']['zcc'], df['hdi_97%']['f_Ia']]
    else:
        params = [df['mean']['alpha'], df['mean']['zcc']]
        bottom = [df['hdi_3%']['alpha'], df['hdi_3%']['zcc']]
        top = [df['hdi_97%']['alpha'], df['hdi_97%']['zcc']]
    
    return(params, bottom, top)



def compare_moodels(dirlist):
    
    # Get stellar data
    catalog = "../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat"
    dfcat = pd.read_csv(catalog)

    
    starnames = ()
    fehs = ()
    abuclasses = ()
    bestmodelnames = ()
    
    
    
    for i, starname in enumerate(dfcat['sobject_id_1']):
        feh = dfcat['fe_h'][i]
        abuclass = dfcat['abuclass'][i]
        
     
        for j, dir in enumerate(dirlist):
            
            modelname = ((dir.split('/'))[2]).replace("MCMCresults_", "")
        
        
            datafile = dir + str(starname) + "_trace.nc"
            if os.path.isfile(datafile):
                data = az.from_netcdf(datafile)
            else:
                continue
            
            if j == 0:
                comp_dict = { modelname: data}
            else:
                comp_dict[modelname] = data 
                
        if len(comp_dict) >= 2:
            df = az.compare(comp_dict, ic = "waic")
    
            bestmodelnames = np.append(bestmodelnames, str(df[df['rank']==0].index))
        
        
            starnames = np.append(starnames, starname)
            fehs = np.append(fehs, feh)
            abuclasses = np.append(abuclasses, abuclass)
        
            #print(str(df[df['rank']==0].index))
        
            df.to_csv("../model_compare/model_compare_" + str(starname) + ".csv")
        else:
            continue
    
    
    
    summary_data = {'starname': starnames, 'feh': fehs, 'abuclass': abuclasses, \
                    'bestmodel': bestmodelnames} 
    
    df_new = pd.DataFrame(summary_data)
    df_new.to_csv("../model_compare/bestmodels.csv", index = False)
    
    return()




def plot_bestmodel_hist(bestmodelfile, axislabelcolor = False):

      
    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")
    alc = "cornsilk"  # FFF8DC in colorcode
    
    df0 = pd.read_csv(bestmodelfile)
    
    fig, ax = plt.subplots(3, 1, figsize = (8, 9), sharex = True)
    
    abuclasses = ['high-alpha', 'low-alpha', 'metal-poor']

    labs = [r'High-$\alpha$', r'Low-$\alpha$', r'Metal-poor']
    
    for i, abuclass in enumerate(abuclasses):

        
        
        
        df = df0[df0['abuclass'] == abuclass]
        
        bestmodels = df['bestmodel'].values
        

        modelids = ["fCh1.00_CC1_Ia1_woSiCa0_Znuplim0", \
                    "fCh0.50_CC1_Ia1_woSiCa0_Znuplim0", \
                    "fCh0.20_CC1_Ia1_woSiCa0_Znuplim0", \
                    "fCh0.00_CC1_Ia1_woSiCa0_Znuplim0", \
                    "fCh0.00_CC1_Ia0_woSiCa0_Znuplim0"]

        Y = np.zeros(len(modelids))
        for j, modelid in enumerate(modelids): 
        
            for bestmodel in bestmodels:
                if re.search(modelid, bestmodel):
                    Y[j] = Y[j] + 1
    
        pos = np.arange(0, len(Y), 1)
        ax[i].bar(pos, Y, color = cmap(i), label = labs[i], alpha = 0.8)
        ax[i].set_xticks(pos)
        if i == 2:
            yticks = [0, 1, 2]
            ax[i].set_ylim(0., 2.3)
        else:
            yticks = np.arange(0, int(np.max(Y)*1.2), int(np.max(Y)/3.))
        ax[i].set_yticks(yticks)
        ax[i].set_yticklabels(yticks)
        
        ax[i].set_ylabel("N stars")
        ax[i].legend()

    ax[2].set_xticklabels([r"$f_{\rm Ch}$=1.0", r"0.5", r"0.2", r"0.0", "CC-only"])

    if axislabelcolor == True:
        for i in range(0, 3):
            ax[i].tick_params(axis='both', which = 'both', colors = alc)
            ax[i].set_ylabel("N stars", color = alc)
                

    
    plt.subplots_adjust(hspace = 0.06)
    plt.savefig("../figs/model_comparison_hist.pdf")
    
    return



def plot_PopIIImodels_hist(model = "CCfix"):

    
    plt.rcParams["font.size"] = 14
    cmap = plt.get_cmap("tab10")


    df = pd.read_csv("../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat")


    resultpath = "../../fit_yield/chi2_XH_GALAHDR3_" + model + "/abupattern/"


    X = [13, 15, 25, 40, 100]

    masses = ()
    abuclasses = ()

    if model == "CCMetals":
        ccfracs = ()
    
    for i, starname in enumerate(df["sobject_id_1"]):

        abuclass = df["abuclass"][df["sobject_id_1"] == starname]


        
        file = resultpath + "bestparams_" + str(starname) + ".dat"
        
        
        f = open(file)
        line = f.readlines()
        f.close()

        linedata = (line[0]).split()

        mass = int(float(linedata[3]))
        
        masses = np.append(masses, mass)
        abuclasses = np.append(abuclasses, abuclass)

        if model == "CCMetals":
            ccfrac = float(linedata[12])
            ccfracs = np.append(ccfracs, ccfrac)
        
        
    fig, ax = plt.subplots(1, 1)

    abucs = ["high-alpha", "low-alpha", "metal-poor"]
    abucs_labs = [r"High-$\alpha$", r"Low-$\alpha$", "Metal-poor"]

    dX = [-0.25, 0.00, 0.25]
    
    for k, abuc in enumerate(abucs):


        Y = ()
        for XX in X:

            filt = (abuclasses == abuc) & (masses == XX)

            print(len(masses[filt]))
            Y = np.append(Y, len(masses[filt]))
            
        pos = np.arange(0, 5, 1)
        ax.bar(pos + dX[k], Y, color = cmap(k), width = 0.25, \
               label = abucs_labs[k], alpha = 0.8)
        
    ax.set_xticks(pos)
    ax.set_xticklabels(['13', '15', '25', '40', '100'])
    ax.set_xlabel(r"Pop III stellar mass [$M_{\odot}$]")
    ax.set_ylabel("N stars")
    ax.legend()

    plt.tight_layout()
    plt.savefig("../figs/PopIIImass_" + model + ".pdf")



    if model == "CCMetals":
        

        fig, bx = plt.subplots(1, 1)

        align = ["left", "mid", "right"]
        bins = np.arange(0., 1.1, 0.1)
        for k, abuc in enumerate(abucs):
            
            filt = (abuclasses == abuc)
            
            bx.hist(ccfracs[filt], bins = bins, align = align[k], \
                    color = cmap(k), histtype = "bar", rwidth = 0.4, \
                    label = abucs_labs[k], alpha = 0.8)
        
        
    plt.show()
    
    

    return()



def plot_GALAH_abupatterns():
    
    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")  
    # Get Observed [Fe/H]
    
    df1 = pd.read_csv("../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat")
    starnames=df1["sobject_id_1"]

    
    #masses=np.zeros(np.size(starnames))
    #ens=np.zeros_like(masses)
    #fehs=np.zeros_like(masses)
    #logfs=np.zeros_like(masses)
    #mmixs=np.zeros_like(masses)
    #mhdils=np.zeros_like(masses)
    #mnis=np.zeros_like(masses)
    #mrems=np.zeros_like(masses)
    #rchi2s=np.zeros_like(masses)
   
    starnames0=()
    masses=()
    ens=()
    fehs=()
    logfs=()
    mmixs=()
    mhdils=()
    mnis=()
    mrems=()
    chi2s=()
    dofs=()
    rchi2s=()
    abuclasses = ()
    
    starnames0_CC=()
    masses_CC=()
    ens_CC=()
    fehs_CC=()
    logfs_CC=()
    mmixs_CC=()
    mhdils_CC=()
    mnis_CC=()
    mrems_CC=()
    ccfrac_CC=()
    zcc_CC=()
    chi2s_CC=()
    dofs_CC=()
    rchi2s_CC=()
    abuclasses_CC = ()
    
    starnames0_Ia=()
    masses_Ia=()
    ens_Ia=()
    fehs_Ia=()
    logfs_Ia=()
    mmixs_Ia=()
    mhdils_Ia=()
    mnis_Ia=()
    mrems_Ia=()
    Iafracs=()
    zcc_Ia=()
    chi2s_Ia=()
    dofs_Ia=()
    rchi2s_Ia=()
    
    
    
    
    for i,starname in enumerate(starnames): 

        abuclass = (df1["abuclass"][df1["sobject_id_1"] == starname]).values


        if abuclass[0] == "high-alpha":
            abucl = r"High-$\alpha$"

        elif abuclass[0] == "low-alpha":
            abucl = r"Low-$\alpha$"

        elif abuclass[0] == "metal-poor":
            abucl = r"Metal-poor"
            
        
        starname=np.int(starname)

       
        feh=np.float64(df1["fe_h"][i])
        
        print(starname,feh)
        
        # Make four different plots 
        
        for k in range(0, 2):
            
            
            if k==0:
              
                ## Pop III yield
                yieldpath = '../../fit_yield/chi2_XH_GALAHDR3_CCfix/'
                
                bestfile = yieldpath + 'abupattern/bestparams_' + \
                    np.str(starname) + '.dat'
       
    
                mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem = \
                    get_bestparams(bestfile)
                rchi2=chi2/dof
       
            
                starnames0=np.hstack((starnames0,str(starname)))
                fehs=np.hstack((fehs,fehobs))
                masses=np.hstack((masses,mass))
                ens=np.hstack((ens,en))
                mhdils=np.hstack((mhdils,h_mass))
                logfs=np.hstack((logfs,logf))
                mmixs=np.hstack((mmixs,mmix))
                mnis=np.hstack((mnis,mni))
                mrems=np.hstack((mrems,mrem))
                chi2s=np.hstack((chi2s,chi2))
                dofs=np.hstack((dofs,dof))
                rchi2s=np.hstack((rchi2s,rchi2))
                abuclasses = np.hstack((abuclasses, abuclass[0]))
      
                figname = "../figs/abupatterns/PopIIIyield_obs_" + \
                    np.str(starname) + ".pdf"             
            
             
                # Abundance patterns of the best-fit model 
                yieldfile = yieldpath + 'abupattern/abupattern_'+\
                    np.str(starname)+"_%.0f_%.0f_"%(mass,en)+"*.txt"
 
                pop3yield=glob.glob(yieldfile)
    
                if np.size(pop3yield)>1:
                    print('More than one yield file!',pop3yield)
                    sys.exit()
        
                znum0,xfe0=np.loadtxt(pop3yield[0],usecols=(0,1),unpack=True)
                filt = (znum0!=6) & (znum0!=7) & (znum0<31)
                znum=znum0[filt]
                xfe=xfe0[filt]
                
                # Abundance patterns of the 2nd-best-fit mmodel 
                #paramfile=yieldpath+'params/params_'+np.str(starname)+'.dat'
                #nparams=5
                #mass_2nd,en_2nd=get_2ndbestfit(paramfile,nparams)
                
              
                #yieldfile=yieldpath+'abupattern/abupattern_'+\
                #    np.str(starname)+"_%.0f_%.0f"%(mass_2nd,en_2nd)+"*.txt"
 
                #pop3yield=glob.glob(yieldfile)
    
              
                #if np.size(pop3yield)>1:
                #    print('More than one yield file!',pop3yield)
                #    sys.exit()
        
                #znum0_2nd,xfe0_2nd=np.loadtxt(pop3yield[0],usecols=(0,1),unpack=True)
                #filt = (znum0_2nd!=6) & (znum0_2nd!=7) & (znum0_2nd<31)
                #znum_2nd=znum0_2nd[filt]
                #xfe_2nd=xfe0_2nd[filt]
                
                
                
                

            elif k==1:
                yieldpath = '../../fit_yield/chi2_XH_GALAHDR3_CCMetals/'
                bestfile = yieldpath + 'abupattern/bestparams_' + \
                    np.str(starname) + '.dat'
    
    
                mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem,ccfrac,zcc = \
                    get_bestparams_CC(bestfile)
                rchi2=chi2/dof
       
            
                starnames0_CC=np.hstack((starnames0_CC,str(starname)))
                fehs_CC=np.hstack((fehs_CC,fehobs))
                masses_CC=np.hstack((masses_CC,mass))
                ens_CC=np.hstack((ens_CC,en))
                mhdils_CC=np.hstack((mhdils_CC,h_mass))
                logfs_CC=np.hstack((logfs_CC,logf))
                mmixs_CC=np.hstack((mmixs_CC,mmix))
                mnis_CC=np.hstack((mnis_CC,mni))
                mrems_CC=np.hstack((mrems_CC,mrem))
                ccfrac_CC=np.hstack((ccfrac_CC,ccfrac))
                zcc_CC=np.hstack((zcc_CC,zcc))
                rchi2s_CC=np.hstack((rchi2s_CC,rchi2))
                chi2s_CC=np.hstack((chi2s_CC,chi2))
                dofs_CC=np.hstack((dofs_CC,dof))
                abuclasses_CC = np.hstack((abuclasses_CC, abuclass[0]))
                
                ## Pop III yield
                
                #plab=panel_labs[panel_no]
                #plabs[i]=plab
                #figname="GALAH_results/PopIIIyield_obs_"+np.str(starname)+"_chi2"+plab[1]+".eps"
                #    panel_no=panel_no+1
                #else: 
                figname="../figs/abupatterns/PopIIIyield_CC_obs_" + \
                    np.str(starname) + ".pdf"
                #plab=""
                #plabs[i]='---'
                    
                    
                    
                yieldfile=yieldpath+'abupattern/abupattern_'+\
                    np.str(starname)+"*.txt"


                # Abundance patterns of the best-fit model 
                yieldfile = yieldpath + 'abupattern/abupattern_'+\
                    np.str(starname)+"_%.0f_%.0f_"%(mass,en)+"*.txt"
 
                pop3yield=glob.glob(yieldfile)
    
                if np.size(pop3yield)>1:
                    print('More than one yield file!',pop3yield)
                    sys.exit()
        
                znum0,xfe0=np.loadtxt(pop3yield[0],usecols=(0,1),unpack=True)
                filt = (znum0!=6) & (znum0!=7) & (znum0<31)
                znum=znum0[filt]
                xfe=xfe0[filt]
                
            
            elif k==2:
                yieldpath='/Users/ishigakimiho/HMP/fit_yield/chi2_XH_GALAH_Ia/'
                bestfile=yieldpath+'abupattern/bestparams_'+np.str(starname)+'.dat'
    
    
                mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem,Iafrac=get_bestparams_Ia(bestfile)
                rchi2=chi2/dof
       
            
                starnames0_Ia=np.hstack((starnames0_Ia,starname))
                fehs_Ia=np.hstack((fehs_Ia,fehobs))
                masses_Ia=np.hstack((masses_Ia,mass))
                ens_Ia=np.hstack((ens_Ia,en))
                mhdils_Ia=np.hstack((mhdils_Ia,h_mass))
                logfs_Ia=np.hstack((logfs_Ia,logf))
                mmixs_Ia=np.hstack((mmixs_Ia,mmix))
                mnis_Ia=np.hstack((mnis_Ia,mni))
                mrems_Ia=np.hstack((mrems_Ia,mrem))
                Iafracs=np.hstack((Iafracs,Iafrac))
                rchi2s_Ia=np.hstack((rchi2s_Ia,rchi2))
                chi2s_Ia=np.hstack((chi2s_Ia,chi2))
                dofs_Ia=np.hstack((dofs_Ia,dof))
      
               
                figname = "../figs/abupatterns/PopIIIyield_Ia_obs_" + \
                    np.str(starname) + ".pdf"
               
                yieldfile = yieldpath + 'abupattern/abupattern_'+\
                    np.str(starname)+"*.txt"
 
                pop3yield=glob.glob(yieldfile)
    
                if np.size(pop3yield)>1:
                    print('More than one yield file!',pop3yield)
                    sys.exit()
        
                znum0,xfe0=np.loadtxt(pop3yield[0],usecols=(0,1),unpack=True)
                filt = (znum0!=6) & (znum0!=7) & (znum0<31)
                znum=znum0[filt]
                xfe=xfe0[filt]
                
                
                
            
            # Observational data
            obsfile = '../../fit_yield/obsabundance/GALAH_DR3/'+np.str(starname)+'_obs.dat'

            if os.path.isfile(obsfile) == False:
                print("No obsfile for ", starname)
                continue
        
            znumobs,xfeobs,xfeerr,flag = \
                np.loadtxt(obsfile,usecols=(0,1,2,3),unpack=True)
    
    
            # Calculate chi2
         
            if k!=0 and k!=1 and k!=2:
                chi2,ndata,mdil=calc_chi2(znum,xfe,feh,mass_fe,znumobs,xfeobs,xfeerr,flag,feh)
                if k==1 or k==2:
                    dof=ndata
                elif k==3:
                    dof=ndata-6
    
    
            # Calculate residual 
            residuals=np.zeros_like(xfeobs)
        
            for kk,xfeo in enumerate(xfeobs):
                if flag[kk]>0 or znumobs[kk]==21 or znumobs[kk]==22 or znumobs[kk]==23 or \
                znumobs[kk]==29 or znumobs[kk]==6.5:
                    continue
                for kkk,xfet in enumerate(xfe):
                    if znumobs[kk]==znum[kkk]:
                        residuals[kk]=xfeo-xfet
        
        
            fig = plt.figure(figsize=(10,6))
            grid = plt.GridSpec(5, 5, hspace=0.0, wspace=0.0)
            main_ax = fig.add_subplot(grid[:-1, :])
            sub_ax = fig.add_subplot(grid[-1, :], sharex=main_ax)
            y1=-1.45
            y2=1.45
            x1=5
            x2=31
            elems=["C","O","Na","Mg","Al","Si","Ca","Sc","Ti","Cr","Mn","Co","Ni","Zn"]
            elemzs=np.array([6,8,11,12,13,14,20,21,22,24,25,27,28,30])

        
           
            # Theoretical lower limit
            melemlows=[21,22,23,29]
            for melemlow in melemlows:
                mlow=xfe[znum==melemlow]
                xx1=melemlow-0.1
                xx2=melemlow+0.1
                yy1=mlow
                yy2=5.0
                main_ax.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')
        
            # Theoretical scatter
            melemsigs=[11,13]
            for melemsig in melemsigs:
                msig=xfe[znum==melemsig]
                xx1=melemsig-0.1
                xx2=melemsig+0.1
                yy1=msig-0.5
                yy2=msig+0.5
                main_ax.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')

        
        
            #main_ax.plot(znum,xfe,marker='',linestyle='-',color=cmap(1),linewidth=2)
            
            if k==0:
                main_ax.plot(znum,xfe,marker='',linestyle='-',color=cmap(1),linewidth=2,\
                         label=r"%.0f$M_\odot$, %.0f${\times}10^{51}$erg"%(mass,en))
                #main_ax.plot(znum_2nd,xfe_2nd,marker='',linestyle='--',color=cmap(2),linewidth=2,\
                #          label=r"%.0f$M_\odot$,%.0f${\times}10^{51}$erg"%(mass_2nd,en_2nd))
            elif k==1:
                #main_ax.plot(znum,xfe,marker='',linestyle='-',color=cmap(1),linewidth=2,label=r"$f_{\rm CC}=%.1f$"%(ccfrac))
                main_ax.plot(znum,xfe,marker='',linestyle='-',color=cmap(1),linewidth=2,label=r"%.0f$M_\odot$, %.0f${\times}10^{51}$erg, $f_{\rm CC}=%.1f$"%(mass,en,ccfrac))
            elif k==2:
                main_ax.plot(znum,xfe,marker='',linestyle='-',color=cmap(1),linewidth=2,label=r"$f_{\rm Ia}=%.1f$"%(Iafrac))
    
    
            # Plot observational data
            main_ax.errorbar(znumobs[flag==0],xfeobs[flag==0],xfeerr[flag==0],\
                               linestyle='',marker='o',mec=cmap(0),mfc=cmap(0),ms=12)
            #main_ax.errorbar(znumobs[flag==-2],xfeobs[flag==-2],yerr=([0.0],[0.3]),\
            #                   lolims=True,linestyle='-',marker='_',ecolor=cmap(0),mec=cmap(0),mfc=cmap(0),ms=8)
   
            textout=np.str(starname)+", " + abucl + ", [Fe/H]=$%.2f$"%(feh)
            main_ax.text(x1+(x2-x1)*0.01,y2-(y2-y1)*0.07,textout,ha='left',va='center')
            textout="$\chi^2/$DoF=%.1f/%i"%(chi2,dof)
            #main_ax.text(x1+(x2-x1)*0.02,y1+(y2-y1)*0.05,textout)
        
        
            emlabels=["C","O","Na","Mg","Al","Si","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
            emlabelzs=np.array([6,8,11,12,13,14,20,21,22,23,24,25,26,27,28,29,30])
            for j,elem in enumerate(emlabels):
                if elem=="C" or elem=="Na" or elem=="Al" or elem=="Ca" or elem=="Ti" or elem=="Cr" or \
                elem=="Fe" or elem=="Ni" or elem=="Zn":
                    yfrac=0.17
                else:
                    yfrac=0.22
                main_ax.text(emlabelzs[j],y2-(y2-y1)*yfrac,elem,ha='center')
    
    
            main_ax.set_ylim(y1,y2)
            main_ax.set_xlim(x1,x2)
            main_ax.set_ylabel("[X/Fe]")
     
            #if k==0:
        
            #    main_ax.text(x1+(x2-x1)*0.02,y2-(y2-y1)*0.05,plab,ha='left',va='center',color='k',fontsize=16)
    
            
            main_ax.tick_params(labelbottom=False) 
    
            sub_ax.bar(znumobs,residuals,fc=cmap(2),ec=cmap(2))
            sub_ax.plot([0,30],[0.0,0.0],linestyle="-",marker="",color=cmap(7))
            sub_ax.set_ylim(-0.8,0.8)
            sub_ax.set_ylabel("Residual")
            sub_ax.set_xlabel("Z")
    
            if k==0 or k==1 or k==2:
                main_ax.legend(loc=4,prop={"size":14})
            
                
                
            plt.savefig(figname)
            #plt.show()
            if k==0:
                a=np.array([znumobs,residuals])
                aa=a.T
                np.savetxt("../outputs/abupatterns/Residual_"+np.str(starname)+".dat",aa,delimiter=",",header="Z,delta[X/Fe](data-model)")
        

    for abuc in ["high-alpha", "low-alpha", "metal-poor"]:

        filt = (abuclasses == abuc) & (dofs >= 3.)
        indx = np.argmin(rchi2s[filt])
        #indx = np.argmax(dofs[filt])
        starname_minchi2 = str((starnames0[filt])[indx])

        print("minimum rchi2 for " + abuc + \
              " occurs for " + starname_minchi2 + " at %.1f"%((rchi2s[filt])[indx]) )
        
        #print("Maximum number of data for  " + abuc + \
        #      " occurs for " + starname_minchi2 + " at %.1f"%((dofs[filt])[indx]) )

    # Produce table for Model A

    tabledata = {"Starname":starnames0, "[Fe/H]": fehs, \
                 "Group":abuclasses, "M": masses, "E51": ens, \
                 "Mmix": mmixs, "logfej": logfs, "M_H": mhdils, \
                 "M_Ni": mnis, "chi2": chi2s, "DoF": dofs}

    columns = ["Starname", "[Fe/H]", "Group", "M", "E51", "Mmix", \
               "logfej", "M_H", "M_Ni", "chi2", "DoF"]
    head = ["GALAH DR3 ID", "[Fe/H]", "OHS subgroup", r"$M$", r"$E_{51}$", \
            r"$M_{\rm mix}$", \
               r"$\log f_{\rm ej}$", r"$M_{\rm H}$", r"$M_{\rm Ni}$", \
            r"$\chi^2$", "DoF"]

    formatters = [fmtis, fmt2, fmts, fmt1, fmt1, fmt1, fmt1, fmte2, fmte2, fmt2, fmt0]
    tablename = "../tabs/Table_A1.tmp.tex"
    csvtablename = "../tabs/Table_A1.csv"

    df = pd.DataFrame(tabledata)

    df[:5].to_latex(tablename, columns = columns, header = head, \
                escape = False, index = False, formatters = formatters)


    df["Starname"] = df["Starname"].map('{:s}'.format)
    df["[Fe/H]"] = df["[Fe/H]"].map('{:.2f}'.format)
    df["Mmix"] = df["Mmix"].map('{:.1f}'.format)
    df["M_H"] = df["M_H"].map('{:.1e}'.format)
    df["M_Ni"] = df["M_Ni"].map('{:.1e}'.format)
    df["chi2"] = df["chi2"].map('{:.2f}'.format)

    df.to_csv(csvtablename, index = False)


    # Produce table for Model B

    tabledata = {"Starname":starnames0_CC, "[Fe/H]": fehs_CC, \
                 "Group":abuclasses_CC, "M": masses_CC, "E51": ens_CC, \
                 "Mmix": mmixs_CC, "logfej": logfs_CC, "M_H": mhdils_CC, \
                 "M_Ni": mnis_CC, "f_CC":ccfrac_CC, "chi2": chi2s_CC, \
                 "DoF": dofs_CC}

    columns = ["Starname", "[Fe/H]", "Group", "M", "E51", "Mmix", \
               "logfej", "M_H", "M_Ni", "f_CC", "chi2", "DoF"]
    head = ["GALAH DR3 ID", "[Fe/H]", "OHS subgroup", r"$M$", r"$E_{51}$", \
            r"$M_{\rm mix}$", \
               r"$\log f_{\rm ej}$", r"$M_{\rm H}$", r"$M_{\rm Ni}$", \
            r"$f_{\rm CC}$", r"$\chi^2$", "DoF"]

    formatters = [fmtis, fmt2, fmts, fmt1, fmt1, fmt1, fmt1, fmte2, fmte2, \
                  fmt2, fmt2, fmt0]
    tablename = "../tabs/Table_A2.tmp.tex"
    csvtablename = "../tabs/Table_A2.csv"

    df = pd.DataFrame(tabledata)

    df[:5].to_latex(tablename, columns = columns, header = head, \
                escape = False, index = False, formatters = formatters)

    df["Starname"] = df["Starname"].map('{:s}'.format)
    df["[Fe/H]"] = df["[Fe/H]"].map('{:.2f}'.format)
    df["Mmix"] = df["Mmix"].map('{:.1f}'.format)
    df["M_H"] = df["M_H"].map('{:.1e}'.format)
    df["M_Ni"] = df["M_Ni"].map('{:.1e}'.format)
    df["f_CC"] = df["f_CC"].map('{:.1f}'.format)
    df["chi2"] = df["chi2"].map('{:.2f}'.format)

    
    df.to_csv(csvtablename, index = False)









    

    #modelid=np.zeros(np.size(starnames0),dtype=object)
    #modelid[:]="A"
    #modeltype=np.zeros(np.size(starnames0),dtype=object)
    #modeltype[:]="Pop III"
    #ccfrac=np.zeros(np.size(starnames0),dtype=float)
    #ccfrac[:]=-9.99
    #zcc=np.zeros(np.size(starnames0),dtype=float)
    #zcc[:]=-9.99
    
    #modelid_CC=np.zeros(np.size(starnames0),dtype=object)
    #modelid_CC[:]="B"
    
    #modeltype_CC=np.zeros(np.size(starnames0),dtype=object)
    #modeltype_CC[:]="Pop III + CC"
    
    
    # see http://stev.oapd.inaf.it/cgi-bin/cmd
    #Z_sun=0.0152 
    
    #df = pd.DataFrame({ 'ModelID': pd.Series(np.hstack((modelid,modelid_CC))),
    #                    'Modeltype': pd.Series(np.hstack((modeltype,modeltype_CC))),
    #                    'Starname' : pd.Series(np.hstack((starnames0,starnames0_CC))),
    #                   'fehobs': pd.Series(np.hstack((fehs,fehs_CC))),
    #                     'M' : pd.Series(np.hstack((masses,masses_CC))),
    #                    'E' : pd.Series(np.hstack((ens,ens_CC))),
    #                    'logf' : pd.Series(np.hstack((logfs,logfs_CC))),
    #                    'Mmix' : pd.Series(np.hstack((mmixs,mmixs_CC))),
    #                   'Hmass': pd.Series(np.hstack((mhdils,mhdils_CC))),
    #                   'ccfrac': pd.Series(np.hstack((ccfrac,ccfrac_CC))),
    #                   'zcc':pd.Series(np.hstack((zcc,zcc_CC))),
    #                   'zcc_zsun':pd.Series(np.hstack((zcc,zcc_CC/Z_sun))),
    #                    'Mni' : pd.Series(np.hstack((mnis,mnis_CC))), 
    #                    'Mrem': pd.Series(np.hstack((mrems,mrems_CC)))  , 
    #                     'chi2': pd.Series(np.hstack((chi2s,chi2s_CC))), 
    #                     'dof': pd.Series(np.hstack((dofs,dofs_CC))),
    #                      'rchi2':pd.Series(np.hstack((rchi2s,rchi2s_CC)))})
    

    #df_s = df.sort_values(by=['Modeltype','fehobs'],ascending=[True,False])
    
    #df_s['figlabels']=pd.Series(["d","e","b","f","c","g","a","h","-","-","-","-","-","-","-","-","-","-","-","-","-","-"])
    
    #df_s_tolatex=df_s[['ModelID','Modeltype','Starname','fehobs','M','E','logf',\
    #                                    'Mmix','Hmass','ccfrac','zcc_zsun','rchi2']]
    
    #df_s_tolatex.to_latex("Table2.tmp.tex",columns=['ModelID','Modeltype','Starname','fehobs','M','E','logf',\
    #                                    'Mmix','Hmass','ccfrac','zcc_zsun','rchi2'],\
    #                      header=["Model ID","Yields","GALAH DR2 ID", "[Fe/H]","$M$","$E$","$\log f_{\\rm ej}$","$M_{\\rm mix}$","$M_{\\rm H}$",\
    #                             "$f_{\\rm CC}$","$Z_{\\rm CC}/Z_{\\odot}$","$\chi^2$/DoF"],\
    #                     escape=False,index=False,\
    #            formatters=[fmtsmr,fmtsmr,fmts,fmt2,fmt0,fmt1,fmt1,fmt2,fmt0,fmt3ors,fmt3ors,fmt2])


    #df_s_c=df_s[['Modeltype','Starname','fehobs','M','E','logf',\
    #            'Mmix','Hmass','ccfrac','zcc_zsun','Mni',\
    #            'Mrem','chi2','dof','rchi2']]
    
    #df_s_c.to_csv("Table2.csv")
    
    
    #sys.exit()
    
    # Pop III+CC
    #df = pd.DataFrame({ 'Starname' : pd.Series(starnames0_CC),
    #                   'fehobs': pd.Series(fehs_CC),
    #                    'M' : pd.Series(masses_CC),
    #                    'E' : pd.Series(ens_CC),
    #                    'logf' : pd.Series(logfs_CC),
    #                    'Mmix' : pd.Series(mmixs_CC),
    #                   'ccfrac': pd.Series(ccfrac_CC),
    #                   'Hmass': pd.Series(mhdils_CC),
    #                    'Mni' : pd.Series(mnis_CC), 
    #                    'Mrem': pd.Series(mrems_CC), 
    #                     'chi2': pd.Series(chi2s_CC),
    #                    'dof':pd.Series(dofs_CC)})
    

    #df_s = df.sort_values('fehobs')
    
    #df_s['figlabels']=pd.Series(["d","e","b","f","c","g","a","h","-","-","-","-","-","-","-","-","-","-","-","-","-","-"])
    
    #df_s.to_latex("Table2.tex",['Starname','fehobs','M','E','logf',\
    #                                    'Mmix','ccfrac','Hmass','Mni',\
    #                                    'Mrem','chi2','dof'],\
    #                      header=["GALAH DR2 ID", "[Fe/H]","$M$","$E$","$\log f_{\\rm ej}$","$M_{\\rm mix}$",\
    #                              "$f_{\\rm CC}$","$M_{\\rm H}$","$M_{\\rm Ni}$",\
    #                             "$M_{\\rm rem}$", "$\chi^2$","DoF"],\
    #                     escape=False,index=False,\
    #            formatters=[fmts,fmt2,fmt0,fmt1,fmt1,fmt2,fmt1,fmte1,fmte1,fmt2,fmt2,fmti])


    #df_s_c=df_s['Starname','fehobs','M','E','logf',\
    #                                    'Mmix','ccfrac','Hmass','Mni',\
    #                                    'Mrem','chi2','dof']
    #df_s_c.to_csv("Table2.csv")
    

    return()



def plot_PopIIImodels_paramhist():

    df = pd.read_csv("../tabs/Table_A1.csv")

    fig,ax = plt.subplots(1, 1)
    ax.hist(df["M_H"])

    plt.show()


    
    return


    







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



def Compare_Models():
    
    plt.rcParams["font.size"] = 15
    cmap = plt.get_cmap("tab10")  
    # Get Observed [Fe/H]
    
    
    df1 = pd.read_csv("../../data/GALAH_DR3/GALAH_DR3_XFe_SelectedSample.dat")
    starnames=df1["sobject_id_1"]
    
    
    for i,starname in enumerate(starnames): 
        
        if (starname!= 160919005701146) and (starname != 140808004701109) and \
        (starname != 160426003501078) : 
            continue
        
        starname=np.int(starname)
        
     

        feh = np.float64(df1["fe_h"][i])

        
        zmax = ad.get_zfrac(feh)
        
        # Parammeters are alpha,zcc, f_Ia
        #    z_Ia is assumed to be 0 if Z<0.1Msun
        #                         0.1Msun if Z>=0.1Msun
        
        zsun=0.0152
        if zmax < 0.1*zsun:
            zIa = 0.0
        else:
            zIa = 0.1*zsun
    
        
        figname = "../figs/Compare_models_"+np.str(starname)+".pdf"
        
        fig, ax = plt.subplots(1,1, figsize = (9,5))
        
        y1 = -1.45
        y2 = 1.45
        x1 = 5
        x2 = 31
        elems = ["C","O","Na","Mg","Al","Si","Ca","Sc","Ti","Cr","Mn","Co","Ni","Zn"]
        elemzs = np.array([6,8,11,12,13,14,20,21,22,24,25,27,28,30])


        
        # Mode yields 
        
        for k in range(0,4):
            
            #if k>=2:
            #    continue
            
            if k==0:
                  
                lab = "PopIII CCSN"
                ls = "--"
            
                ## Pop III yield
                yieldpath='../../fit_yield/chi2_XH_GALAHDR3_CCfix/'
                
                bestfile=yieldpath+'abupattern/bestparams_'+np.str(starname)+'.dat'
       
    
                mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem=get_bestparams(bestfile)
                rchi2=chi2/dof
       
            
        
             
                # Abundance patterns of the best-fit model 
                yieldfile=yieldpath+'abupattern/abupattern_'+\
                    np.str(starname)+"_%.0f_%.0f_"%(mass,en)+"*.txt"
 
                pop3yield=glob.glob(yieldfile)
    
                if np.size(pop3yield)>1:
                    print('More than one yield file!',pop3yield)
                    sys.exit()
        
                znum0,xfe0=np.loadtxt(pop3yield[0],usecols=(0,1),unpack=True)
                
                
                

            elif k==1:
                
                lab = "PopIII + Normal CCSNe"
                ls = ":"
                
                yieldpath='../../fit_yield/chi2_XH_GALAHDR3_CCMetals/'
                bestfile=yieldpath+'abupattern/bestparams_'+np.str(starname)+'.dat'
    
    
                mass,en,h_mass,chi2,dof,fehobs,logf,mmix,mni,mrem,ccfrac,zcc=get_bestparams_CC(bestfile)
                rchi2=chi2/dof
       
            
                # Abundance patterns of the best-fit model 
                yieldfile=yieldpath+'abupattern/abupattern_'+\
                    np.str(starname)+"_%.0f_%.0f_"%(mass,en)+"*.txt"
 
                    
                    
                #yieldfile=yieldpath+'abupattern/abupattern_'+\
                #    np.str(starname)+"*.txt"
 
                pop3yield=glob.glob(yieldfile)
    
                if np.size(pop3yield)>1:
                    print('More than one yield file!',pop3yield)
                    sys.exit()
        
                znum0,xfe0=np.loadtxt(pop3yield[0],usecols=(0,1),unpack=True)
                
            
            
            elif k==2:
            
                lab = "Normal CCSNe"
                ls = "-."


                # Get best-fit parameters
                TypeIa = False

                mcmcfile = \
                    "../outputs/MCMCresults_fCh0.00_CC1_Ia0_woSiCa0_Znuplim0/" \
                    + str(starname) + "_trace.nc"
                
                mean_params, bottom, top = read_netcdf(mcmcfile, TypeIa)
                
                znum0, xfe0, mass_fe  = calc_yields_CC(mean_params)   
                
            
            elif k==3:
                
                lab = "Normal CCSNe + SN Ia"
                ls = "-"

                TypeIa = True
                f_Ch = 1.0

                mcmcfile = \
                    "../outputs/MCMCresults_fCh1.00_CC1_Ia1_woSiCa0_Znuplim0/" \
                    + str(starname) + "_trace.nc" 

                mean_params, bottom, top = read_netcdf(mcmcfile, TypeIa)
                znum0, xfe0, mass_fe, mass_Type2, mass_TypeIa = \
                    calc_yields_CC_Ia(mean_params, zIa, f_Ch)   
                

                
            filt = (znum0!=6) & (znum0!=7) & (znum0<31)
            znum=znum0[filt]
            xfe=xfe0[filt]
                    
                
        
            # Theoretical lower limit
            melemlows=[21,22,23,29]
            for melemlow in melemlows:
                mlow=xfe[znum==melemlow]
                xx1=melemlow-0.1
                xx2=melemlow+0.1
                yy1=mlow
                yy2=5.0
                ax.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')
        
            # Theoretical scatter
            #melemsigs=[11,13]
            #for melemsig in melemsigs:
            #    msig=xfe[znum==melemsig]
            #    xx1=melemsig-0.1
            #    xx2=melemsig+0.1
            #    yy1=msig-0.5
            #    yy2=msig+0.5
            #    ax.fill([xx1,xx2,xx2,xx1],[yy1,yy1,yy2,yy2],color='#DCDCDC',edgecolor='#DCDCDC')

        
        
            #main_ax.plot(znum,xfe,marker='',linestyle='-',color=cmap(1),linewidth=2)
            #if k == 0:
            #     ax.plot(znum,xfe,marker='',linestyle=ls,color=cmap(7),linewidth=2,\
            #             label = lab, alpha = 0.5)
            if k == 0 or k == 1 or k == 2 or k == 3:
            
                ax.plot(znum,xfe,marker='',linestyle=ls,color=cmap(k),linewidth=2,\
                         label = lab)

    
    
        # Observational data
        obsfile='../../fit_yield/obsabundance/GALAH_DR3/' + \
            np.str(starname) + '_obs.dat'
        znumobs,xfeobs,xfeerr,flag=np.loadtxt(obsfile,usecols=(0,1,2,3),unpack=True)
    
        # Plot observational data
        ax.errorbar(znumobs[flag==0],xfeobs[flag==0],xfeerr[flag==0],\
                               linestyle='',marker='o',mec='k',mfc='k',ms=10)
            
        textout=np.str(starname)+", [Fe/H]=$%.2f$"%(feh)
        ax.text(x1+(x2-x1)*0.01,y2-(y2-y1)*0.07,textout,ha='left',va='center')
        #textout="$\chi^2/$DoF=%.1f/%i"%(chi2,dof)
        #main_ax.text(x1+(x2-x1)*0.02,y1+(y2-y1)*0.05,textout)
        
        
        emlabels=["C","O","Na","Mg","Al","Si","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn"]
        emlabelzs=np.array([6,8,11,12,13,14,20,21,22,23,24,25,26,27,28,29,30])
        
        for j,elem in enumerate(emlabels):
            if elem=="C" or elem=="Na" or elem=="Al" or elem=="Ca" or elem=="Ti" or elem=="Cr" or \
            elem=="Fe" or elem=="Ni" or elem=="Zn":
                yfrac=0.17
            else:
                yfrac=0.22
            ax.text(emlabelzs[j],y2-(y2-y1)*yfrac,elem,ha='center')
    
    
        ax.set_ylim(y1,y2)
        ax.set_xlim(x1,x2)
        ax.set_ylabel("[X/Fe]")
        ax.set_xlabel("Z")
            
        ax.legend(loc = 3,prop = {"size":14}, ncol = 2)
            
           
                
        plt.savefig(figname)
    
    

    return()



def plot_apogee_abupattern(paramdir, starname, f_Ch=0.2):


    modelid = (paramdir.split("results_"))[1]

    paramfile = paramdir + starname + "_bestfitparams.csv"
    df = pd.read_csv(paramfile)



    # Read observed abundances



    elems = ["c", "o", "na", "mg", "al", "si", "ca", "v", "cr", "mn", "co", "ni", "cu"]
    z, xfe, xfeerr, feh = ad.read_APOGEE_xfe(starname, elems)

    
    abuclasslab="APOGEE"    
    

    zmax = ad.get_zfrac(feh)
    zsun=0.0152
    if zmax < 0.1*zsun:
        zIa = 0.0
    else:
        zIa = 0.1*zsun



    # Get best-fit parameters
    ncfile = paramdir + starname + "_trace.nc"

    mean_params, bottom, top = read_netcdf(ncfile, TypeIa=True)


    chi2 = np.nan
    dof = np.nan
    outpath = "../../../APOGEE_sample/figs/"
    plot_bestfitmodel_CC_Ia(starname,outpath, abuclasslab, z,xfe, \
                                    xfeerr,feh,mean_params, \
                                    zIa,f_Ch,chi2,dof, modelid, \
                                    bottom, top)




if __name__ == "__main__":

    starname = "2M00024677+0127542"

    paramdir = "../../../APOGEE_sample/outputs/MCMCresults_fCh0.20_CC1_Ia1_woSiCa0_Znuplim0/"
    plot_apogee_abupattern(paramdir, starname)



