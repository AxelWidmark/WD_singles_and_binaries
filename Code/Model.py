import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import RectBivariateSpline,RegularGridInterpolator
pi=math.pi
from scipy.special import erfc

def prior_spatial(coords,coordsystem="proper"):
    #everything in kpc here, parallax in milliarcsecond
    if coordsystem=="galcyl":
        r,z,phi=coords
    elif coordsystem=="xyz":
        Rsun=8. #kpc
        r=np.sqrt((Rsun-coords[0])**2.+coords[1]**2.)
        z=coords[2]
        phi=np.arctan(coords[1]/(Rsun-coords[0]))
    elif coordsystem=="proper":
        Rsun=8. #kpc
        l,b,dist=coords
        assert (l>=0. and l<=2.*np.pi and b>=-np.pi/2. and b<=np.pi/2. and dist>=0.)
        #if dist<0.:
        #    return 0.
        r=np.sqrt((np.cos(b)*dist)**2.-2.*Rsun*dist*np.cos(b)*np.cos(l)+Rsun**2.)
        z=np.sin(b)*dist
        phi=np.arctan(np.cos(b)*np.cos(l)*dist-Rsun*dist*np.cos(b)*np.sin(l))
    Lthin=3.5
    Lthick=3.5
    Lhalo=8.5
    Hthin=.26
    Hthick=1.
    fthick=0.06
    fhalo=5e-5
    Rcore=1.
    qhalo=.64
    etahalo=2.
    nms=np.exp(-r/Lthin-np.abs(z)/Hthin)+fthick*np.exp(-r/Lthick-np.abs(z)/Hthick)+fhalo*(np.sqrt(r**2.+(z/qhalo)**2.+Rcore**2.)/Lhalo)**(-etahalo)
    return nms


m0s = np.array( [24.63,25.11,24.80,24.36,22.83] )






from sklearn.mixture import GaussianMixture as GMM
from scipy import spatial
def compute_nearest_neighbours(vector):
    vector = np.array(vector)
    scale_lengths = np.zeros(len(vector[0]))
    volume_rescaling = 1.
    vector_no_duplicates = [vector[0]]
    for i in range(1,len(vector)):
        if vector[i-1].all!=vector[i].all:
            vector_no_duplicates.append(vector[i])
    for i in range(len(vector[0])):
        scale_lengths[i] = np.cov(vector[:,i])**0.5
        volume_rescaling = volume_rescaling*scale_lengths[i]
    for j in range(len(vector)):
        for k in range(len(vector[0])):
            vector[j,k] = vector[j,k]/scale_lengths[k]
    distances,neighbours = spatial.KDTree(vector_no_duplicates).query(vector,k=2)
    volumes = 4.*pi/3.*distances[:,1]**3.*volume_rescaling
    return np.array(neighbours)[:,1],volumes
def calc_evidence(obj_params,obj_probs):
    neighbours,volumes = compute_nearest_neighbours(obj_params)
    return np.sum( np.array(volumes)*obj_probs )


class WD_system():
    
    # Initiation function
    def __init__(self,Gmag,Gmag_err,ugriz_flux,ugriz_flux_err,l,b,parallax,parallax_err):
        self.Gmag = Gmag
        self.Gmag_err = Gmag_err
        self.ugriz_flux = ugriz_flux
        self.ugriz_flux_err = ugriz_flux_err
        self.l = l
        self.b = b
        self.parallax = parallax
        self.parallax_err = parallax_err
        
        self.mass_vec = np.linspace(0.15,1.5,100)
        self.log10age_vec = np.linspace(5.,10.,100)
        
        kx=1
        ky=1
        
        matrix = np.load('../InterpolationMatrices/DA_Gaia.npy')
        matrix[matrix < 1000.] = 0.
        matrix[matrix > 1000.] = 1.
        self.DA_forbidden_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, matrix ,kx=1,ky=1)
        self.DA_Gaia_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DA_Gaia.npy') ,kx=kx,ky=ky)
        self.DA_u_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DA_u.npy') ,kx=kx,ky=ky)
        self.DA_g_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DA_g.npy') ,kx=kx,ky=ky)
        self.DA_r_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DA_r.npy') ,kx=kx,ky=ky)
        self.DA_i_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DA_i.npy') ,kx=kx,ky=ky)
        self.DA_z_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DA_z.npy') ,kx=kx,ky=ky)
        
        matrix = np.load('../InterpolationMatrices/DB_Gaia.npy')
        matrix[matrix < 1000.] = 0.
        matrix[matrix > 1000.] = 1.
        self.DB_forbidden_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, matrix ,kx=1,ky=1)
        self.DB_Gaia_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DB_Gaia.npy') ,kx=kx,ky=ky)
        self.DB_u_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DB_u.npy') ,kx=kx,ky=ky)
        self.DB_g_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DB_g.npy') ,kx=kx,ky=ky)
        self.DB_r_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DB_r.npy') ,kx=kx,ky=ky)
        self.DB_i_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DB_i.npy') ,kx=kx,ky=ky)
        self.DB_z_f = RectBivariateSpline(self.mass_vec,self.log10age_vec, np.load('../InterpolationMatrices/DB_z.npy') ,kx=kx,ky=ky)
        
        self.BmV_to_ugriz = np.array([5.155,3.793,2.751,2.086,1.479])
        
        npzfile = np.load("../Data/3D_dust_interpolation.npz")
        x = npzfile['x']
        y = npzfile['y']
        z = npzfile['z']
        BmV_matrix = npzfile['BmV_matrix']
        self.BmV_interp = RegularGridInterpolator((x,y,z), BmV_matrix, method='linear')
        
    def ugriz_DA(self,mass,log10age):
        return np.array( [self.DA_u_f.ev(mass,log10age), self.DA_g_f.ev(mass,log10age), self.DA_r_f.ev(mass,log10age), self.DA_i_f.ev(mass,log10age), self.DA_z_f.ev(mass,log10age)] )
    
    def ugriz_DB(self,mass,log10age):
        return np.array( [self.DB_u_f.ev(mass,log10age), self.DB_g_f.ev(mass,log10age), self.DB_r_f.ev(mass,log10age), self.DB_i_f.ev(mass,log10age), self.DB_z_f.ev(mass,log10age)] )
    
    ######## LNPROB FUNCTIONS ########
    
    def lnprob_single_flat(self,params,WDtype,distance_trick=False):
        mass,log10age,dist = params
        if dist<0. or log10age>10.:
            return -np.inf
        if WDtype==0.:
            if self.DA_forbidden_f.ev(mass,log10age)!=0.:
                return -np.inf
        elif WDtype==1.:
            if self.DB_forbidden_f.ev(mass,log10age)!=0.:
                return -np.inf
        if WDtype==0.:
            colorModelAbs = self.ugriz_DA(mass,log10age)
        elif WDtype==1.:
            colorModelAbs = self.ugriz_DB(mass,log10age)
        if distance_trick:
            if WDtype==0.:
                GmagModel = self.DA_Gaia_f.ev(mass,log10age)
            elif WDtype==1.:
                GmagModel = self.DB_Gaia_f.ev(mass,log10age)
            Gmag_average_distance = 10.**((self.Gmag-GmagModel)/5.-2.)
            dist = Gmag_average_distance*dist
        dist_pc = min(295.,1e3*dist)
        if dist_pc*math.sin(self.b)>295.:
            dist_pc = dist_pc*dist_pc*math.sin(self.b)/295.
        xyzcoords = [dist_pc*math.cos(self.l)*math.cos(self.b), dist_pc*math.sin(self.l)*math.cos(self.b), dist_pc*math.sin(self.b)]
        BmV = self.BmV_interp(xyzcoords)
        colorModel = colorModelAbs+(5.*(np.log10(1e3*dist)-1.))+BmV*self.BmV_to_ugriz
        fluxModel = np.exp(-0.921034*colorModel)
        # TODO
        # fix priors and normalization
        # safety priors!!
        res = 0.
        for i in range(5):
            res += -(fluxModel[i]-self.ugriz_flux[i])**2./(2.*self.ugriz_flux_err[i]**2.)   -np.log( np.sqrt(2.*pi*self.ugriz_flux_err[i]**2.) )
        res += -(1./dist-self.parallax)**2./(2.*self.parallax_err**2.)   -np.log( np.sqrt(2.*pi*self.parallax_err**2.) )
        res += np.log( dist**2. )
        res += np.log( prior_spatial([self.l,self.b,dist]) )
        # jacobian due to log10age, corresponding to flat prior in age
        res += np.log( np.log(10.)*10.**log10age/1e10 )
        # there is a det(JACOBIAN) term here, due to the distance trick, equal to partial(distance)/partial(params[2])
        if distance_trick:
            res += np.log(Gmag_average_distance)
        # soft boundaries in mass
        if mass<0.225:
            res += -(mass-0.225)**2./(2.*0.05**2.)
        if WDtype==0. and mass>1.375:
            res += -(mass-1.375)**2./(2.*0.05**2.)
        elif WDtype==1. and mass>1.150:
            res += -(mass-1.150)**2./(2.*0.05**2.)
        return res
    
    
    def lnprob_binary_flat(self,params,WDtypes,agediff=5e8,distance_trick=False):
        mass1,log10age1,mass2,age2_rel_age1,dist = params
        if 10.**log10age1+age2_rel_age1>1e5:
            log10age2 = np.log10( 10.**log10age1+age2_rel_age1 )
        else:
            return -np.inf
        WDtype1,WDtype2 = WDtypes
        if dist<0. or log10age1>10. or log10age2>10.:
            return -np.inf
        if WDtype1==0.:
            if self.DA_forbidden_f.ev(mass1,log10age1)!=0.:
                return -np.inf
        elif WDtype1==1.:
            if self.DB_forbidden_f.ev(mass1,log10age1)!=0.:
                return -np.inf
        if WDtype2==0.:
            if self.DA_forbidden_f.ev(mass2,log10age2)!=0.:
                return -np.inf
        elif WDtype2==1.:
            if self.DB_forbidden_f.ev(mass2,log10age2)!=0.:
                return -np.inf
        if WDtype1==0.:
            colorModelAbs1=self.ugriz_DA(mass1,log10age1)
        elif WDtype1==1.:
            colorModelAbs1=self.ugriz_DB(mass1,log10age1)
        if WDtype2==0.:
            colorModelAbs2=self.ugriz_DA(mass2,log10age2)
        elif WDtype2==1.:
            colorModelAbs2=self.ugriz_DB(mass2,log10age2)
        if distance_trick:
            if WDtype1==0.:
                GmagModel1 = self.DA_Gaia_f.ev(mass1,log10age1)
            elif WDtype1==1.:
                GmagModel1 = self.DB_Gaia_f.ev(mass1,log10age1)
            if WDtype2==0.:
                GmagModel2 = self.DA_Gaia_f.ev(mass2,log10age2)
            elif WDtype2==1.:
                GmagModel2 = self.DB_Gaia_f.ev(mass2,log10age2)
            GmagModel = -2.5*np.log10( 10.**(-GmagModel1/2.5) + 10.**(-GmagModel2/2.5) )
            Gmag_average_distance = 10.**((self.Gmag-GmagModel)/5.-2.)
            dist = Gmag_average_distance*dist
        colorModelAbs = -2.5*np.log10( 10.**(-np.array(colorModelAbs1)/2.5) + 10.**(-np.array(colorModelAbs2)/2.5) )
        dist_pc = min(295.,1e3*dist)
        if dist_pc*math.sin(self.b)>295.:
            dist_pc = dist_pc*dist_pc*math.sin(self.b)/295.
        xyzcoords = [dist_pc*math.cos(self.l)*math.cos(self.b), dist_pc*math.sin(self.l)*math.cos(self.b), dist_pc*math.sin(self.b)]
        BmV = self.BmV_interp(xyzcoords)
        colorModel = colorModelAbs+(5.*(np.log10(1e3*dist)-1.))+BmV*self.BmV_to_ugriz
        fluxModel = np.exp(-0.921034*colorModel)
        # TODO
        # fix priors and normalization
        res = 0.
        for i in range(5):
            res += -(fluxModel[i]-self.ugriz_flux[i])**2./(2.*self.ugriz_flux_err[i]**2.)   -np.log( np.sqrt(2.*pi*self.ugriz_flux_err[i]**2.) )
        res += -(1./dist-self.parallax)**2./(2.*self.parallax_err**2.)   -np.log( np.sqrt(2.*pi*self.parallax_err**2.) )
        res += np.log( dist**2. )
        # this term prevents multiplicity in masses, but with a soft edge
        res += np.log( erfc((mass1-mass2)/0.05)/2. )
        res += np.log( prior_spatial([self.l,self.b,dist]) )
        # jacobian due to log10age, corresponding to flat prior in age
        res += np.log( np.log(10.)*(10.**log10age1)/1e10 )
        # agediff prior, accounts for normalization constant coming from age difference distribution
        res += -1./2.*(age2_rel_age1/agediff)**2.    -np.log( np.sqrt(2.*pi*agediff**2.) )
        # there is a det(JACOBIAN) term here, due to the distance trick, equal to partial(distance)/partial(params[2])
        if distance_trick:
            res += np.log(Gmag_average_distance)
        if mass1<0.225:
            res += -(mass1-0.225)**2./(2.*0.05**2.)
        if WDtype1==0. and mass1>1.375:
            res += -(mass1-1.375)**2./(2.*0.05**2.)
        elif WDtype1==1. and mass1>1.150:
            res += -(mass1-1.150)**2./(2.*0.05**2.)
        if mass2<0.225:
            res += -(mass2-0.225)**2./(2.*0.05**2.)
        if WDtype2==0. and mass2>1.375:
            res += -(mass2-1.375)**2./(2.*0.05**2.)
        elif WDtype2==1. and mass2>1.150:
            res += -(mass2-1.150)**2./(2.*0.05**2.)
        return res
    
    def fit_single_GMM(self,index):
        masses = np.linspace(0.15,1.5,10)
        log10ages = np.linspace(5.,10.,15)
        for WDtype in [0.,1.]:
            print('\n\n\n')
            print('WDtype:',WDtype)
            best = [-np.inf,None,None]
            for mass_i in range(len(masses)):
                mass = masses[mass_i]
                for log10age_i in range(len(log10ages)):
                    log10age = log10ages[log10age_i]
                    res = self.lnprob_single_flat([mass,log10age,1./self.parallax],WDtype)
                    if res>best[0]:
                        best = [res,mass,log10age]
            def ln_func(params):
                return self.lnprob_single_flat(params,WDtype,distance_trick=True)
            p0 = [best[1],best[2],1.]
            initialstepcovar = np.array( [0.1**2., 0.1**2., 1e-2**2.] )
            s = sampler(ln_func,3,p0,initialstepcovar)
            for run_i in range(1,6):
                s.run(100)
                s.set_stepcovar(10**(-run_i)*initialstepcovar)
            s.burnin(1000)
            s.burnin(1000)
            s.burnin(2000)
            s.burnin(3000)
            s.set_stepcovar(0.2*s.get_stepcovar())
            s.burnin(5000)
            s.set_stepcovar(0.2*s.get_stepcovar())
            chain,lnprobchain,acc = s.run(30000)
            thinning_factor = min(int(5./acc),50)
            num_steps = int(3e4*thinning_factor-3e4)
            print('Thinning factor:',thinning_factor)
            print('Number of steps:',num_steps)
            s.run(num_steps)
            chain = s.get_chain()[::thinning_factor]
            lnprobchain = s.get_lnprobchain()[::thinning_factor]
            evidence = calc_evidence(chain,np.exp(lnprobchain))
            print('Evidence:',evidence)
            bic0 = np.inf
            for n_comp in range(2,7):
                gmm = GMM(n_components=n_comp,max_iter=10000,tol=1e-4,n_init=5)
                gmm.fit([[c[0]] for c in chain])
                bic = gmm.bic(np.array( [[c[0]] for c in chain] ))
                if bic<bic0:
                    bic0 = bic
                    gmm_weights = gmm.weights_
                    gmm_means = gmm.means_
                    gmm_covs = gmm.covariances_
                else:
                    break
            print('# of GMM components:',len(gmm_weights))
            np.savez('../Data/Obj_GMMs/'+str(index)+'_'+str(int(WDtype)),evidence=evidence,gmm_weights=gmm_weights,gmm_means=gmm_means,gmm_covs=gmm_covs)
        return 0.
    
    
    def fit_binary_GMM(self,index,agediff=5e8):
        masses = np.linspace(0.15,1.5,10)
        log10ages = np.linspace(5.,10.,15)
        for WDtypes in [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]:
            print('\n\n\n')
            print('WDtypes:',WDtypes)
            best = [-np.inf,None,None,None,None]
            for mass_i in range(len(masses)):
                mass1 = masses[mass_i]
                for mass_j in range(mass_i,len(masses)):
                    mass2 = masses[mass_j]
                    for log10age_i in range(len(log10ages)):
                        log10age1 = log10ages[log10age_i]
                        for age2_rel_age1 in np.linspace(10.**log10age1-agediff,10.**log10age1+agediff,5):
                            res = self.lnprob_binary_flat([mass1,log10age1,mass2,age2_rel_age1,1./self.parallax],WDtypes,agediff=agediff)
                            if res>best[0]:
                                best = [res,mass1,log10age1,mass2,age2_rel_age1]
            def ln_func(params):
                return self.lnprob_binary_flat(params,WDtypes,agediff=agediff,distance_trick=True)
            p0 = [best[1],best[2],best[3],best[4],1.]
            initialstepcovar = np.array( [0.1**2., 0.1**2., 0.1**2., agediff**2., 1e-2**2.] )
            s = sampler(ln_func,5,p0,initialstepcovar)
            for run_i in range(1,5):
                s.run(1000)
                s.set_stepcovar(10**(-run_i)*initialstepcovar)
            s.burnin(1000)
            s.burnin(1000)
            s.burnin(2000)
            s.burnin(3000)
            s.set_stepcovar(0.2*s.get_stepcovar())
            s.burnin(5000)
            s.set_stepcovar(0.2*s.get_stepcovar())
            chain,lnprobchain,acc = s.run(30000)
            thinning_factor = min(int(5./acc),50)
            num_steps = int(3e4*thinning_factor-3e4)
            print('Thinning factor:',thinning_factor)
            print('Number of steps:',num_steps)
            s.run(num_steps)
            chain = s.get_chain()[::thinning_factor]
            lnprobchain = s.get_lnprobchain()[::thinning_factor]
            evidence = calc_evidence(chain,np.exp(lnprobchain))
            print('Evidence:',evidence)
            bic0 = np.inf
            for n_comp in range(2,9):
                gmm = GMM(n_components=n_comp,max_iter=10000,tol=1e-4,n_init=5)
                gmm.fit([[c[0],c[2]] for c in chain],GMM)
                #print("GMM likelihood:",np.exp(gmm.lower_bound_))
                bic = gmm.bic(np.array( [[c[0],c[2]] for c in chain] ))
                if bic<bic0:
                    bic0 = bic
                    gmm_weights = gmm.weights_
                    gmm_means = gmm.means_
                    gmm_covs = gmm.covariances_
                else:
                    break
            print('# of GMM components:',len(gmm_weights))
            np.savez('../Data/Obj_GMMs/'+str(index)+'_'+str(int(WDtypes[0]))+'-'+str(int(WDtypes[1]))+'_'+str(int(agediff/1e6))+'Myr',evidence=evidence,gmm_weights=gmm_weights,gmm_means=gmm_means,gmm_covs=gmm_covs)
        return 0.



class sampler():
      
    def __init__(self,lnprob,dim,p0,initialstepcovar,sub=False):
        self.chain = [p0]
        self.lnprobchain = None
        if np.shape(initialstepcovar)==(dim-sub,dim-sub):
            self.stepcovar = initialstepcovar
        elif np.shape(initialstepcovar)==(dim-sub,):
            self.stepcovar=np.zeros((dim-sub,dim-sub))
            for i in range(dim-sub):
                self.stepcovar[i][i]=initialstepcovar[i]
        else:
            raise ValueError('Stepsize vector/matrix has incorrect dimension')
        self.lnprob = lnprob
        self.dim = dim
        self.sub = sub
    
    
    def run(self,n,printsteps=False):
        # tail is the end of the new chain
        tail = np.zeros((n,self.dim))
        lnprobtail = np.zeros(n)
        params0 = self.chain[-1]
        if self.lnprobchain is None:
            lh0 = self.lnprob(params0)
            self.lnprobchain = [float(lh0)]
        else:
            lh0 = float(self.lnprobchain[-1])
        accepts=0
        params1 = np.zeros(self.dim)
        # ITERATE
        for i in range(n):
            if self.sub:
                if np.random.rand()<0.5:
                    WDtype1 = 0.
                else:
                    WDtype1 = 1.
                params1 = list(np.random.multivariate_normal(params0[0:3],self.stepcovar))
                params1.append(WDtype1)
            else:
                params1 = np.random.multivariate_normal(params0,self.stepcovar)
            lh1 = self.lnprob(params1)
            lhratio = np.exp(lh1-lh0)
            if lhratio>1. or lhratio>np.random.rand():
                accepts += 1
                tail[i] = params1
                lnprobtail[i] = float(lh1)
                params0 = list(params1)
                lh0 = lh1
            else:
                tail[i] = params0
                lnprobtail[i] = float(lh0)
            if printsteps:
                print (params0) 
        self.chain = np.concatenate((self.chain,tail))
        self.lnprobchain = np.concatenate((self.lnprobchain,lnprobtail))
        return tail,lnprobtail,float(accepts)/float(n)
    
    
    def burnin(self,n):
        self.clear_chain()
        tail,lnprobtail,acc = self.run(n)
        if acc<.1 or acc>.9:
            print ("BURNIN WARNING, acc =",acc)
            print (self.stepcovar)
        self.set_stepcovar()
        self.clear_chain()
        return acc
    
    
    def set_stepcovar(self,matrix=[]):
        if matrix==[]:
            if self.sub:
                self.stepcovar=np.cov(self.chain[:,0:3],rowvar=False)
            else:
                self.stepcovar=np.cov(self.chain,rowvar=False)
        else:
            if np.shape(matrix)==(self.dim-self.sub,self.dim-self.sub):
                self.stepcovar = matrix
            elif np.shape(matrix)==(self.dim-self.sub,):
                self.stepcovar=np.zeros((self.dim-self.sub,self.dim-self.sub))
                for i in range(self.dim-self.sub):
                    self.stepcovar[i][i]=matrix[i]
    
    
    def clear_chain(self):
        self.chain = [self.chain[-1]]
        self.lnprobchain = [self.lnprobchain[-1]]
        return self.chain
    
    
    def get_chain(self):
        return self.chain
    
    def get_lnprobchain(self):
        return self.lnprobchain
    
    def get_stepcovar(self):
        return self.stepcovar





