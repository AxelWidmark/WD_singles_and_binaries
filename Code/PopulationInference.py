import math
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

pi=math.pi

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
numTFThreads = 2

numpopgaussians = 6
pop_weights = np.random.uniform(0.,1.,numpopgaussians)
pop_means   = np.random.uniform(0.4,1.,numpopgaussians)
pop_covs    = np.random.uniform(0.1**2.,0.2**2.,numpopgaussians)
pop_fractions = np.random.uniform(0.,0.5,(numpopgaussians,3))



pop_weights = np.array(pop_weights)
pop_means = np.array(pop_means)
pop_covs = np.array(pop_covs)
pop_fractions = np.array(pop_fractions)
#pop_fractions[:,1] = 1e-2
#pop_fractions[:,2] = 1e-2

def sigmoid_inv(x):
    return -np.log(1./x-1.)

npzfile = np.load('../Data/all_objects_G-19.npz')
single_objects_evidence      = npzfile['single_objects_evidence']
single_objects_weights       = npzfile['single_objects_weights']
single_objects_means         = npzfile['single_objects_means']
single_objects_covs          = npzfile['single_objects_covs']
double_objects_evidence      = npzfile['double_objects_evidence']
double_objects_weights       = npzfile['double_objects_weights']
double_objects_means         = npzfile['double_objects_means']
double_objects_covs          = npzfile['double_objects_covs']


print( "objects loaded", len(single_objects_evidence) )
#exit()


include_binaries = False
mass_norm_bins = 280
obj_evidence_eps = 0.
print('\n\n\n\n\n')
appGlimit = 19.
npzfile = np.load('../Data/dN_0_G-'+str(int(appGlimit))+'.npz')
mass_0_vec = npzfile['mass_vec']
dN_0 = npzfile['dNdmass_vec']
npzfile = np.load('../Data/dN_1_G-'+str(int(appGlimit))+'.npz')
mass_1_vec = npzfile['mass_vec']
dN_1 = npzfile['dNdmass_vec']
mass_vec = np.linspace(0.1,1.5,mass_norm_bins+1)
mass_norm_stepsize = 1.4/(mass_norm_bins)


def lookupNearest1d(x0, vec, vec1):
    xi = np.abs(vec1-x0).argmin()
    return vec[xi]
dN_0 = [lookupNearest1d(mass1, dN_0, mass_0_vec) for mass1 in mass_vec]
dN_1 = [lookupNearest1d(mass1, dN_1, mass_1_vec) for mass1 in mass_vec]


# TODO remove this line later
mass_1_vec = np.linspace(0.2,1.2,41)

npzfile = np.load('../Data/dN_0-0_G-'+str(int(appGlimit))+'.npz')
dN_00 = npzfile['dNdmass_matrix']
npzfile = np.load('../Data/dN_0-1_G-'+str(int(appGlimit))+'.npz')
dN_01 = npzfile['dNdmass_matrix']
npzfile = np.load('../Data/dN_1-1_G-'+str(int(appGlimit))+'.npz')
dN_11 = npzfile['dNdmass_matrix']
def lookupNearest2d(x0, y0, matrix, vec1, vec2):
    xi = np.abs(vec1-x0).argmin()
    yi = np.abs(vec2-y0).argmin()
    return matrix[xi][yi]
dN_00 = [[lookupNearest2d(mass1, mass2, dN_00, mass_0_vec, mass_0_vec) for mass2 in mass_vec] for mass1 in mass_vec]
dN_01 = [[lookupNearest2d(mass1, mass2, dN_01, mass_0_vec, mass_1_vec) for mass2 in mass_vec] for mass1 in mass_vec]
dN_11 = [[lookupNearest2d(mass1, mass2, dN_11, mass_1_vec, mass_1_vec) for mass2 in mass_vec] for mass1 in mass_vec]



#################################
##### tensorflow code below #####
#################################


# object GMM placeholders
Single_objects_evidence = tf.placeholder(shape=[None,2], dtype=tf.float32)
Single_objects_weights = tf.placeholder(shape=[None,2,6], dtype=tf.float32)
Single_objects_means = tf.placeholder(shape=[None,2,6], dtype=tf.float32)
Single_objects_covs = tf.placeholder(shape=[None,2,6], dtype=tf.float32)
Double_objects_evidence = tf.placeholder(shape=[None,4], dtype=tf.float32)
Double_objects_weights = tf.placeholder(shape=[None,4,8], dtype=tf.float32)
Double_objects_means = tf.placeholder(shape=[None,4,8,2], dtype=tf.float32)
Double_objects_covs = tf.placeholder(shape=[None,4,8,3], dtype=tf.float32)

# normalization placeholders
Mass_0_vec = tf.placeholder(shape=[None], dtype=tf.float32)
Mass_1_vec = tf.placeholder(shape=[None], dtype=tf.float32)
DNm_0 = tf.placeholder(shape=[None], dtype=tf.float32)
DNm_1 = tf.placeholder(shape=[None], dtype=tf.float32)
DNm_00 = tf.placeholder(shape=[None,None], dtype=tf.float32)
DNm_01 = tf.placeholder(shape=[None,None], dtype=tf.float32)
DNm_11 = tf.placeholder(shape=[None,None], dtype=tf.float32)



# posterior variables
Pop_weights_non_normalized = tf.sigmoid( tf.Variable( sigmoid_inv(pop_weights) ,dtype=tf.float32) ) # tf.Variable(np.random.uniform(0.,1.,numpopgaussians),dtype=tf.float32)
Pop_means = 0.2 + 1.2*tf.sigmoid( tf.Variable( sigmoid_inv((pop_means-0.2)/1.2),dtype=tf.float32) ) # tf.Variable(np.random.uniform(0.4,1.0,numpopgaussians),dtype=tf.float32)
Pop_covs = tf.exp( tf.Variable( np.log(pop_covs),dtype=tf.float32)  ) # tf.Variable(np.random.uniform(0.05**2.,0.20**2.,numpopgaussians),dtype=tf.float32)
# Pop_fractions contains: helium fraction, binary fractions for hydrogen and helium

Pop_fractions_atm = tf.sigmoid( tf.Variable( sigmoid_inv(pop_fractions[:,0]), dtype=tf.float32) ) #
if include_binaries:
    Pop_fractions_binary = tf.sigmoid( tf.Variable( sigmoid_inv(pop_fractions[:,1:3]), dtype=tf.float32) )
else:
    Pop_fractions_binary = tf.constant(np.zeros((len(pop_fractions),2)), dtype=tf.float32) #
Pop_fractions = tf.concat( [Pop_fractions_atm[:,None],Pop_fractions_binary], 1 )
     # tf.Variable(np.random.uniform(0.,1.,(numpopgaussians,3)),dtype=tf.float32)

Pop_weights = Pop_weights_non_normalized/tf.reduce_sum(Pop_weights_non_normalized)
Total_binary_fraction = tf.reduce_sum( Pop_weights*( (1.-Pop_fractions[:,0])*Pop_fractions[:,1] + Pop_fractions[:,0]*Pop_fractions[:,2] ) )






### CALCULATE OBJECT EVIDENCES
# indices are: object, obj gauss, pop gauss
Objev_0 = Pop_weights[None, None, :] * (1.-Pop_fractions[None, None, :, 0]) * (1.-Pop_fractions[None, None, :, 1])    \
    * Single_objects_evidence[:, 0, None, None] * Single_objects_weights[:, 0, :, None]   \
    * tf.exp( -tf.pow(Single_objects_means[:, 0, :, None]-Pop_means[None, None, :], 2) / (2.*( Pop_covs[None, None, :]+Single_objects_covs[:, 0, :, None] )) )  \
    / tf.sqrt(2.*pi*( Pop_covs[None, None, :]+Single_objects_covs[:, 0, :, None] ))
Objev_1 = Pop_weights[None, None, :] * Pop_fractions[None, None, :, 0] * (1.-Pop_fractions[None, None, :, 2])    \
    * Single_objects_evidence[:, 1, None, None] * Single_objects_weights[:, 1, :, None]   \
    * tf.exp( -tf.pow(Single_objects_means[:, 1, :, None]-Pop_means[None, None, :], 2) / (2.*( Pop_covs[None, None, :]+Single_objects_covs[:, 1, :, None] )) )  \
    / tf.sqrt(2.*pi*( Pop_covs[None, None, :]+Single_objects_covs[:, 1, :, None] ))
if include_binaries:
    # indices are: object, obj gauss, pop gauss 1, pop gauss 2
    Objev_00 = Pop_weights[None,None,:,None] * (1.-Pop_fractions[None,None,:,None,0]) * Pop_fractions[None,None,:,None,1]       \
        * Pop_weights[None,None,None,:] * (1.-Pop_fractions[None,None,None,:,0]) * Pop_fractions[None,None,None,:,1] / Total_binary_fraction    \
        * Double_objects_evidence[:, 0, None, None, None] * Double_objects_weights[:, 0, :, None, None]   \
        * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 0, :, 1, None, None]) * tf.pow( Double_objects_means[:, 0, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 0, :, 0, None, None]) * tf.pow( Double_objects_means[:, 0, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 0, :, 2, None, None] * ( Double_objects_means[:, 0, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 0, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 0, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 0, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 0, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 0, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 0, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 0, :, 2, None, None], 2)     ) )    \
        * tf.exp( -0.5*tf.pow((Double_objects_means[:, 0, :, 0, None, None]-Double_objects_means[:, 0, :, 1, None, None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    Objev_01 = Pop_weights[None,None,:,None] * (1.-Pop_fractions[None,None,:,None,0]) * Pop_fractions[None,None,:,None,1]       \
        * Pop_weights[None,None,None,:] * Pop_fractions[None,None,None,:,0] * Pop_fractions[None,None,None,:,2] / Total_binary_fraction    \
        * Double_objects_evidence[:, 1, None, None, None] * Double_objects_weights[:, 1, :, None, None]   \
        * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 1, :, 1, None, None]) * tf.pow( Double_objects_means[:, 1, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 1, :, 0, None, None]) * tf.pow( Double_objects_means[:, 1, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 1, :, 2, None, None] * ( Double_objects_means[:, 1, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 1, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 1, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 1, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 1, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 1, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 1, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 1, :, 2, None, None], 2)     ) )    \
        * tf.exp( -0.5*tf.pow((Double_objects_means[:, 1, :, 0, None, None]-Double_objects_means[:, 1, :, 1, None, None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    Objev_10 = Pop_weights[None,None,:,None] * Pop_fractions[None,None,:,None,0] * Pop_fractions[None,None,:,None,2]       \
        * Pop_weights[None,None,None,:] * (1.-Pop_fractions[None,None,None,:,0]) * Pop_fractions[None,None,None,:,1] / Total_binary_fraction    \
        * Double_objects_evidence[:, 2, None, None, None] * Double_objects_weights[:, 2, :, None, None]   \
        * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 2, :, 1, None, None]) * tf.pow( Double_objects_means[:, 2, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 2, :, 0, None, None]) * tf.pow( Double_objects_means[:, 2, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 2, :, 2, None, None] * ( Double_objects_means[:, 2, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 2, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 2, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 2, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 2, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 2, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 2, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 2, :, 2, None, None], 2)     ) )    \
        * tf.exp( -0.5*tf.pow((Double_objects_means[:, 2, :, 0, None, None]-Double_objects_means[:, 2, :, 1, None, None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    Objev_11 = Pop_weights[None,None,:,None] * Pop_fractions[None,None,:,None,0] * Pop_fractions[None,None,:,None,2]       \
        * Pop_weights[None,None,None,:] * Pop_fractions[None,None,None,:,0] * Pop_fractions[None,None,None,:,2] / Total_binary_fraction    \
        * Double_objects_evidence[:, 3, None, None, None] * Double_objects_weights[:, 3, :, None, None]   \
        * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 3, :, 1, None, None]) * tf.pow( Double_objects_means[:, 3, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 3, :, 0, None, None]) * tf.pow( Double_objects_means[:, 3, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 3, :, 2, None, None] * ( Double_objects_means[:, 3, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 3, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 3, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 3, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 3, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 3, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 3, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 3, :, 2, None, None], 2)     ) )    \
        * tf.exp( -0.5*tf.pow((Double_objects_means[:, 3, :, 0, None, None]-Double_objects_means[:, 3, :, 1, None, None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    # factor 2 for binaries comes from the mass multiplicity constraint (m1<m2)
    LogObjEvidences = tf.log(   obj_evidence_eps + tf.reduce_sum(Objev_0, [1,2]) + tf.reduce_sum(Objev_1, [1,2])   \
        + (tf.reduce_sum(Objev_00, [1,2,3]) + tf.reduce_sum(Objev_01, [1,2,3]) + tf.reduce_sum(Objev_10, [1,2,3]) + tf.reduce_sum(Objev_11, [1,2,3]))   )
else:
    LogObjEvidences = tf.log( obj_evidence_eps + tf.reduce_sum(Objev_0, [1,2]) + tf.reduce_sum(Objev_1, [1,2]) )




### CALCULATE NORM
# indices are: mass, pop gauss
Norm_0 = DNm_0[:, None] * Pop_weights[None, :] * ( 1.-Pop_fractions[None, :, 0] ) * ( 1.-Pop_fractions[None, :, 1] )       \
    * tf.exp( -tf.pow((Mass_0_vec[:,None]-Pop_means[None,:]),2)/(2.*Pop_covs[None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,:])
Norm_1 = DNm_1[:,None] * Pop_weights[None,:]* Pop_fractions[None, :, 0] * ( 1.-Pop_fractions[None, :, 2] )      \
    * tf.exp( -tf.pow((Mass_1_vec[:,None]-Pop_means[None,:]),2)/(2.*Pop_covs[None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,:])
if include_binaries:
    # indices are: mass 1, mass 2, pop gauss 1, pop gauss 2
    Norm_00 = DNm_00[:,:,None,None] * Pop_weights[None,None,:,None] * (1.-Pop_fractions[None,None,:,None,0]) * Pop_fractions[None,None,:,None,1]       \
        * (1.-Pop_fractions[None,None,None,:,0]) * Pop_fractions[None,None,None,:,1] * Pop_weights[None,None,None,:] / Total_binary_fraction       \
        * tf.exp( -tf.pow((Mass_0_vec[:,None,None,None]-Pop_means[None,None,:,None]),2)/(2.*Pop_covs[None,None,:,None]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,:,None])        \
        * tf.exp( -tf.pow((Mass_0_vec[None,:,None,None]-Pop_means[None,None,None,:]),2)/(2.*Pop_covs[None,None,None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,None,:])   \
        * tf.exp( -0.5*tf.pow((Mass_0_vec[:,None,None,None]-Mass_0_vec[None,:,None,None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    Norm_01 = DNm_01[:,:,None,None] * Pop_weights[None,None,:,None] * (1.-Pop_fractions[None,None,:,None,0]) * Pop_fractions[None,None,:,None,1]       \
        * Pop_fractions[None,None,None,:,0] * Pop_fractions[None,None,None,:,2] * Pop_weights[None,None,None,:] / Total_binary_fraction       \
        * tf.exp( -tf.pow((Mass_0_vec[:,None,None,None]-Pop_means[None,None,:,None]),2)/(2.*Pop_covs[None,None,:,None]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,:,None])        \
        * tf.exp( -tf.pow((Mass_1_vec[None,:,None,None]-Pop_means[None,None,None,:]),2)/(2.*Pop_covs[None,None,None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,None,:])   \
        * tf.exp( -0.5*tf.pow((Mass_0_vec[:,None,None,None]-Mass_1_vec[None,:,None,None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    Norm_11 = DNm_11[:,:,None,None] * Pop_weights[None,None,:,None] * Pop_fractions[None,None,:,None,0] * Pop_fractions[None,None,:,None,2]       \
        * Pop_fractions[None,None,None,:,0] * Pop_fractions[None,None,None,:,2] * Pop_weights[None,None,None,:] / Total_binary_fraction       \
        * tf.exp( -tf.pow((Mass_1_vec[:,None,None,None]-Pop_means[None,None,:,None]),2)/(2.*Pop_covs[None,None,:,None]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,:,None])        \
        * tf.exp( -tf.pow((Mass_1_vec[None,:,None,None]-Pop_means[None,None,None,:]),2)/(2.*Pop_covs[None,None,None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,None,:])   \
        * tf.exp( -0.5*tf.pow((Mass_1_vec[:,None,None,None]-Mass_1_vec[None,:,None,None])/0.2, 2) )/tf.sqrt(2.*pi*0.2**2.)
    LogNorm = tf.log( (tf.reduce_sum(Norm_0) + tf.reduce_sum(Norm_1))*mass_norm_stepsize    \
        + (tf.reduce_sum(Norm_00) + 2.*tf.reduce_sum(Norm_01) + tf.reduce_sum(Norm_11))*mass_norm_stepsize**2. )
else:
    LogNorm = tf.log( (tf.reduce_sum(Norm_0) + tf.reduce_sum(Norm_1))*mass_norm_stepsize )






LogLoss = - tf.pow((1.-tf.reduce_sum(Pop_weights_non_normalized))/(2.*0.05), 2)      \
    - tf.reduce_sum(tf.pow(Pop_covs/0.4**2., 6))

# Total posterior value is this!
LogPosterior = LogLoss + tf.reduce_sum(LogObjEvidences) - tf.cast(tf.shape(Single_objects_evidence)[0], tf.float32) * LogNorm

MinusLogPosterior = -LogPosterior












Optimizer = tf.train.AdamOptimizer(learning_rate=1e-1).minimize(MinusLogPosterior)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)

with tf.Session(config=session_conf) as sess:
    
    sess.run( tf.global_variables_initializer() )
    
    for i in range(10000):
        _, minusLogPosterior, pop_weights, pop_means, pop_covs, pop_fractions =  \
            sess.run([Optimizer, MinusLogPosterior, Pop_weights, Pop_means, Pop_covs, Pop_fractions],
                     feed_dict={
                             DNm_0: dN_0,
                             DNm_1: dN_1,
                             DNm_00: dN_00,
                             DNm_01: dN_01,
                             DNm_11: dN_11,
                             Mass_0_vec: mass_vec,
                             Mass_1_vec: mass_vec,
                             Single_objects_evidence: single_objects_evidence,
                             Single_objects_weights: single_objects_weights,
                             Single_objects_means: single_objects_means,
                             Single_objects_covs: single_objects_covs,
                             Double_objects_evidence: double_objects_evidence,
                             Double_objects_weights: double_objects_weights,
                             Double_objects_means: double_objects_means,
                             Double_objects_covs: double_objects_covs,
                     })
        if i%100==0:
            print(i)
            print(minusLogPosterior)
            print("pop_weights =", list(pop_weights))
            print("pop_means   =", list(pop_means))
            print("pop_covs    =", list(pop_covs))
            print("pop_fractions =", [list(pf) for pf in pop_fractions])
            print('\n\n\n')


















"""
class population_model():
    def __init__(self,appGlimit):
        npzfile = np.load('../Data/dN_0_G-'+str(int(appGlimit))+'.npz')
        self.mass_0_vec = npzfile['mass_vec']
        self.dN_0 = npzfile['dNdmass_vec']
        npzfile = np.load('../Data/dN_1_G-'+str(int(appGlimit))+'.npz')
        self.mass_1_vec = npzfile['mass_vec']
        self.dN_1 = npzfile['dNdmass_vec']
        
        npzfile = np.load('../Data/dN_0-0_G-'+str(int(appGlimit))+'.npz')
        self.dN_00 = npzfile['dNdmass_vec']
        npzfile = np.load('../Data/dN_0-1_G-'+str(int(appGlimit))+'.npz')
        self.dN_01 = npzfile['dNdmass_vec']
        npzfile = np.load('../Data/dN_1-1_G-'+str(int(appGlimit))+'.npz')
        self.dN_11 = npzfile['dNdmass_vec']
    
    def evidence_single(self,obj_evidence,obj_means,obj_weights,obj_covs,obj_fractions,pop_means,pop_weights,pop_covs):
        res = 0.
        for i_obj in range(len(obj_means)):
            for i_pop in range(len(pop_means)):
                weight = obj_evidence*obj_weights[i_obj]*obj_fractions[i_obj]*pop_weights[i_pop]
                mean = obj_means[i_obj]-pop_means[i_pop]
                cov = obj_covs[i_obs]+pop_covs[i_pop]
                res += weight*np.exp(-mean**2./(2.*cov))/np.sqrt(2.*pi*cov)
        return res
    
    def evidence_double(self,obj_evidence,obj_means,obj_weights,obj_covs,obj_fractions,pop_means,pop_weights,pop_covs):
        res = 0.
        for i_obj in range(len(obj_means)):
            for i_pop in range(len(pop_means)):
                for j_pop in range(len(pop_means)):
                    weight = obj_evidence*obj_weights[i_obj]*obj_fractions[i_obj]*pop_weights[i_pop]*pop_weights[j_pop]/np.sum(pop_weights)
                    mean = obj_means[i_obj]-[pop_means[i_pop],pop_means[j_pop]]
                    pop_cov = [pop_covs[i_pop],pop_covs[j_pop]]
                    cov = obj_covs[i_obs]+pop_cov
                    cov_inv = np.linalg.inv(cov)
                    cov_det = np.linalg.det(cov)
                    res += weight*np.exp(-1./2.*np.dot(np.dot(cov_inv,diff),diff))/np.sqrt((2.*pi)**dim*cov_det)
        return res
    
    def norm(self,means,covs,weights,hydrogen_fractions,binary_fractions):
        # placeholder
        return 1.
    
    
    def lnprob(self,means,covs,weights,hydrogen_fractions,binary_fractions):
        res_total = len(self.list_of_objects)*self.norm(means,covs,weights,hydrogen_fractions,binary_fractions)
        for obj in self.list_of_objects:
            obj_ev = 0.
            for WDtype in [0.,1.]:
                obj_ev = self.evidence_single()
            for WDtypes in [[0.,0.],[0.,1.],[1.,0.],[1.,1.]]:
                obj_ev += self.evidence_double()
            res_total += np.log( obj_ev )
#exit()
#"""