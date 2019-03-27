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

numpopgaussians = 4
pop_weights = np.random.uniform(0.,1.,numpopgaussians)
pop_means   = np.random.uniform(0.4,1.,numpopgaussians)
pop_covs    = np.random.uniform(0.1**2.,0.2**2.,numpopgaussians)
pop_fractions = np.random.uniform(0.,0.5,(numpopgaussians,3))


#pop_weights = [0.109644674, 0.18276168, 0.05045538, 0.09125646, 0.5658817]
#pop_means   = [0.6795613, 0.8047465, 1.1459588, 0.42990243, 0.5665329]
#pop_covs    = [0.0013379187, 0.0078122653, 0.0122329965, 0.008229718, 0.001401991]
#pop_fractions = [[0.60139513, 0.0, 0.0], [0.06404678, 0.0, 0.0], [0.15761541, 0.0, 0.0], [0.122598596, 0.0, 0.0], [0.13031293, 0.0, 0.0]]


pop_weights   = [0.50928295, 0.08955507, 0.16754642, 0.2336155]
pop_means     = [0.5689328, 0.43965518, 0.75278515, 0.9961493]
pop_covs      = [0.0016807423, 0.009188197, 0.006768013, 0.019205853]
pop_fractions = [[0.078893304, 0.0478085, 2.3841858e-07], [0.18116482, 0.39472187, 0.9906285], [0.07137887, 5.066395e-06, 2.9861927e-05], [0.3614726, 0.55450857, 0.940295]]

pop_weights = np.array(pop_weights)
pop_means = np.array(pop_means)
pop_covs = np.array(pop_covs)
pop_fractions = np.array(pop_fractions)

#pop_fractions[:,1] = 5e-1
#pop_fractions[:,2] = 5e-1

include_binaries = True
massdiff = 0.2
agediff = 1e9

def sigmoid_inv(x):
    return -np.log(1./x-1.)

if include_binaries:
    npzfile = np.load('../Data/all_objects_G-19_'+str(massdiff)+'_'+str(int(agediff/1e6))+'Myr'+'.npz')
else:
    npzfile = np.load('../Data/all_objects_G-19_singles-only.npz')
single_objects_evidence      = npzfile['single_objects_evidence']
single_objects_weights       = npzfile['single_objects_weights']
single_objects_means         = npzfile['single_objects_means']
single_objects_covs          = npzfile['single_objects_covs']
if include_binaries:
    double_objects_evidence      = npzfile['double_objects_evidence']
    double_objects_weights       = npzfile['double_objects_weights']
    double_objects_means         = npzfile['double_objects_means']
    double_objects_covs          = npzfile['double_objects_covs']
else:
    double_objects_evidence      = np.zeros((1,4))
    double_objects_weights       = np.zeros((1,4,8))
    double_objects_means         = np.zeros((1,4,8,2))
    double_objects_covs          = np.zeros((1,4,8,3))

print( "objects loaded", len(single_objects_evidence) )
#exit()


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



def lookupNearest2d(x0, y0, matrix, vec1, vec2):
    xi = np.abs(vec1-x0).argmin()
    yi = np.abs(vec2-y0).argmin()
    return matrix[xi][yi]
npzfile = np.load('../Data/dN_0-0_G-'+str(int(appGlimit))+'.npz')
dN_00 = npzfile['dNdmass_matrix']
mass_0_vec = npzfile['mass_vec1']
mass_1_vec = npzfile['mass_vec2']
dN_00 = [[lookupNearest2d(mass1, mass2, dN_00, mass_0_vec, mass_0_vec) for mass2 in mass_vec] for mass1 in mass_vec]
npzfile = np.load('../Data/dN_0-1_G-'+str(int(appGlimit))+'.npz')
dN_01 = npzfile['dNdmass_matrix']
mass_0_vec = npzfile['mass_vec1']
mass_1_vec = npzfile['mass_vec2']
dN_01 = [[lookupNearest2d(mass1, mass2, dN_01, mass_0_vec, mass_1_vec) for mass2 in mass_vec] for mass1 in mass_vec]
npzfile = np.load('../Data/dN_1-1_G-'+str(int(appGlimit))+'.npz')
dN_11 = npzfile['dNdmass_matrix']
mass_0_vec = npzfile['mass_vec1']
mass_1_vec = npzfile['mass_vec2']
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
Mass_vec = tf.placeholder(shape=[None], dtype=tf.float32)
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



Hydrog_single_fractions = (1.-Pop_fractions[:, 0]) * (1.-Pop_fractions[:, 1])
Helium_single_fractions = Pop_fractions[:, 0] * (1.-Pop_fractions[:, 2])
Hydrog_binary_fractions = (1.-Pop_fractions[:, 0]) * Pop_fractions[:, 1]
Helium_binary_fractions = Pop_fractions[:, 0] * Pop_fractions[:, 2]

### CALCULATE OBJECT EVIDENCES
# indices are: object, obj gauss, pop gauss
Objev_0 = Pop_weights[None, None, :] * Hydrog_single_fractions[None, None, :]    \
    * Single_objects_evidence[:, 0, None, None] * Single_objects_weights[:, 0, :, None]   \
    * tf.exp( -tf.pow(Single_objects_means[:, 0, :, None]-Pop_means[None, None, :], 2) / (2.*( Pop_covs[None, None, :]+Single_objects_covs[:, 0, :, None] )) )  \
    / tf.sqrt(2.*pi*( Pop_covs[None, None, :]+Single_objects_covs[:, 0, :, None] ))

Objev_1 = Pop_weights[None, None, :] * Helium_single_fractions[None, None, :]    \
    * Single_objects_evidence[:, 1, None, None] * Single_objects_weights[:, 1, :, None]   \
    * tf.exp( -tf.pow(Single_objects_means[:, 1, :, None]-Pop_means[None, None, :], 2) / (2.*( Pop_covs[None, None, :]+Single_objects_covs[:, 1, :, None] )) )  \
    / tf.sqrt(2.*pi*( Pop_covs[None, None, :]+Single_objects_covs[:, 1, :, None] ))

if include_binaries:
    Binary_fraction_connections = tf.reduce_sum( ( Hydrog_binary_fractions[None, :] + Helium_binary_fractions[None, :] )   \
        * tf.exp( -tf.pow(Pop_means[:,None]-Pop_means[None,:], 2) / (2. * (Pop_covs[:,None] + Pop_covs[None,:] + massdiff**2.)) ) \
        / tf.sqrt(2.*pi*(Pop_covs[:,None]+Pop_covs[None,:]+massdiff**2.)), [1] )
    
    Binary_fraction_connections_INV = ( 1./Binary_fraction_connections[:, None] + 1./Binary_fraction_connections[None, :] )
    
    # indices are: object, obj gauss, pop gauss 1, pop gauss 2
    Pop_prefactor_00 = Pop_weights[:, None] * Hydrog_binary_fractions[:, None]       \
        * Pop_weights[None, :] * Hydrog_binary_fractions[None, :] * Binary_fraction_connections_INV
    Objev_00 = Pop_prefactor_00[None, None, :, :] * Double_objects_evidence[:, 0, None, None, None] * Double_objects_weights[:, 0, :, None, None] * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 0, :, 1, None, None]) * tf.pow( Double_objects_means[:, 0, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 0, :, 0, None, None]) * tf.pow( Double_objects_means[:, 0, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 0, :, 2, None, None] * ( Double_objects_means[:, 0, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 0, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 0, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 0, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 0, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 0, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 0, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 0, :, 2, None, None], 2)     ) )
        
    Pop_prefactor_01 = Pop_weights[:,None] * Hydrog_binary_fractions[:, None]       \
        * Pop_weights[None,:] * Helium_binary_fractions[None, :] * Binary_fraction_connections_INV
    Objev_01 = Pop_prefactor_01[None, None, :, :] * Double_objects_evidence[:, 1, None, None, None] * Double_objects_weights[:, 1, :, None, None] * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 1, :, 1, None, None]) * tf.pow( Double_objects_means[:, 1, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 1, :, 0, None, None]) * tf.pow( Double_objects_means[:, 1, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 1, :, 2, None, None] * ( Double_objects_means[:, 1, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 1, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 1, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 1, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 1, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 1, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 1, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 1, :, 2, None, None], 2)     ) )
    
    Pop_prefactor_10 = Pop_weights[:,None] * Helium_binary_fractions[:, None]       \
        * Pop_weights[None,:] * Hydrog_binary_fractions[None, :] * Binary_fraction_connections_INV
    Objev_10 = Pop_prefactor_10[None, None, :, :] * Double_objects_evidence[:, 2, None, None, None] * Double_objects_weights[:, 2, :, None, None] * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 2, :, 1, None, None]) * tf.pow( Double_objects_means[:, 2, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 2, :, 0, None, None]) * tf.pow( Double_objects_means[:, 2, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 2, :, 2, None, None] * ( Double_objects_means[:, 2, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 2, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 2, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 2, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 2, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 2, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 2, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 2, :, 2, None, None], 2)     ) )
    
    Pop_prefactor_11 = Pop_weights[:,None] * Helium_binary_fractions[:, None]       \
        * Pop_weights[None,:] * Helium_binary_fractions[None, :] * Binary_fraction_connections_INV
    Objev_11 = Pop_prefactor_11[None, None, :, :] * Double_objects_evidence[:, 3, None, None, None] * Double_objects_weights[:, 3, :, None, None] * tf.exp( - (     \
        (Pop_covs[None, None, None, :] + Double_objects_covs[:, 3, :, 1, None, None]) * tf.pow( Double_objects_means[:, 3, :, 0, None, None]-Pop_means[None, None, :, None], 2)    \
        + (Pop_covs[None, None, :, None] + Double_objects_covs[:, 3, :, 0, None, None]) * tf.pow( Double_objects_means[:, 3, :, 1, None, None]-Pop_means[None, None, None, :], 2)    \
        - 2. * Double_objects_covs[:, 3, :, 2, None, None] * ( Double_objects_means[:, 3, :, 0, None, None]-Pop_means[None, None, :, None] ) * ( Double_objects_means[:, 3, :, 1, None, None]-Pop_means[None, None, None, :] )   \
        ) / ( 2.*( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 3, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 3, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 3, :, 2, None, None], 2)      ))   \
        )       \
        / ( 2.*pi * tf.sqrt( (Pop_covs[None, None, None, :]+Double_objects_covs[:, 3, :, 1, None, None])*(Pop_covs[None, None, :, None] + Double_objects_covs[:, 3, :, 0, None, None])     \
        -tf.pow( Double_objects_covs[:, 3, :, 2, None, None], 2)     ) )
    
    # factor 2 for binaries comes from the mass multiplicity constraint (m1<m2)
    SinglesEvidence = tf.reduce_sum(Objev_0, [1,2]) + tf.reduce_sum(Objev_1, [1,2])
    DoublesEvidence = tf.reduce_sum(Objev_00, [1,2,3]) + tf.reduce_sum(Objev_01, [1,2,3]) + tf.reduce_sum(Objev_10, [1,2,3]) + tf.reduce_sum(Objev_11, [1,2,3])
    LogObjEvidences = tf.log(   obj_evidence_eps + SinglesEvidence + DoublesEvidence  )
else:
    LogObjEvidences = tf.log( obj_evidence_eps + tf.reduce_sum(Objev_0, [1,2]) + tf.reduce_sum(Objev_1, [1,2]) )




### CALCULATE NORM
# indices are: mass, pop gauss
Norm_0 = tf.reduce_sum(   DNm_0[:, None] * Pop_weights[None, :] * Hydrog_single_fractions[None, :]       \
    * tf.exp( -tf.pow(Mass_vec[:, None]-Pop_means[None,:], 2)/(2.*Pop_covs[None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,:])    )
Norm_1 = tf.reduce_sum(   DNm_1[:, None] * Pop_weights[None, :] * Helium_single_fractions[None, :]        \
    * tf.exp( -tf.pow(Mass_vec[:, None]-Pop_means[None,:], 2)/(2.*Pop_covs[None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,:])    )
if include_binaries:
    # indices are: mass 1, mass 2, pop gauss 1, pop gauss 2
    Norm_factor = tf.exp( -tf.pow((Mass_vec[:,None,None,None]-Pop_means[None,None,:,None]),2)/(2.*Pop_covs[None,None,:,None]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,:,None])        \
        * tf.exp( -tf.pow((Mass_vec[None,:,None,None]-Pop_means[None,None,None,:]),2)/(2.*Pop_covs[None,None,None,:]) ) / tf.sqrt(2.*pi*Pop_covs[None,None,None,:])   \
        * tf.exp( -1./2.*tf.pow((Mass_vec[:,None,None,None]-Mass_vec[None,:,None,None])/massdiff, 2) ) / np.sqrt(2.*pi*massdiff**2.)   \
        * Pop_weights[None, None, :, None] * Pop_weights[None, None, None, :]   \
        * Binary_fraction_connections_INV[None, None, :, :]
    Norm_00 = tf.reduce_sum(   DNm_00[:, :, None, None]       \
        * Hydrog_binary_fractions[None, None, :, None] * Hydrog_binary_fractions[None, None, None, :] \
        * Norm_factor           )
    Norm_01 = tf.reduce_sum(   DNm_01[:, :, None, None]       \
        * Hydrog_binary_fractions[None, None, :, None] * Helium_binary_fractions[None, None, None, :]     \
        * Norm_factor    )
    Norm_11 = tf.reduce_sum(   DNm_11[:, :, None, None]       \
        * Helium_binary_fractions[None, None, :, None] * Helium_binary_fractions[None, None, None, :]    \
        * Norm_factor   )
    LogNorm = tf.log( (Norm_0 + Norm_1)*mass_norm_stepsize    \
        + (Norm_00 + 2.*Norm_01 + Norm_11)*mass_norm_stepsize**2. )
else:
    LogNorm = tf.log( (Norm_0 + Norm_1)*mass_norm_stepsize )






LogLoss = - tf.pow((1.-tf.reduce_sum(Pop_weights_non_normalized))/(2.*0.05), 2)      \
    - tf.reduce_sum(tf.pow(0.01**2./Pop_covs, 6)) - tf.reduce_sum(tf.pow(Pop_covs/0.4**2., 6))

# Total posterior value is this!
LogPosterior = LogLoss + tf.reduce_sum(LogObjEvidences) - tf.cast(tf.shape(Single_objects_evidence)[0], tf.float32) * LogNorm

MinusLogPosterior = -LogPosterior












Optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(MinusLogPosterior)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=numTFThreads, inter_op_parallelism_threads=numTFThreads)

with tf.Session(config=session_conf) as sess:
    
    sess.run( tf.global_variables_initializer() )
    
    #"""
    singlesEvidence, doublesEvidence, norm_0, norm_1, norm_00, norm_01, norm_11 =  \
        sess.run([SinglesEvidence, DoublesEvidence, Norm_0, Norm_1, Norm_00, Norm_01, Norm_11],
                 feed_dict={
                         DNm_0: dN_0,
                         DNm_1: dN_1,
                         DNm_00: dN_00,
                         DNm_01: dN_01,
                         DNm_11: dN_11,
                         Mass_vec: mass_vec,
                         Single_objects_evidence: single_objects_evidence,
                         Single_objects_weights: single_objects_weights,
                         Single_objects_means: single_objects_means,
                         Single_objects_covs: single_objects_covs,
                         Double_objects_evidence: double_objects_evidence,
                         Double_objects_weights: double_objects_weights,
                         Double_objects_means: double_objects_means,
                         Double_objects_covs: double_objects_covs,
                 })
    
    print(norm_0,norm_1)
    print(mass_norm_stepsize*norm_00,2.*mass_norm_stepsize*norm_01,mass_norm_stepsize*norm_11)
    print(np.sum([norm_0,norm_1]),mass_norm_stepsize*np.sum( [norm_00,2.*norm_01,norm_11] ))
    print(np.sum( pop_weights/np.sum(pop_weights)*((1.-pop_fractions[:,0])*pop_fractions[:,1] + pop_fractions[:,0]*pop_fractions[:,2]) ))
    plt.scatter(np.log10(singlesEvidence), np.log10(doublesEvidence),s=1,alpha=0.5)
    plt.xlabel('log10( evidence for single WD)')
    plt.ylabel('log10( evidence for binary WD)')
    plt.plot([20,35],[20,35],'k--')
    plt.show()
    plt.scatter([np.sum([single_objects_weights[k][0][kk]*single_objects_means[k][0][kk] for kk in range(6)]) for k in range(len(single_objects_weights))],np.log10(singlesEvidence/doublesEvidence),s=1,alpha=0.5)
    plt.show()
    plt.plot(np.linspace(0.,100.,1001),np.percentile(-np.log10(singlesEvidence/doublesEvidence),np.linspace(0.,100.,1001)))
    plt.show()
    exit()
    #"""
    
    
    
    for i in range(10000):
        _, minusLogPosterior, pop_weights, pop_means, pop_covs, pop_fractions =  \
            sess.run([Optimizer, MinusLogPosterior, Pop_weights, Pop_means, Pop_covs, Pop_fractions],
                     feed_dict={
                             DNm_0: dN_0,
                             DNm_1: dN_1,
                             DNm_00: dN_00,
                             DNm_01: dN_01,
                             DNm_11: dN_11,
                             Mass_vec: mass_vec,
                             Single_objects_evidence: single_objects_evidence,
                             Single_objects_weights: single_objects_weights,
                             Single_objects_means: single_objects_means,
                             Single_objects_covs: single_objects_covs,
                             Double_objects_evidence: double_objects_evidence,
                             Double_objects_weights: double_objects_weights,
                             Double_objects_means: double_objects_means,
                             Double_objects_covs: double_objects_covs,
                     })
        if i%10==0:
            print(i)
            print(minusLogPosterior)
            print("pop_weights   =", list(pop_weights))
            print("pop_means     =", list(pop_means))
            print("pop_covs      =", list(pop_covs))
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