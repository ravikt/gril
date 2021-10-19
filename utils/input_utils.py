import os, re, threading, time
import numpy as np
from IPython import embed
from scipy import misc

def preprocess_gaze_heatmap(GHmap, sigmaH, sigmaW, bg_prob_density, debug_plot_result=False):
    from scipy.stats import multivariate_normal
    import tensorflow as tf, keras as K # don't move this to the top, as people who import this file might not have keras or tf

    model = K.models.Sequential()

    model.add(K.layers.Lambda(lambda x: x+bg_prob_density, input_shape=(GHmap.shape[1],GHmap.shape[2],1)))

    if sigmaH > 0.25 and sigmaW > 0.25: # was 0,0; if too small don't blur #TODO
        lh, lw = int(4*sigmaH), int(4*sigmaW)
        x, y = np.mgrid[-lh:lh+1:1, -lw:lw+1:1] # so the kernel size is [lh*2+1,lw*2+1]
        pos = np.dstack((x, y))
        gkernel=multivariate_normal.pdf(pos,mean=[0,0],cov=[[sigmaH*sigmaH,0],[0,sigmaW*sigmaW]])
        assert gkernel.sum() > 0.95, "Simple sanity check: prob density should add up to nearly 1.0"

        model.add(K.layers.Lambda(lambda x: tf.pad(x,[(0,0),(lh,lh),(lw,lw),(0,0)],'REFLECT')))
        model.add(K.layers.Conv2D(1, kernel_size=gkernel.shape, strides=1, padding="valid", use_bias=False,
              activation="linear", kernel_initializer=K.initializers.Constant(gkernel)))
    else:
        print ("WARNING: Gaussian filter's sigma is 0, i.e. no blur.")
    # The following normalization hurts accuracy. I don't know why. But intuitively it should increase accuracy
    #def GH_normalization(x):
    #    sum_per_GH = tf.reduce_sum(x,axis=[1,2,3])
    #    sum_per_GH_correct_shape = tf.reshape(sum_per_GH, [tf.shape(sum_per_GH)[0],1,1,1])
    #    # normalize values to range [0,1], on a per heap-map basis
    #    x = x/sum_per_GH_correct_shape
    #    return x
    #model.add(K.layers.Lambda(lambda x: GH_normalization(x)))
    
    model.compile(optimizer='rmsprop', # not used
          loss='categorical_crossentropy', # not used
          metrics=None)
    output=model.predict(GHmap, batch_size=500)

    if debug_plot_result:
        print (r"""debug_plot_result is True. Entering IPython console. You can run:
                %matplotlib
                import matplotlib.pyplot as plt
                f, axarr = plt.subplots(1,2)
                axarr[0].imshow(gkernel)
                rnd=np.random.randint(output.shape[0]); print "rand idx:", rnd
                axarr[1].imshow(output[rnd,...,0])""")
        embed()
    
    shape_before, shape_after = GHmap.shape, output.shape
    assert shape_before == shape_after, """
    Simple sanity check: shape changed after preprocessing. 
    Your preprocessing code might be wrong. Check the shape of output tensor of your tensorflow code above"""
    return output