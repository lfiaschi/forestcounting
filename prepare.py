#===============================================================================
# File used for the preprocessing of the demo features
# This file is meant for illustration purpose only, to actually run this code 
# requires additional libraries, in particular vigra
# https://github.com/ukoethe/vigra.git
#===============================================================================
import pickle

import vigra
import numpy as np
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)
from joblib import Parallel,delayed
from spatialAverage import *
from utils import *
from time import time


def formMultispectral(img):
    #Extract the features responses on the image
    img=img.copy()
    img=img.astype(np.float32)
    
    a=[]
    blue=img[:,:,2]
    
    a.append(blue)
    
    N=3
    base=0.8
    r2=2
    for n in range(N):
        f=base*(r2**n)
        gg= vigra.filters.gaussianGradientMagnitude(blue,f).view(np.ndarray)
        a.append(gg)
        te=vigra.filters.structureTensorEigenvalues(blue,f,f*2).view(np.ndarray)
        a.append(te)
        
        lg=vigra.filters.laplacianOfGaussian(blue,f).view(np.ndarray)
        a.append(lg)

    res=np.dstack(a)
    return res

if __name__=="__main__":
    #===========================================================================
    # Produce the multispectral images
    #===========================================================================
    
    stop=-1
    
    
    dataFolder='data/cells/'
    imgs=importImagesFolder(dataFolder,'*cell.png',stop=stop)
    densities=importImagesFolder(dataFolder,'*dots.png',stop=stop)
    
    
    #Parameter
    sigmadots=2.5
    
    
    
    origdots=[]
    for k,dot in enumerate(densities):
        dot=dot[:,:,0]
        
        #print dot.max(),dot.min()
        dot/=255
        origdots.append(dot.copy())
        dot=vigra.filters.gaussianSmoothing(dot, sigmadots).squeeze()
        densities[k]=dot.view(np.ndarray).astype(np.float32)
    
    mimgs=Parallel(8)(delayed(formMultispectral)(img) for img in imgs)
    
    pickle.dump(mimgs,open("data/cells-multi/multipsectral_images.pkl","wb"))
    pickle.dump(origdots,open("data/cells-multi/originaldottedimages.pkl","wb"))
    pickle.dump(imgs,open("data/cells-multi/images.pkl","wb"))
    pickle.dump(densities,open("data/cells-multi/density.pkl","wb"))
    
    
    print "done"
    