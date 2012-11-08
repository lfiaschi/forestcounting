#===========================================================================
# Dependency loading
#===========================================================================
import pickle
import numpy as np
import signal,sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
from spatialAverage import *
from utils import *
import time



try:
    from sklearn.ensemble import ExtraTreesRegressor
except Exception,e:
    print "This code depends on the excellent library sklearn for the random forest"
    print "please install it from https://github.com/scikit-learn/scikit-learn"

    sys.exit()

try:
    import pylab
except:
    print "pylab is needed only if you want to display the images of the results"

#===========================================================================
# Some helpers functions
#===========================================================================
def testOnImg(RF,mimgtest,gtdots,pw,id):
    fMatrixTest,pos=extract_dense(mimgtest,pw=pw)
    ih,iw=gtdots.shape
    fMatrixTest = fMatrixTest.astype(np.float64)
    
    pred=RF.predict(fMatrixTest)
    resImg=spatialAverage((ih,iw),pos,pred,pw=pw) #spatial average of the prediction
    

    npred=resImg.sum()
    ntrue=gtdots.sum()

    return ntrue,npred,resImg,gtdots

def displayFeatures(mimg,img,density,dot):

    pylab.figure(figsize=(12,12))
    pylab.subplot(3,3,1)
    pylab.imshow(img.astype(np.uint8))
    pylab.xticks([])
    pylab.yticks([])
    pylab.title("image")
    
    pylab.subplot(3,3,2)
    pylab.imshow(dot,interpolation=None)
    pylab.xticks([])
    pylab.yticks([])
    pylab.title("annotations N = %d" %np.sum(dot))
    
    pylab.subplot(3,3,3)
    pylab.imshow(density,cmap="jet")
    pylab.xticks([])
    pylab.yticks([])
    pylab.title("cell density N = %.2f" %np.sum(density))
    
    for k in range(1,7):
        pylab.subplot(3,3,3+k)
        pylab.imshow(mimg[:,:,k],cmap="jet")
        pylab.xticks([])
        pylab.yticks([])
        pylab.title("feature ch N = %d"%k)
    
    
    
    
    pylab.savefig("feature_rapresentation.pdf")
    pylab.show()


def showres(image,id, gtdens,truecount,resImg,predcount):
    
    pylab.subplot(1,3,1)
    pylab.imshow(image.astype(np.uint8))
    pylab.xticks([])
    pylab.yticks([])
    pylab.title("image %d"%id)
    
    
    vmax=0.1 #fix the max for renormalization cmap
    
    pylab.subplot(1,3,2)
    pylab.imshow(gtdens,vmin=0,vmax=vmax)
    pylab.xticks([])
    pylab.yticks([])
    pylab.title(" true density N = %.2f"%truecount)
    
    pylab.subplot(1,3,3)
    pylab.imshow(resImg,vmin=0,vmax=vmax)
    pylab.xticks([])
    pylab.yticks([])
    pylab.title("predicted N = %.2f"%predcount)


#===========================================================================
# START DEMO OF LEARNING TO COUNT WITH A REGRESSION FOREST
#===========================================================================
if __name__=="__main__":

    
    #Parameters
    pw=7 #patch with 
    Nr=500 #Number of patches extracted from the training images
    visualize=True #weather to visualize the ouput images
    
    
    #Load the data
    print "Loading the precomputed features"
    mimgs=pickle.load(open("data/cells-multi/multipsectral_images.pkl","r")) #Multispectral feature images
    
    origdots=pickle.load(open("data/cells-multi/originaldottedimages.pkl","r")) #original dotted image
    imgs=pickle.load(open("data/cells-multi/images.pkl","r")) #original images
    densities=pickle.load(open("data/cells-multi/density.pkl","r")) #Target densities 
    
    
    if 1: #Display the computed features
        try:
            displayFeatures(mimgs[0],imgs[0],densities[0],origdots[0])
        except Exception,e:
            pass
    

    
    # Use the first 4 images for training and the rest for testing
    
    trainImgs=mimgs[:4]
    trainDots=densities[:4]
    
    testImgs=mimgs[4:]
    testDots=densities[4:]
    
    
    
    
    #Extract Random patches densities from the training imgs
    positions=[]
    opatches=[]
    for k,dot in enumerate(trainDots):
        dot=dot.astype(np.float32)

        opatch,pos=extract_at_random(dot,pw=pw,N=Nr)
        opatches.append(opatch)
        positions.append(pos)
    opatches=np.vstack(opatches)
    
    fMatrix=[]
    for k,pos in enumerate(positions):
        mimg=trainImgs[k]
        pos=positions[k]
        
        fMatrix.append(extract_at_pos(mimg,pos,pw=pw))
    
    fMatrix=np.vstack(fMatrix)
    
    
    print  "The dimension of the feature matrix is ",fMatrix.shape
    print "The dimension of the target matrix is ",opatches.shape
    
    if 0:
        print "Training a Extra Tree Regressor for sklearn"
        s=time.time()
        RF=ExtraTreesRegressor(24, max_depth=10, min_samples_split=20, min_samples_leaf=10, max_features=pw*pw*3, bootstrap=True, n_jobs=-1,random_state=41)
        RF.fit(fMatrix, opatches)
        print "done %.2f sec"%( time.time()-s)
        pickle.dump(RF, open("data/serielized_rf.pkl","wb"))
    else:
        RF=pickle.load(open("data/serielized_rf.pkl","r"))
    
    RF.n_jobs=1 #Strange but if parallel is slower
    
    
    
    
    
    
    if visualize:
        try: #for visualization
            pylab.ion()
            pylab.figure(figsize=(12,6))
        except:
            pass
    
    ntrueall=[]
    npredall=[]
    
    print 
    print "Start prediction ..."
    for id,none in enumerate(testImgs):
        
        s=time.time()
        ntrue,npred,resImg,gtdots=testOnImg(RF,testImgs[id],testDots[id],pw,id)
        print "image : %d , ntrue = %.2f ,npred = %.2f , time =%.2f sec"%(id,ntrue,npred,time.time()-s)
    
        ntrueall.append(ntrue)
        npredall.append(npred)
        if visualize:
            try:
                
                showres(imgs[id+4],id,gtdots,ntrue,resImg,npred)
                pylab.draw()
            except:
                pass
        
    ntrueall=np.asarray(ntrueall)
    npredall=np.asarray(npredall)
    print "done ! mean absolute error %.2f"%np.mean(np.abs(ntrueall-npredall))
    
