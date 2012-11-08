import os 
import glob
import vigra
import numpy as np

def importImagesFolder(datafolder,pattern,skip=1,stop=-1,verbose=True):
    '''import all images from a folder that follow a certain pattern'''
    
    p=os.path.join(datafolder,pattern)
    names=sorted(glob.glob(p))
    
    count=0
    imgs=[]
    for name in names[::skip]:
        if verbose: print name
        img=vigra.impex.readImage(name).view(np.ndarray).swapaxes(0,1).squeeze()
        imgs.append(img)
        count+=1
        if count>=stop and stop!=-1:
            break
    
    return imgs


def shuffleWithIndex(listv,seed=None):
    #Shuffle a list and return the indexes
    if seed!=None: np.random.seed(seed)
    listvp=np.asarray(listv,dtype=object)
    ind=np.arange(len(listv))
    ind=np.random.permutation(ind)
    listvp=listvp[ind]
    listvp=list(listvp)
    return listvp,ind
    

def takeIndexFromList(listv,ind):
    listvp=np.asarray(listv,dtype=object)
    return list(listvp[ind])


def shuffleRows(array):
    ind=np.arange(array.shape[0])
    np.random.shuffle(ind)
    array=np.take(array,ind,axis=0)
    return array,ind
    
def generateRandomOdd(pwbase,treeCount):
    #Generate random odd numbers in the interavel [0,pwbase]
    res=[]
    count=0
    while count<treeCount:
        ext=np.random.randint(0,pwbase,1)
        if np.mod(ext,2)==1:
            res.append(ext)
            count+=1
    
    return res