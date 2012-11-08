import numpy as np
#from tools import showCC
from cythonf import _spatialAverage, _extract_dense2D
from cythonf import _e, _extract_dense3D, _extract_at_pos2D, _extract_at_pos3D
from utils import shuffleRows

def extract_stats(patch):
    nchannels=patch.shape[2]
    r=[]
    for i in range(nchannels):
        m=np.mean(patch[:,:,i])
        s=np.std(patch[:,:,i])
        
        r.append(m)
        r.append(s)
    
    r=np.hstack(r)
    r.reshape(1,-1)
    return r


def extract_at_pos_average(img,pos,pw=15):
    #extract dense patches on the image at determined pos
    # pos should be an array N*2, the positions should be inside
    #the image so that the patch does notgo outside
    im=img.view(np.ndarray)
    ih=im.shape[0]
    iw=im.shape[1]
    
    if img.ndim==3:
        nchannels=im.shape[2]
    else:
        nchannels=1
    
    plist=[]
    
    dx=dy=pw/2
    for el in pos:
        x,y=el    
        
        sx=slice(x-dx,x+dx+1,None)
        sy=slice(y-dy,y+dy+1,None)
        
        t=im[sx,sy,...]
        tt=extract_stats(t)
        plist.append(tt)
            
            
    plist=np.vstack(plist)
    return plist



def spatialAverage(imshape,pos,pred,pw):
    h,w=imshape
    pos=np.vstack(pos)
    
    return _spatialAverage(h,w,pos.astype(np.int32),pred,int(pw))




def extract_at_pos(img,pos,pw=15):
    img=img.view(np.ndarray)
    pos=pos.astype(np.uint32)

    if img.ndim==2:
        return _extract_at_pos2D(img.astype(np.float64),pos,pw)
    else:
        return _extract_at_pos3D(img.astype(np.float64),pos,pw)
    
    
    
def extract_dense_average(img,pw=15,stride=1):
    #extract dense patches on the image
    
    im=img.view(np.ndarray)
    ih=im.shape[0]
    iw=im.shape[1]
    
    dx=dy=pw/2
    
    if img.ndim==3:
        nchannels=im.shape[2]
    else:
        nchannels=1
    
    plist=[]
    pos=[]
    for y in range(dy,iw-dy)[::stride]:
        for x in range(dx,ih-dx)[::stride]:
            
            
            
            sx=slice(x-dx,x+dx+1,None)
            sy=slice(y-dy,y+dy+1,None)
            
            t=im[sx,sy,...]
            tt=extract_stats(t)

            #print t.shape,'hehe'
            #pylab.imshow(t)
            #pylab.show()
            
            #t=t.reshape(-1,2*nchannels)
            
           
            plist.append(tt)
            pos.append([x,y])
            
            
    plist=np.vstack(plist)
    pos=np.vstack(pos)
    return plist,pos


def extract_dense(img,pw=15,stride=1):
    #extract dense patches on the image
    if img.ndim==2:
        return _extract_dense2D(img.astype(np.float64),pw,stride)
    else:
        return _extract_dense3D(img.astype(np.float64),pw,stride)



def extract_at_random(img,pw=15,N=100):
    #extract dense patches on the image at random positions
    # etractd dense patches on the image
    im=img.view(np.ndarray)
    ih=im.shape[0]
    iw=im.shape[1]
    
    dx=dy=pw/2
    
    
    if img.ndim==3:
        nchannels=im.shape[2]
    else:
        nchannels=1
    
    y=np.random.randint(dy,iw-dy,N).reshape(N,1)
    x=np.random.randint(dx,ih-dx,N).reshape(N,1)
    
    pos=np.hstack((x,y))
    
    
    plist=[]
    for el in pos:
        x,y=el
        sx=slice(x-dx,x+dx+1,None)
        sy=slice(y-dy,y+dy+1,None)
        
        t=im[sx,sy,...]
        
        t=t.reshape(-1,pw*pw)
        plist.append(t)
        
            
    plist=np.vstack(plist)
    return plist,pos

def extract_at_random_average(img,pw=15,N=100):
    #extract dense patches on the image at random positions
    # 
    im=img.view(np.ndarray)
    ih=im.shape[0]
    iw=im.shape[1]
    
    dx=dy=pw/2
    
    
    if img.ndim==3:
        nchannels=im.shape[2]
    else:
        nchannels=1
    
    y=np.random.randint(dy,iw-dy,N).reshape(N,1)
    x=np.random.randint(dx,ih-dx,N).reshape(N,1)
    
    pos=np.hstack((x,y))
    
    
    plist=[]
    for el in pos:
        x,y=el
        sx=slice(x-dx,x+dx+1,None)
        sy=slice(y-dy,y+dy+1,None)
        
        t=im[sx,sy,...]
        tt=extract_stats(t)
        #t=t.reshape(-1,pw*pw)
        plist.append(tt)
        
            
    plist=np.vstack(plist)
    return plist,pos


    




if __name__=="__main__":
    #Small test
    import pylab
    
    pred=np.ones((4,5*5))
    
    pred[1,:]=2
    pred[2,:]=3
    pred[3,:]=4
    
    
#    print pred.reshape((10,10))
#    print 
#    pos=[(4,4),(2,7),(7,2),(7,7)]
#    
#    res=spatialAverage((10,10),pos,pred,pw=5)
#    print res
#    showCC(res)
    #pylab.show()
    
    
    
    pos=[(2,2),(2,7),(7,2),(7,7)]
    orig=spatialAverageOld((10,10),pos,pred,pw=5)
    
    #pylab.subplot(1,2,1)
    #pylab.title("original")
    #showCC(orig)
    
    print "original"
    print orig
 
    
    pred,pos=extract_dense(orig,pw=5)
    res=spatialAverage((10,10),pos,pred,pw=5)
   
    
    print "reconstructed"
    print res
    
    

    