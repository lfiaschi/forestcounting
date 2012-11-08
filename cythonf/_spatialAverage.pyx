import numpy as np
cimport numpy as np
from time import time
from cpython cimport bool
import cython
from cython import Py_ssize_t
np.import_array()


cdef Py_ssize_t _getIndex(Py_ssize_t i,  Py_ssize_t j,Py_ssize_t pw) nogil:
    return j+pw*i

cdef Py_ssize_t _getIndex2(Py_ssize_t i,  Py_ssize_t j,Py_ssize_t m,Py_ssize_t pw,Py_ssize_t nc) nogil:
    return i*nc*pw+j*nc+m


#@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def _spatialAverage(int imh,int imw,np.ndarray[int, ndim=2] pos, np.ndarray[np.float64_t, ndim=2] pred,int pw):
    cdef np.ndarray[np.float64_t, ndim=2] res=np.zeros((imh,imw))
    cdef np.ndarray[np.float64_t, ndim=2] counter=np.zeros((imh,imw))
    cdef Py_ssize_t dx, dy,i,j,k,x,y
    dx=pw/2
    dy=pw/2
    k=0
    with nogil:
        for k in range(pos.shape[0]):
            x=pos[k,0]
            y=pos[k,1]
            
            for i from x-dx <= i< x+dx+1:
                for j from y-dy<= j <y+dy+1:
                    res[i,j]=res[i,j]+pred[k,_getIndex(i-x+dx,j-y+dy,pw)]
                                   
                    counter[i,j]=counter[i,j]+1
                    
    res=res/counter
    #print counter
    return res

def _e(np.ndarray[np.float64_t, ndim=2] im, int pw=15):
    return im

@cython.boundscheck(False)
@cython.wraparound(False)
def _extract_dense2D(np.ndarray[np.float64_t, ndim=2] im, int pw,int stride=1):
    #extract dense patches on the image
    
    cdef Py_ssize_t ih=im.shape[0]
    cdef Py_ssize_t iw=im.shape[1]
    
    cdef Py_ssize_t dx=pw/2
    cdef Py_ssize_t dy=pw/2
    cdef Py_ssize_t x,y,i,j,k
    
    cdef Py_ssize_t npatches=len(range(dy,iw-dy,stride))*len(range(dx,ih-dx,stride))
    
    cdef np.ndarray[int, ndim=2] pos=np.zeros((npatches,2)).astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] pmat=np.zeros((npatches,pw*pw))
    
    k=-1
    x=dx
    with nogil:
        while x<ih-dx:
            y=dy
            while y<iw-dy:
        #for x in range(dx,ih-dx,stride):
        #    for y in range(dy,iw-dy,stride):
                k+=1
                pos[k,0]=x
                pos[k,1]=y
                for i in range(x-dx,x+dx+1):
                    for j in range(y-dy,y+dy+1):
                        pmat[k,_getIndex(i-x+dx,j-y+dy,pw)]=im[i,j]
                y+=stride
            x+=stride
                        
    
    return pmat,pos       



@cython.boundscheck(False)
@cython.wraparound(False)
def _extract_dense3D(np.ndarray[np.float64_t, ndim=3] mim, int pw,int stride=1):
    #extract dense patches on the image
    
    cdef Py_ssize_t ih=mim.shape[0]
    cdef Py_ssize_t iw=mim.shape[1]
    cdef Py_ssize_t nc=mim.shape[2]
    
    
    cdef Py_ssize_t dx=pw/2
    cdef Py_ssize_t dy=pw/2
    cdef Py_ssize_t x,y,i,j,k,m
    
    cdef Py_ssize_t npatches=len(range(dy,iw-dy,stride))*len(range(dx,ih-dx,stride))
    
    cdef np.ndarray[int, ndim=2] pos=np.zeros((npatches,2)).astype(np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] pmat=np.zeros((npatches,pw*pw*nc))
    
    k=-1
    x=dx
    with nogil:
        while x<ih-dx:
            y=dy
            while y<iw-dy:
        #for x in range(dx,ih-dx,stride):
        #    for y in range(dy,iw-dy,stride):
                k+=1
                pos[k,0]=x
                pos[k,1]=y
                for i in range(x-dx,x+dx+1):
                    for j in range(y-dy,y+dy+1):
                        for m in range(nc):
                            pmat[k,_getIndex2(i-x+dx,j-y+dy,m,pw,nc)]=mim[i,j,m]
                y+=stride
            x+=stride
                            
    
    return pmat,pos 



def _extract_at_pos2D(np.ndarray[np.float64_t, ndim=2] img,np.ndarray[np.uint32_t, ndim=2] pos,pw=15):
    #extract dense patches on the image at determined pos
    # pos should be an array N*2, the positions should be inside
    #the image so that the patch does notgo outside
    cdef Py_ssize_t ih=img.shape[0]
    cdef Py_ssize_t iw=img.shape[1]
    
    cdef Py_ssize_t dx=pw/2
    cdef Py_ssize_t dy=pw/2

    cdef Py_ssize_t npatches=pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] pmat=np.zeros((npatches,pw*pw))
    
    
    cdef Py_ssize_t x,y,i,j,k
    
    for k in range(npatches):
        x=pos[k,0]
        y=pos[k,1]    
        
        for i in range(x-dx,x+dx+1):
            for j in range(y-dy,y+dy+1):
                pmat[k,_getIndex(i-x+dx,j-y+dy,pw)]=img[i,j]
        
    return pmat
    
    
def _extract_at_pos3D(np.ndarray[np.float64_t, ndim=3] mim,np.ndarray[np.uint32_t, ndim=2] pos,pw=15):
    #extract dense patches on the image at determined pos
    # pos should be an array N*2, the positions should be inside
    #the image so that the patch does notgo outside
    cdef Py_ssize_t x,y,i,j,k,nchannels
    
    cdef Py_ssize_t ih=mim.shape[0]
    cdef Py_ssize_t iw=mim.shape[1]
    cdef Py_ssize_t nc=mim.shape[2]
    
    
    cdef Py_ssize_t dx=pw/2
    cdef Py_ssize_t dy=pw/2

    cdef Py_ssize_t npatches=pos.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] pmat=np.zeros((npatches,pw*pw*nc))
    
    
    for k in range(npatches):
        x=pos[k,0]
        y=pos[k,1]    
        
        for i in range(x-dx,x+dx+1):
            for j in range(y-dy,y+dy+1):
                for m in range(nc):
                    pmat[k,_getIndex2(i-x+dx,j-y+dy,m,pw,nc)]=mim[i,j,m]
        
    return pmat        
