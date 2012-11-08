import numpy as np
cimport numpy as np
from time import time
from cpython cimport bool
import cython
from cython import Py_ssize_t
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int  _getImax(np.ndarray[np.uint32_t, ndim=1] h):# nogil:
    cdef int k
    cdef int imax=0
    
    cdef int maxv=h[0]
    for k in range(1,256):
        if h[k]>maxv:
            imax=k
            maxv=h[k]
    
    return imax
    
    



@cython.boundscheck(False)
@cython.wraparound(False)
def _fastMode(np.ndarray[np.uint8_t, ndim=3] image):
    cdef np.ndarray[np.uint32_t, ndim=1] h=np.zeros(256,dtype=np.uint32)
    
    cdef Py_ssize_t i,j,k,ih,iw,id,l
    
    ih=image.shape[0]
    iw=image.shape[1]
    id=image.shape[2]
    assert image.min()>=0 and image.max()<=255
    cdef np.ndarray[np.uint8_t, ndim=2] res=np.zeros((ih,iw),dtype=np.uint8)
    
    #with nogil:
    for j in range(iw):
        for i in range(ih):
            
            for l in range(256):
                h[l]=0 
            
            for k in range(id):
                h[image[i,j,k]]+=1
            res[i,j]=_getImax(h)
                
    return res

