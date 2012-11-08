import numpy as np
cimport numpy as np
from time import time
from cpython cimport bool
import cython
from cython import Py_ssize_t
np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int weighted_randint_dicho(np.ndarray[np.float64_t, ndim=1] cum_weights,float x):
    cdef int L = cum_weights.shape[0]
    cdef int start = 0 
    cdef int mid= L
    cdef int stop=L
    while start < stop - 1:
        mid = (start + stop) / 2
        if cum_weights[mid] <= x:
            start = mid
        else:
            stop = mid
    return start



#@cython.boundscheck(False)
#@cython.wraparound(False)
def _sampleFromImg(np.ndarray[np.float64_t, ndim=2] image,int N,int pw=15):
    
    cdef int ih=image.shape[0]
    cdef int iw=image.shape[1]
    
    cdef np.ndarray[np.float64_t, ndim=1] cum_weights=np.cumsum(image.ravel())
    cdef np.ndarray[np.uint32_t, ndim=1]  ind=np.empty(N).astype(np.uint32)
    cdef np.ndarray[np.float64_t, ndim=1]  xs=np.random.rand(N)
    
    cdef int i
   
    for i in range(N):
        ind[i]=weighted_randint_dicho(cum_weights,xs[i]* cum_weights[-1])
    
    res=np.unravel_index(list(ind),(ih,iw))
    res=np.hstack([res[0].reshape(-1,1),res[1].reshape(-1,1)])
    
    cdef int count,x,y,dx,dy
    dx=pw/2
    dy=pw/2
    count=0
    for i in range(N):
       x=res[i,0]
       y=res[i,1]
       if not (x>dx and x<ih-dx and y>dy and y<ih-dy):
           res[i,0]=-1
           res[i,1]=-1
    
    res=np.take(res,np.where(res>0)[0],axis=0)
    return res


