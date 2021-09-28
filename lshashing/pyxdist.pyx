#from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

cdef extern from "dist.h":
	long double euclidean_dist_c(double * x, double * y, int ln)

cpdef long double euclidean_dist_pyx(x, y):
	cdef int ln = x.shape[0]
	cdef int i
	cpdef long double dist
	#cdef np.ndarray[double, ndim=1, mode="c"] x_cy = np.asarray(x, dtype = float, order="C")
	#cdef np.ndarray[double, ndim=1, mode="c"] y_cy = np.asarray(y, dtype = float, order="C")
					 
	#cdef float * x_c = <float *>malloc(ln * sizeof(float))
	#cdef float * y_c = <float *>malloc(ln * sizeof(float))
	#if not x_c or not y_c: 
	#	raise MemoryError
		
	#for i in range(ln): 
	#	x_c[i] = x_cy[i]
	#	y_c[i] = y_cy[i]
	
	cdef double [::1] x_cy = np.ascontiguousarray(x, dtype = "double")
	cdef double [::1] y_cy = np.ascontiguousarray(y, dtype = "double")
		
	#try:
	dist = euclidean_dist_c(&x_cy[0], &y_cy[0], ln)
	return dist
	#finally:
	#	free(x_c)
	#	free(y_c)



