from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np

cdef extern from "dist.h":
	long double euclidean_dist_c(double * x, double * y, int ln)
	int compare ( const void *pa, const void *pb )
	void q_sort(double ** x, int n, int (*comparator)(const void*,const void*))

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
	
cpdef np.ndarray[double, ndim = 2] knn_search_pyx(np.ndarray data, np.ndarray new_point, int k, set cands):
	cdef int n = data.shape[0]
	cdef int c = data.shape[1]
	cdef int m = 2
	cdef int i, j
	cdef double ** x
	cdef double * row_arr
	cdef np.ndarray[double, ndim=1, mode="c"] x_cy
	cdef np.ndarray[double, ndim=1, mode="c"] y_cy = np.asarray(new_point, dtype = float, order="C")
	cpdef np.ndarray[double, ndim = 2] knn_indx = np.zeros(shape = (k, 2), dtype = float)

	x = <double**>malloc(n * sizeof(double *))
	row_arr = <double*>malloc(n * m * sizeof(x[0]))
	if not x or not row_arr: 
		raise MemoryError
	for i, v in enumerate(cands):
		x[i] = row_arr + (i * m)
		x_cy = np.asarray(data[i, :], dtype = float, order="C")
		x[i][0] = euclidean_dist_c(&x_cy[0], &y_cy[0], c)
		x[i][1] = <double>v
	
	q_sort(x, n, compare)
	
	for j in range(k):
		knn_indx[j, 0] = x[j][0]
		knn_indx[j, 1] = x[j][1]

	try:
		return knn_indx
	finally:
		free(x)
		free(row_arr)




