#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dist.h"

long double euclidean_dist_c(double * x, double * y, int ln){
	int i = 0;
	double diff;
	long double dist = 0.0;
	for(;i < ln; ++i){
		diff = x[i] - y[i];
		dist += pow(diff, 2);
	}
	return sqrt(dist);
}

int compare ( const void *pa, const void *pb ) {
    double *a = *(double **)pa;
    double *b = *(double **)pb;
	return a[0] - b[0];
}

void q_sort(double ** x, int n, int (*comparator)(const void*,const void*)){
	qsort(x, n, sizeof(x[0]), comparator);
}


