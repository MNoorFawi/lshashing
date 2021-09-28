#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dist.h"

long double euclidean_dist_c(double * x, double * y, int ln){
	int i = 0;
	long double sum = 0.0;
	for(;i < ln; ++i){
		sum += x[i] - y[i];
	}
	long double dist = sqrt(pow(sum, 2));
	return dist;
}

