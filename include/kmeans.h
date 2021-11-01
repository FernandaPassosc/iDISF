/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans.h   (an OpenMP version)                            */
/*   Description:  header file for a simple k-means clustering program       */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department Northwestern University                         */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

int call_kmeans_elbow(float **objects,   /* in: [numObjs][numCoords] */
                 int numObjs,       /* no. objects */
                 int numCoords,     /* no. coordinates */
                 int *membership    /* out: best clustering */);

int call_kmeans_silhouette(float **objects,   /* in: [numObjs][numCoords] */
                 int numObjs,       /* no. objects */
                 int numCoords,     /* no. coordinates */
                 int *membership    /* out: best clustering */);

float omp_kmeans_elbow(int, float**, int, int, int, float, int*, float**);
void omp_kmeans_silhouette(int, float**, int, int, int, float, int*, float**);

double  wtime(void);

extern int _debug;

#endif
