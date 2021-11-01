
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         kmeans_clustering.c  (OpenMP version)                     */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <omp.h>
#include "kmeans.h"

/* ************************
* KMEANS WITH ELBOW METHOD
************************** */

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster_elbow(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters,
                         float *distance)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    (*distance) = min_dist;
    return(index);
}

/* Use the elbow method in the sumSquaredDistances vector 
to choice the best k value, k in [minClusters, maxClusters].
Return the best k value. */
/*
Calculate the Within-Cluster-Sum of Squared Errors (WSS) for different values of k, 
and choose the k for which WSS becomes first starts to diminish. 
In the plot of WSS-versus-k, this is visible as an elbow.
*/
int optimal_clusters_elbow(float *sumSquaredDistances, /* vector of the sum squared distances [lenDstances] */
                                int lenDistances,
                                int minClusters, int maxClusters)
{
    float x0, y0, xi, yi, dy, dx, maxDistance;
    float denominator;
    int index;

    /* Min and max points */
    x0 = (float)minClusters;        
    xi = (float)maxClusters;
    y0 = sumSquaredDistances[0];    
    yi = sumSquaredDistances[lenDistances-1];
    dx = xi - x0;                   
    dy = yi - y0; 

    denominator = sqrtf(dy*dy - dx*dx);
    index = 0;
    maxDistance = 0.0;

    /* elbow method to find the best k */
    for (int i=0; i<lenDistances; i++)
    {
        float numerator, x1, y1;

        x1 = i + x0;
        y1 = sumSquaredDistances[i];
        
        numerator = abs(dy*x1 - dx*y1 + x0*yi + y0*xi);
        if((numerator/denominator) > maxDistance){
            index = i;
            maxDistance = numerator/denominator;
        }
    }
    index+=minClusters;
    //return index;
    return 3;
}


/* Call k-means with k = [2, numObjs] clusters and
use the elbow method to choice the best k value. 
Return a membership vector [numObjs]. */
int call_kmeans_elbow(float **objects,   /* in: [numObjs][numCoords] */
                 int numObjs,       /* no. objects */
                 int numCoords,     /* no. coordinates */
                 int *membership    /* out: best clustering */)
{
    float threshold = 0.001;
    float **clusters;      /* [numClusters][numCoords] cluster center */
    int numClusters, maxClusters, minClusters;
    float *sumSquaredDistances; /* vector used to elbow method [maxClusters-minClusters] */
    int lenDistances;

    // n_eigenvalues = numCoords -1
    // n_features (numCoords here) = n_eigenvalues -1
    minClusters = 2;
    if(numObjs < numCoords) maxClusters = numObjs-1;
    else maxClusters = numCoords;
    /*
    if(numObjs < 10) maxClusters = numObjs;
    else maxClusters = 10;
    numClusters = minClusters;
    */

    lenDistances = maxClusters-minClusters+1;
    sumSquaredDistances = (float*) malloc(lenDistances * sizeof(float));
    assert(sumSquaredDistances != NULL);
    
    /* allocate a 2D space for clusters[] (coordinates of cluster centers)
       this array should be the same across all processes                  */
    clusters    = (float**) malloc(maxClusters *             sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(maxClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (int i=1; i<maxClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    /* start the core computation -------------------------------------------*/
    /* membership: the cluster id for each data object */
    //membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);
    
    printf("\n\n");
    /* run k-means to each possible no. of clusters and store the sumSquaredDistances */
    for (numClusters=minClusters; numClusters <= maxClusters; numClusters++)
    {
        int step = (int)(numObjs/numClusters);
        /* copy the first numClusters elements in feature[] */
        for (int i=0; i<numClusters; i++){
            for (int j=0; j<numCoords; j++){
                clusters[i][j] = objects[i*step][j];
            }
        }
        
        //sumSquaredDistances[numClusters-minClusters] = omp_kmeans(1, objects, numCoords, numObjs,
        //       numClusters, threshold, membership, clusters);
        sumSquaredDistances[numClusters-minClusters] = omp_kmeans_elbow(1, objects, numClusters, numObjs,
               numClusters, threshold, membership, clusters); 
        printf("distance k=%d : %f\n", numClusters, sumSquaredDistances[numClusters-minClusters]);
    }
    
    numClusters = optimal_clusters_elbow(sumSquaredDistances, lenDistances, minClusters, maxClusters);

    /* copy the first numClusters elements in feature[] */
    int step = (int)(numObjs/numClusters);
    /* copy the first numClusters elements in feature[] */
    for (int i=0; i<numClusters; i++){
        for (int j=0; j<numCoords; j++){
            clusters[i][j] = objects[i*step][j];
        }
    }

    //sumSquaredDistances[0] = omp_kmeans(1, objects, numCoords, numObjs,
    //        numClusters, threshold, membership, clusters);
    sumSquaredDistances[0] = omp_kmeans_elbow(1, objects, numClusters, numObjs,
            numClusters, threshold, membership, clusters);
    
    free(clusters[0]);
    free(clusters);
    free(sumSquaredDistances);

    return numClusters;
}


/*----< kmeans_clustering() >------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
float omp_kmeans_elbow(int     is_perform_atomic, /* in: */
               float **objects,           /* in: [numObjs][numCoords] */
               int     numCoords,         /* no. coordinates */
               int     numObjs,           /* no. objects */
               int     numClusters,       /* no. clusters */
               float   threshold,         /* % objects change membership */
               int    *membership,        /* out: [numObjs] */
               float **clusters)          /* out: [numClusters][numCoords] */
{
    int      i, j, k, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **newClusters;    /* [numClusters][numCoords] */
    //double   timing;

    int      nthreads;             /* no. threads */
    int    **local_newClusterSize; /* [nthreads][numClusters] */
    float ***local_newClusters;    /* [nthreads][numClusters][numCoords] */
    float *squared_distances, sum_squared_distances;

    nthreads = omp_get_max_threads();
    sum_squared_distances = 0.0;
    
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);

    squared_distances = (float*)  calloc(numObjs, sizeof(float));
    
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    if (!is_perform_atomic) {
        /* each thread calculates new centers using a private space,
           then thread 0 does an array reduction on them. This approach
           should be faster */
        local_newClusterSize    = (int**) malloc(nthreads * sizeof(int*));
        assert(local_newClusterSize != NULL);
        local_newClusterSize[0] = (int*)  calloc(nthreads*numClusters,
                                                 sizeof(int));
        assert(local_newClusterSize[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;

        /* local_newClusters is a 3D array */
        local_newClusters    =(float***)malloc(nthreads * sizeof(float**));
        assert(local_newClusters != NULL);
        local_newClusters[0] =(float**) malloc(nthreads * numClusters *
                                               sizeof(float*));
        assert(local_newClusters[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusters[i] = local_newClusters[i-1] + numClusters;
        for (i=0; i<nthreads; i++) {
            for (j=0; j<numClusters; j++) {
                local_newClusters[i][j] = (float*)calloc(numCoords,
                                                         sizeof(float));
                assert(local_newClusters[i][j] != NULL);
            }
        }
    }

    //if (_debug) timing = omp_get_wtime();
    do {
        delta = 0.0;

        if (is_perform_atomic) {
            #pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(numObjs,numClusters,numCoords) \
                    shared(objects,clusters,membership,newClusters,newClusterSize,squared_distances) \
                    schedule(static) \
                    reduction(+:delta)
            for (i=0; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster_elbow(numClusters, numCoords, objects[i],
                                             clusters, &(squared_distances[i]));

                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
                #pragma omp atomic
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++)
                    #pragma omp atomic
                    newClusters[index][j] += objects[i][j];
            }
        }
        else {
            #pragma omp parallel \
                    shared(objects,clusters,membership,local_newClusters,local_newClusterSize, squared_distances)
            {
                int tid = omp_get_thread_num();
                #pragma omp for \
                            private(i,j,index) \
                            firstprivate(numObjs,numClusters,numCoords) \
                            schedule(static) \
                            reduction(+:delta)
                for (i=0; i<numObjs; i++) {
                    /* find the array index of nestest cluster center */
                    index = find_nearest_cluster_elbow(numClusters, numCoords,
                                                 objects[i], clusters, &(squared_distances[i]));

                    /* if membership changes, increase delta by 1 */
                    if (membership[i] != index) delta += 1.0;

                    /* assign the membership to object i */
                    membership[i] = index;

                    /* update new cluster centers : sum of all objects located
                       within (average will be performed later) */
                    local_newClusterSize[tid][index]++;
                    for (j=0; j<numCoords; j++)
                        local_newClusters[tid][index][j] += objects[i][j];
                }
            } /* end of #pragma omp parallel */

            /* let the main thread perform the array reduction */
            for (i=0; i<numClusters; i++) {
                for (j=0; j<nthreads; j++) {
                    newClusterSize[i] += local_newClusterSize[j][i];
                    local_newClusterSize[j][i] = 0.0;
                    for (k=0; k<numCoords; k++) {
                        newClusters[i][k] += local_newClusters[j][i][k];
                        local_newClusters[j][i][k] = 0.0;
                    }
                }
            }
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
            
        delta /= numObjs;
    //} while (delta > threshold && loop++ < 500);
    } while (delta > 0 && loop++ < 500);

    /*
    if (_debug) {
        timing = omp_get_wtime() - timing;
        printf("nloops = %2d (T = %7.4f)",loop,timing);
    }
    */

    for (i=0; i<numObjs; i++)
        sum_squared_distances+=squared_distances[i];

    free(squared_distances);
    if (!is_perform_atomic) {
        free(local_newClusterSize[0]);
        free(local_newClusterSize);

        for (i=0; i<nthreads; i++)
            for (j=0; j<numClusters; j++)
                free(local_newClusters[i][j]);
        free(local_newClusters[0]);
        free(local_newClusters);
    }
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return sum_squared_distances;
}


/* ****************************
* KMEANS WITH SOLHOUETTE METHOD
****************************** */

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}


/*----< kmeans_clustering() >------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
void omp_kmeans_silhouette(int     is_perform_atomic, /* in: */
               float **objects,           /* in: [numObjs][numCoords] */
               int     numCoords,         /* no. coordinates */
               int     numObjs,           /* no. objects */
               int     numClusters,       /* no. clusters */
               float   threshold,         /* % objects change membership */
               int    *membership,        /* out: [numObjs] */
               float **clusters)          /* out: [numClusters][numCoords] */
{
    int      i, j, k, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  **newClusters;    /* [numClusters][numCoords] */
    //double   timing;

    int      nthreads;             /* no. threads */
    int    **local_newClusterSize; /* [nthreads][numClusters] */
    float ***local_newClusters;    /* [nthreads][numClusters][numCoords] */

    nthreads = omp_get_max_threads();
    
    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *            sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;

    if (!is_perform_atomic) {
        /* each thread calculates new centers using a private space,
           then thread 0 does an array reduction on them. This approach
           should be faster */
        local_newClusterSize    = (int**) malloc(nthreads * sizeof(int*));
        assert(local_newClusterSize != NULL);
        local_newClusterSize[0] = (int*)  calloc(nthreads*numClusters,
                                                 sizeof(int));
        assert(local_newClusterSize[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusterSize[i] = local_newClusterSize[i-1]+numClusters;

        /* local_newClusters is a 3D array */
        local_newClusters    =(float***)malloc(nthreads * sizeof(float**));
        assert(local_newClusters != NULL);
        local_newClusters[0] =(float**) malloc(nthreads * numClusters *
                                               sizeof(float*));
        assert(local_newClusters[0] != NULL);
        for (i=1; i<nthreads; i++)
            local_newClusters[i] = local_newClusters[i-1] + numClusters;
        for (i=0; i<nthreads; i++) {
            for (j=0; j<numClusters; j++) {
                local_newClusters[i][j] = (float*)calloc(numCoords,
                                                         sizeof(float));
                assert(local_newClusters[i][j] != NULL);
            }
        }
    }

    //if (_debug) timing = omp_get_wtime();
    do {
        delta = 0.0;

        if (is_perform_atomic) {
            #pragma omp parallel for \
                    private(i,j,index) \
                    firstprivate(numObjs,numClusters,numCoords) \
                    shared(objects,clusters,membership,newClusters,newClusterSize) \
                    schedule(static) \
                    reduction(+:delta)
            for (i=0; i<numObjs; i++) {
                /* find the array index of nestest cluster center */
                index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                             clusters);

                /* if membership changes, increase delta by 1 */
                if (membership[i] != index) delta += 1.0;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of objects located within */
                #pragma omp atomic
                newClusterSize[index]++;
                for (j=0; j<numCoords; j++)
                    #pragma omp atomic
                    newClusters[index][j] += objects[i][j];
            }
        }
        else {
            #pragma omp parallel \
                    shared(objects,clusters,membership,local_newClusters,local_newClusterSize)
            {
                int tid = omp_get_thread_num();
                #pragma omp for \
                            private(i,j,index) \
                            firstprivate(numObjs,numClusters,numCoords) \
                            schedule(static) \
                            reduction(+:delta)
                for (i=0; i<numObjs; i++) {
                    /* find the array index of nestest cluster center */
                    index = find_nearest_cluster(numClusters, numCoords,
                                                 objects[i], clusters);

                    /* if membership changes, increase delta by 1 */
                    if (membership[i] != index) delta += 1.0;

                    /* assign the membership to object i */
                    membership[i] = index;

                    /* update new cluster centers : sum of all objects located
                       within (average will be performed later) */
                    local_newClusterSize[tid][index]++;
                    for (j=0; j<numCoords; j++)
                        local_newClusters[tid][index][j] += objects[i][j];
                }
            } /* end of #pragma omp parallel */

            /* let the main thread perform the array reduction */
            for (i=0; i<numClusters; i++) {
                for (j=0; j<nthreads; j++) {
                    newClusterSize[i] += local_newClusterSize[j][i];
                    local_newClusterSize[j][i] = 0.0;
                    for (k=0; k<numCoords; k++) {
                        newClusters[i][k] += local_newClusters[j][i][k];
                        local_newClusters[j][i][k] = 0.0;
                    }
                }
            }
        }

        /* average the sum and replace old cluster centers with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 1)
                    clusters[i][j] = newClusters[i][j] / newClusterSize[i];
                newClusters[i][j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        }
            
        delta /= numObjs;
    } while (delta > threshold && loop++ < 300);
    //} while (delta > 0 && loop++ < 500);
    //printf("loop=%d, delta=%f \n", loop, threshold);

    /*
    if (_debug) {
        timing = omp_get_wtime() - timing;
        printf("nloops = %2d (T = %7.4f)",loop,timing);
    }
    */

    if (!is_perform_atomic) {
        free(local_newClusterSize[0]);
        free(local_newClusterSize);

        for (i=0; i<nthreads; i++)
            for (j=0; j<numClusters; j++)
                free(local_newClusters[i][j]);
        free(local_newClusters[0]);
        free(local_newClusters);
    }
    free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

}


/* Call k-means with k = [2, numObjs] clusters and
use the silhouette method to choice the best k value. 
Return a membership vector [numObjs]. */
int call_kmeans_silhouette(float **objects,   /* in: [numObjs][numCoords] */
                 int numObjs,       /* no. objects */
                 int numCoords,     /* no. coordinates */
                 int *membership    /* out: best clustering */)
{
    float threshold = 0.001;
    float **clusters;      /* [numClusters][numCoords] cluster center */
    int numClusters, maxClusters, minClusters;
    int lenDistances;
    float *sc; // silhouette score
    int i,j;

    minClusters = 2;
    if(numObjs < numCoords) maxClusters = numObjs-1;
    else maxClusters = numCoords;
    lenDistances = maxClusters-minClusters+1;
    
    /* allocate a 2D space for clusters[] (coordinates of cluster centers)
       this array should be the same across all processes                  */
    clusters    = (float**) malloc(maxClusters *             sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(maxClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (int i=1; i<maxClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    /* start the core computation -------------------------------------------*/
    /* membership: the cluster id for each data object */
    //membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    sc = (float*) calloc(lenDistances, sizeof(float));
    assert(sc != NULL);
    
    //printf("\n\n");
    /* run k-means to each possible no. of clusters */
    for (numClusters=minClusters; numClusters <= maxClusters; numClusters++)
    {
        int *cluster_sizes = (int*) calloc(numClusters, sizeof(int));
        assert(cluster_sizes != NULL);
        float s = 0.0;
        int i,j;
        int step = (int)(numObjs/numClusters);

        /* copy the first numClusters elements in feature[] */
    #pragma omp parallel for private(i,j)     \
        firstprivate(numClusters, numCoords, step) \
        shared(clusters, objects)
        for (i=0; i<numClusters; i++){
            for (j=0; j<numCoords; j++){
                clusters[i][j] = objects[(i*step + (i+1)*step) /2][j];
            }
        }

        omp_kmeans_silhouette(0, objects, numClusters-1, numObjs, numClusters, threshold, membership, clusters); 
        
        for(i=0; i < numObjs; i++){ cluster_sizes[membership[i]]++; }

        /*
        printf("\n k=%d \n[", numClusters);
        for(int i=0; i < numObjs; i++){ 
            printf("%d, ", membership[i]);
        }
        printf("]\n\n");
        */

        float a, min_b;
        int cluster_i;

    #pragma omp parallel for private(i,j, a, min_b, cluster_i)   \
        firstprivate(numObjs, numClusters) \
        shared(membership, objects, cluster_sizes)       \
        reduction(+:s)
        for(i=0; i < numObjs; i++)
        {
            a = 0;
            float *b = (float*) calloc(numClusters, sizeof(float));

            cluster_i = membership[i];

            if(cluster_sizes[cluster_i] > 1)
            {
                for(j=0; j < numObjs; j++)
                {
                    if(j != i)
                    {
                        if(cluster_i == membership[j]) 
                            a += sqrt(euclid_dist_2(numClusters-1, objects[i], objects[j]));
                        else {
                            b[membership[j]] += sqrt(euclid_dist_2(numClusters-1, objects[i], objects[j]));
                        }
                    }
                }

                a /= (cluster_sizes[cluster_i] - 1);
                
                min_b = INFINITY;
                for(j=0; j < numClusters; j++)
                {
                    float tmp = b[j]/(float)(cluster_sizes[j]);
                    if(tmp < min_b && cluster_i != j){ min_b = tmp; }
                }
                
                if(a > min_b) s += (min_b - a) / a;
                else s += (min_b - a) / min_b;
            }
            free(b);
        }
        
        sc[numClusters-minClusters] = s / numObjs;
        
        //printf("silhouette k=%d : %f\n", numClusters, sc[numClusters-minClusters]);
        free(cluster_sizes);
    }
    
    float sc_max = sc[0];
    numClusters = minClusters;
    for(i=1; i < lenDistances; i++)
    {
        if(sc[i] > sc_max)
        { 
            sc_max = sc[i];
            numClusters = minClusters+i;
        }
    }

    int step = (int)(numObjs/numClusters);

#pragma omp parallel for private(i,j)     \
    firstprivate(numClusters, numCoords, step) \
    shared(clusters, objects)
    for (i=0; i<numClusters; i++){
        for (j=0; j<numCoords; j++){
            clusters[i][j] = objects[(i*step + (i+1)*step) /2][j];
        }
    }

    omp_kmeans_silhouette(0, objects, numClusters-1, numObjs, numClusters, threshold, membership, clusters);
    
    free(sc);
    free(clusters[0]);
    free(clusters);

    return numClusters;
}


