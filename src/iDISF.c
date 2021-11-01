#include "iDISF.h"
#include "kmeans.h"
#include "eigs.h"


/**
 * Construct the unnormalized laplacian matrix
 * @param coords_markers in : NodeCoords[num_coords] cordinates to cluster
 * @param num_coords in : number of coordinate points
 * @param max_coords
 * @param image in : Image* with the pixels colors
 * @param graph in : Graph* image graph
 * @return double* unnormalized laplacian matrix [num_coords*num_coords], L = D - W
 */
double *laplacian_unnormalized(NodeCoords *coords_markers, int num_coords, int max_coords, Image *image, Graph *graph)
{
    int num_feats = image->num_channels;
    //int num_rows = image->num_rows;
    int num_cols = image->num_cols;
    int j;

    double *laplacian; // laplacian matrix
    double degree = 0.0;
    float media_vizinhos = 0;

    laplacian = (double *)calloc(max_coords * max_coords, sizeof(double));
    
    //printf("\n\n POINTS: \n");
    /*
    #pragma omp parallel for private(j) \
    firstprivate(num_coords, num_cols, num_feats) \
    lastprivate(degree) \
    shared(coords_markers, M, image, laplacian) \
    schedule(dynamic,1)
    */
    for (j = 0; j < num_coords; j++)
    {
        NodeCoords node_coords;
        
        int node_index, k;
        degree = 0.0;

        node_coords = coords_markers[j];
        node_index = node_coords.y * num_cols + node_coords.x;
        
        /*
        printf("%d (%d %d %d) , %d \n", node_marker_index - 1, image->val[node_index][0],
               image->val[node_index][1], image->val[node_index][2], node_index);
        */
        float feat1[3];

        for (int m = 0; m < num_feats; m++)
        {
            feat1[m] = (float)(image->val[node_index][m]);
        }

        float vizinhos = 0;
        
        // For each adjacent node
        // Mais lento com parallel for 
        for (k = 0; k < j; k++)
        {
            NodeCoords adj_coords;
            int adj_index, m;
            double w;
            float feat2[3];

            adj_coords = coords_markers[k];
            adj_index = adj_coords.y * num_cols + adj_coords.x;
            
            for (m = 0; m < num_feats; m++)
            {
                feat2[m] = (float)(image->val[adj_index][m]);
            }
            w = euclDistance(feat1, feat2, num_feats);
            //w = euclDistance(graph->feats[node_index], graph->feats[adj_index], num_feats);

            w = exp(-w);
            // L* axis (lightness) ranges from 0 to 100 ; a* and b* (color attributes) axis range from -128 to +127
            // So, the maximum euclidean distance is sqrt(100*100 + 255*255 + 255*255)

            laplacian[j * num_coords + k] = -w; // line node_index, column adj_index
            laplacian[k * num_coords + j] = -w; // line adj_index, column node_index
            laplacian[k * num_coords + k] += w;  // diagonal matrix (increase the degree of adj)
            degree += w;

            if(w > 0) vizinhos+=1;
        }
        laplacian[j * num_coords + j] = degree;
        media_vizinhos += vizinhos;
    }
    media_vizinhos /= num_coords;

    /*
    printf("\nLAPLACIAN (L = D - W):");
    for (int j = 0; j < num_coords * num_coords; j++)
    {
        if (j % num_coords == 0) printf("\n");
        printf("%f ", laplacian[j]);
    }
    */
    
    return laplacian;
}


/**
 * Construct the normalized laplacian matrix (symetric version)
 * @param coords_markers in : NodeCoords[num_coords] cordinates to cluster
 * @param num_coords in : number of coordinate points
 * @param image in : Image* with the pixels colors
 * @param adj_rel in : NodeAdj* adjacent relation used to matrix contruction
 * @return double* normalized laplacian matrix (symetric version) [num_coords*num_coords], L = D - W
 */
double *laplacian_normalized_sym(NodeCoords *coords_markers, int num_coords, int max_coords, Image *image)
{
    int num_feats = image->num_channels;
    int num_rows = image->num_rows;
    int num_cols = image->num_cols;

    int node_marker_index = 0;
    double *laplacian; // laplacian matrix
    int *M;            // mask matrix : store the index of each coord point (+1) in laplacian matrix and indicates the visited ones
    double *D;

    laplacian = (double *)calloc(max_coords * max_coords, sizeof(double));
    M = (int *)calloc(num_cols * num_rows, sizeof(int)); // initializes the mask matrix
    D = (double *)calloc(num_coords, sizeof(double));    // initializes the degree vector (only the diagonal)

    //printf("\n\n POINTS: \n");
    for (int j = 0; j < num_coords; j++)
    {
        NodeCoords node_coords;
        double degree = 0.0;
        int node_index;

        node_coords = coords_markers[j];
        node_index = node_coords.y * num_cols + node_coords.x;
        node_marker_index++;
        M[node_index] = node_marker_index; // maps the node index to the marker index in laplacian matrix +1

        //printf("%d (%d %d %d) , %d \n", node_marker_index - 1, image->val[node_index][0],
        //       image->val[node_index][1], image->val[node_index][2], node_index);

        // For each adjacent node
        for (int k = 0; k < j; k++)
        {
            NodeCoords adj_coords;

            adj_coords = coords_markers[k];

            int adj_marker_index;
            int adj_index;
            double w;

            adj_index = adj_coords.y * num_cols + adj_coords.x;
            adj_marker_index = M[adj_index];
            //w = euclDistance(graph->feats[node_index], graph->feats[adj_index], num_feats);
            float feat1[3], feat2[3];
            for (int m = 0; m < num_feats; m++)
            {
                feat1[m] = (float)(image->val[node_index][m]);
                feat2[m] = (float)(image->val[adj_index][m]);
            }
            w = euclDistance(feat1, feat2, num_feats);
            w = exp(-w);
            // L* axis (lightness) ranges from 0 to 100 ; a* and b* (color attributes) axis range from -128 to +127
            // So, the maximum euclidean distance is sqrt(100*100 + 255*255 + 255*255)

            laplacian[(node_marker_index - 1) * num_coords + (adj_marker_index - 1)] = -w; // line node_index, column adj_index
            laplacian[(adj_marker_index - 1) * num_coords + (node_marker_index - 1)] = -w; // line adj_index, column node_index
            D[adj_marker_index - 1] += w;
            degree += w;
        }
        D[node_marker_index - 1] += degree;
    }

    /*
    printf("\n AFFINITY MATRIX: ");
    for (int j = 0; j < num_coords * num_coords; j++)
    {
        if (j % num_coords == 0)
            printf("\n");
        printf("%f ", laplacian[j]);
    }
    */

    //printf("\n\n DEGREE VECTOR: ");
    for (int j = 0; j < num_coords; j++)
    {
        //printf("%f \t", D[j]);
        for (int k = j + 1; k < num_coords; k++)
        {
            double d = sqrt(D[j] * D[k]);
            laplacian[j * num_coords + k] /= d;
            laplacian[k * num_coords + j] /= d;
        }
        laplacian[j * num_coords + j] = 1;
    }
    //printf("\n");

    /*
    printf("\n LAPLACIAN MATRIX (L = I - D^(-1/2)*W*D^(-1/2)): ");
    for (int j = 0; j < num_coords * num_coords; j++)
    {
        if (j % num_coords == 0)
            printf("\n");
        printf("%f ", laplacian[j]);
    }
    */

    free(D);
    free(M);
    return laplacian;
}

/**
 * Construct the normalized laplacian matrix (random walk version)
 * @param coords_markers in : NodeCoords[num_coords] cordinates to cluster
 * @param num_coords in : number of coordinate points
 * @param image in : Image* with the pixels colors
 * @param adj_rel in : NodeAdj* adjacent relation used to matrix contruction
 * @return double* normalized laplacian matrix (random walk version) [num_coords*num_coords], L = D - W
 */
double *laplacian_normalized_rw(NodeCoords *coords_markers, int num_coords, int max_coords, Image *image)
{
    int num_feats = image->num_channels;
    int num_rows = image->num_rows;
    int num_cols = image->num_cols;

    int node_marker_index = 0;
    double *laplacian; // laplacian matrix
    int *M;            // mask matrix : store the index of each coord point (+1) in laplacian matrix and indicates the visited ones
    double *D;

    laplacian = (double *)calloc(max_coords * max_coords, sizeof(double));
    M = (int *)calloc(num_cols * num_rows, sizeof(int)); // initializes the mask matrix
    D = (double *)calloc(num_coords, sizeof(double));    // initializes the degree vector (only the diagonal)

    //printf("\n\n POINTS: \n");
    for (int j = 0; j < num_coords; j++)
    {
        NodeCoords node_coords;
        double degree = 0.0;
        int node_index;

        node_coords = coords_markers[j];
        node_index = node_coords.y * num_cols + node_coords.x;
        node_marker_index++;
        M[node_index] = node_marker_index; // maps the node index to the marker index in laplacian matrix +1

        //printf("%d (%d %d %d) , %d \n", node_marker_index - 1, image->val[node_index][0],
        //       image->val[node_index][1], image->val[node_index][2], node_index);

        // For each adjacent node
        for (int k = 0; k < j; k++)
        {
            NodeCoords adj_coords;

            adj_coords = coords_markers[k];

            int adj_marker_index;
            int adj_index;
            double w;

            adj_index = adj_coords.y * num_cols + adj_coords.x;
            adj_marker_index = M[adj_index];
            //w = euclDistance(graph->feats[node_index], graph->feats[adj_index], num_feats);
            float feat1[3], feat2[3];
            for (int m = 0; m < num_feats; m++)
            {
                feat1[m] = (float)(image->val[node_index][m]);
                feat2[m] = (float)(image->val[adj_index][m]);
            }
            w = euclDistance(feat1, feat2, num_feats);
            w = exp(-w);
            // L* axis (lightness) ranges from 0 to 100 ; a* and b* (color attributes) axis range from -128 to +127
            // So, the maximum euclidean distance is sqrt(100*100 + 255*255 + 255*255)

            laplacian[(node_marker_index - 1) * num_coords + (adj_marker_index - 1)] = -w; // line node_index, column adj_index
            laplacian[(adj_marker_index - 1) * num_coords + (node_marker_index - 1)] = -w; // line adj_index, column node_index
            D[adj_marker_index - 1] += w;
            degree += w;
        }
        D[node_marker_index - 1] += degree;
    }

    /*
    printf("\n AFFINITY MATRIX: ");
    for (int j = 0; j < num_coords * num_coords; j++)
    {
        if (j % num_coords == 0)
            printf("\n");
        printf("%f ", laplacian[j]);
    }
    */

    //printf("\n\n DEGREE VECTOR: ");
    for (int j = 0; j < num_coords; j++)
    {
        //printf("%f \t", D[j]);
        for (int k = j + 1; k < num_coords; k++)
        {
            laplacian[j * num_coords + k] /= D[j]; // L[j][k]
            laplacian[k * num_coords + j] /= D[k];
        }
        laplacian[j * num_coords + j] = 1;
    }
    //printf("\n");

    /*
    printf("\n LAPLACIAN MATRIX (L = I - D^(-1)*W): ");
    for (int j = 0; j < num_coords * num_coords; j++)
    {
        if (j % num_coords == 0) printf("\n");
        printf("%f ", laplacian[j]);
    }
    */

    free(D);
    free(M);

    return laplacian;
}

/**
 * Spetral clustering for a set of points
 * @param graph in : graph
 * @param coords_markers in : NodeCoords[num_user_seeds][marker_sizes]
 * @param marker_sizes in : &int[num_user_seeds] -- the size of each marker
 * @param num_markers in : int number of markers
 * @param numTotalClusters out : &int total number of clusters (all markers)
 * @param normalize in : int 0 (unnormalized), 1 (laplacian sym), or 2 (laplacian rw)
 * @return int vector with the labels of the markers coordinates
 */
int *clusterPoints(Image *image, Graph *graph, NodeCoords **coords_markers, int *marker_sizes, int num_markers, int *numTotalClusters, int normalize)
{
    //NodeAdj *adj_rel = create4NeighAdj();
    //double maxDistance = sqrt(100*100+2*255*255); // color: L=[0,100] a=[-127,128] b=[-127,128]
    //double maxDistance = sqrt(3 * 255 * 255);
    int *labels;
    int total_coords = 0; // controls labels vector index
    int lastCluster = 0;  // costrols labels vector values

    for (int i = 0; i < num_markers; i++)
        total_coords += marker_sizes[i];
    labels = (int *)allocMem(total_coords, sizeof(int));
    total_coords = 0;

    for (int i = 0; i < num_markers; i++)
    {
        int num_coords;
        double *laplacian; // laplacian matrix
        double *eigenvalues;
        double *eigenvectors;
        int n_eigenValues, numClusters;
        int n_features;
        double *D;
        float **objects;
        int *membership;

        num_coords = marker_sizes[i];            // get the total number of marker coords
        n_eigenValues = MIN(11, num_coords - 1); // spectra framework has a limitation of maximum num_coords-1 eigenvalues/eigenvectors
        laplacian = (double *)calloc(num_coords * num_coords, sizeof(double));

        switch (normalize)
        {
        case 0:
            laplacian = laplacian_unnormalized(coords_markers[i], num_coords, num_coords, image, graph);
            break;
        case 1:
            laplacian = laplacian_normalized_sym(coords_markers[i], num_coords, num_coords, image);
            break;
        case 2:
            laplacian = laplacian_normalized_rw(coords_markers[i], num_coords, num_coords, image);
            break;
        default:
            printError("clusterPoints", "Invalid normalize option.");
            break;
        }

        // spectra returns from higher to small eigenvalue ordering)
        eigenvalues = (double *)malloc(n_eigenValues * sizeof(double));
        eigenvectors = (double *)malloc(num_coords * n_eigenValues * sizeof(double));
        // eigenvectors : an matrix (in a unique vector) num_coords x n_eigenValues,
        // in which each line is a point and its columns are the features
        smallest_eigenvalues(laplacian, num_coords, n_eigenValues, eigenvalues, eigenvectors);

        //n_features = n_eigenValues-1; // we do not use the last eigenvector (whose with the smallest eigenvalue -- always 0)
        n_features = n_eigenValues - 1; // we do not use the last eigenvector (whose with the smallest eigenvalue -- always 0)
        objects = (float **)malloc(num_coords * sizeof(float *));
        if (objects == NULL)
            printError("clusterPoints_normalized", "Was do not possible to alloc objects.");

        /*
        printf("\n\nEIGENVALUES: ");
        for (int j = 0; j < n_eigenValues; j++) printf("%f \t", eigenvalues[j]);

        
        printf("\n\nEIGENVECTORS: ");
        for (int j = 0; j < num_coords * n_eigenValues; j++)
        {
            if (j % n_eigenValues == 0) printf("\n");
            printf("%f ", eigenvectors[j]);
        }
        */

        switch (normalize)
        {
        case 0:
            for (int j = 0; j < num_coords; j++)
            {
                objects[j] = (float *)malloc(n_features * sizeof(float));
                for (int k = 0; k < n_features; k++)
                {
                    objects[j][k] = (float)(eigenvectors[j * n_eigenValues + (n_features - 1 - k)]);
                    // the object j is the positions [j,{k_0, .., k_max}] of the eigenvectors matrix
                    // we also invert the columns (features) positions to obtain an ascending order concerning to its eigenvalues
                }
            }
            break;
        case 1:
            D = (double *)calloc(num_coords, sizeof(double)); // store the sum of eigenvectors[i][k]^2 for k=[0,..,n_eigenValues] to normalize it
            for (int j = 0; j < num_coords; j++)
            {
                double tmp = 0;
                for (int k = 0; k < n_eigenValues; k++)
                {
                    double tmp2 = eigenvectors[j * n_eigenValues + k];
                    tmp += tmp2 * tmp2;
                }
                D[j] = sqrt(tmp);
            }

            //printf("\n\n NORMALIZED EIGENVECTORS: \n");
            for (int j = 0; j < num_coords; j++)
            {
                objects[j] = (float *)malloc(n_features * sizeof(float));
                for (int k = 0; k < n_features; k++)
                {
                    objects[j][k] = (float)(eigenvectors[j * n_eigenValues + (n_features - 1 - k)] / D[j]);
                    //printf("%f ", objects[j][k]/D[j]);
                    // the object j is the positions [j,{k_0, .., k_max}] of the eigenvectors matrix
                    // we also invert the columns (features) positions to obtain an ascending order concerning to its eigenvalues
                }
                //printf("\n");
            }
            free(D);
            break;
        case 2:
            for (int j = 0; j < num_coords; j++)
            {
                objects[j] = (float *)malloc(n_features * sizeof(float));
                for (int k = 0; k < n_features; k++)
                {
                    objects[j][k] = (float)(eigenvectors[j * n_eigenValues + (n_features - 1 - k)]);
                    //printf("%f ", objects[j][k]);
                    // the object j is the positions [j,{k_0, .., k_max}] of the eigenvectors matrix
                    // we also invert the columns (features) positions to obtain an ascending order concerning to its eigenvalues
                }
                //printf("\n");
            }
            break;
        default:
            printError("clusterPoints", "Invalid normalize option.");
            break;
        }

        /* membership: the cluster id for each data object */
        membership = (int *)malloc(num_coords * sizeof(int));
        if (membership == NULL)
            printError("gridSampling_scribbles_clust", "Was do not possible to alloc membership.");

        //numClusters = call_kmeans_elbow(objects, num_coords, n_features, membership);
        /*numClusters = */call_kmeans_silhouette(objects, num_coords, n_features, membership);
        //printf("\n numClusters = %d\n", numClusters);

        for (int j = 0; j < num_coords; j++)
        {
            //printf("%d %d\n", j, membership[j]);
            labels[total_coords + j] = membership[j] + lastCluster;
        }
        total_coords += num_coords;
        lastCluster += numClusters;

        free(eigenvectors);
        free(eigenvalues);
        free(laplacian);
        for (int j = 0; j < num_coords; j++)
            free(objects[j]);
        free(objects);
        free(membership);
    }
    (*numTotalClusters) = lastCluster;
    //printf("\n FIM CLUSTER POINTS \n");
    return labels;
}

//=============================================================================
// Constructors & Deconstructors
//=============================================================================
NodeAdj *create4NeighAdj()
{
    NodeAdj *adj_rel;

    adj_rel = allocMem(1, sizeof(NodeAdj));

    adj_rel->size = 4;
    adj_rel->dx = allocMem(4, sizeof(int));
    adj_rel->dy = allocMem(4, sizeof(int));

    adj_rel->dx[0] = -1;
    adj_rel->dy[0] = 0; // Left
    adj_rel->dx[1] = 1;
    adj_rel->dy[1] = 0; // Right

    adj_rel->dx[2] = 0;
    adj_rel->dy[2] = -1; // Top
    adj_rel->dx[3] = 0;
    adj_rel->dy[3] = 1; // Bottom

    return adj_rel;
}

NodeAdj *create8NeighAdj()
{
    NodeAdj *adj_rel;

    adj_rel = allocMem(1, sizeof(NodeAdj));

    adj_rel->size = 8;
    adj_rel->dx = allocMem(8, sizeof(int));
    adj_rel->dy = allocMem(8, sizeof(int));

    adj_rel->dx[0] = -1;
    adj_rel->dy[0] = 0; // Center-Left
    adj_rel->dx[1] = 1;
    adj_rel->dy[1] = 0; // Center-Right

    adj_rel->dx[2] = 0;
    adj_rel->dy[2] = -1; // Top-Center
    adj_rel->dx[3] = 0;
    adj_rel->dy[3] = 1; // Bottom-Center

    adj_rel->dx[4] = -1;
    adj_rel->dy[4] = 1; // Bottom-Left
    adj_rel->dx[5] = 1;
    adj_rel->dy[5] = -1; // Top-Right

    adj_rel->dx[6] = -1;
    adj_rel->dy[6] = -1; // Top-Left
    adj_rel->dx[7] = 1;
    adj_rel->dy[7] = 1; // Bottom-Right

    return adj_rel;
}

Graph *createGraph(Image *img)
{
    int normval, num_nodes, num_channels, i;
    //NodeAdj *adj_rel;
    Graph *graph;

    normval = getNormValue(img);

    graph = allocMem(1, sizeof(Graph));

    graph->num_cols = img->num_cols;
    graph->num_rows = img->num_rows;
    graph->num_feats = 3; // L*a*b cspace
    num_nodes = graph->num_nodes = img->num_pixels;
    num_channels = img->num_channels;

    graph->feats = allocMem(num_nodes, sizeof(float *));

#pragma omp parallel for private(i)                \
    firstprivate(num_nodes, num_channels, normval) \
    shared(graph, img)
    for (i = 0; i < num_nodes; i++)
    {
        if (num_channels <= 2) // Grayscale w/ w/o alpha
            graph->feats[i] = convertGrayToLab(img->val[i], normval);
        else // sRGB
            graph->feats[i] = convertsRGBToLab(img->val[i], normval);

        /*
        float *srgb;
        srgb = allocMem(3, sizeof(float));
        srgb[0] = srgb[1] = srgb[2] = (float)img->val[i][0];
        graph->feats[i] = srgb;
        NodeCoords coords = getNodeCoords(num_cols, i);
       printf("%d;%d -> %f;  ", coords.x, coords.y, srgb[2]);
       */
    }
    /*
    adj_rel = create8NeighAdj();

    // Smoothing
    for (int i = 0; i < graph->num_nodes; i++)
    {
        NodeCoords node_coords;

        node_coords = getNodeCoords(num_cols, i);

        for (int j = 0; j < graph->num_feats; j++)
        {
            float smooth_val;

            smooth_val = graph->feats[i][j] * GAUSSIAN_3x3[0];

            // For each adjacent node
            for (int k = 0; k < adj_rel->size; k++)
            {
                NodeCoords adj_coords;

                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, k);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index;

                    adj_index = getNodeIndex(num_cols, adj_coords);

                    smooth_val += graph->feats[adj_index][j] * GAUSSIAN_3x3[k];
                }
            }

            graph->feats[i][j] = smooth_val;
        }
    }

    freeNodeAdj(&adj_rel);
    */
    return graph;
}

Tree *createTree(int root_index, int num_feats)
{
    Tree *tree;

    tree = allocMem(1, sizeof(Tree));

    tree->root_index = root_index;
    tree->num_nodes = 0;
    tree->num_feats = num_feats;
    tree->sum_grad = 0;
    tree->sum_grad_2 = 0;
    tree->minDist_userSeed = INFINITY;

    tree->sum_feat = allocMem(num_feats, sizeof(float));

    return tree;
}

void freeNodeAdj(NodeAdj **adj_rel)
{
    if (*adj_rel != NULL)
    {
        NodeAdj *tmp;

        tmp = *adj_rel;

        freeMem(tmp->dx);
        freeMem(tmp->dy);
        freeMem(tmp);

        *adj_rel = NULL;
    }
}

void freeGraph(Graph **graph)
{
    if (*graph != NULL)
    {
        Graph *tmp;
        int i, num_nodes;

        tmp = *graph;
        num_nodes = tmp->num_nodes;

    #pragma omp parallel for \
        private(i) \
        firstprivate(num_nodes) \
        shared(tmp)
        for (i = 0; i < num_nodes; i++)
            freeMem(tmp->feats[i]);
        freeMem(tmp->feats);
        freeMem(tmp);

        *graph = NULL;
    }
}

void freeTree(Tree **tree)
{
    if (*tree != NULL)
    {
        Tree *tmp;

        tmp = *tree;

        freeMem(tmp->sum_feat);
        freeMem(tmp);

        *tree = NULL;
    }
}

//=============================================================================
// Bool Functions
//=============================================================================
inline bool areValidNodeCoords(int num_rows, int num_cols, NodeCoords coords)
{
    return (coords.x >= 0 && coords.x < num_cols) &&
           (coords.y >= 0 && coords.y < num_rows);
}

//=============================================================================
// Int Functions
//=============================================================================
inline int getNodeIndex(int num_cols, NodeCoords coords)
{
    return coords.y * num_cols + coords.x;
}

//=============================================================================
// Double Functions
//=============================================================================
inline double euclDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for (int i = 0; i < num_feats; i++)
        dist += (feat1[i] - feat2[i]) * (feat1[i] - feat2[i]);
    dist = sqrtf(dist);

    return dist;
}

inline double taxicabDistance(float *feat1, float *feat2, int num_feats)
{
    double dist;

    dist = 0;

    for (int i = 0; i < num_feats; i++)
        dist += fabs(feat1[i] - feat2[i]);

    return dist;
}

inline double euclDistanceCoords(NodeCoords feat1, NodeCoords feat2)
{
    double dist;

    dist = 0;

    dist += ((float)feat1.x - (float)feat2.x) * ((float)feat1.x - (float)feat2.x);
    dist += ((float)feat1.y - (float)feat2.y) * ((float)feat1.y - (float)feat2.y);
    dist = sqrtf(dist);

    return dist;
}

inline double calcPathCost(float *mean_feat_tree, float *feats, int num_feats, double cost_map, int num_nodes_tree, double grad_adj, double coef_variation_tree, double alpha, double c2, int function)
{
    double arc_cost, path_cost, diff_grad, beta;

    arc_cost = euclDistance(mean_feat_tree, feats, num_feats);

    if (function == 1) // color distance
        path_cost = MAX(cost_map, arc_cost);
    else
    {
        if (function == 2)
        { // not normalization
            beta = MAX(MAX(1, alpha * c2), coef_variation_tree);
            diff_grad = arc_cost * pow(grad_adj, (1 / alpha)) * (1 / beta);
            path_cost = MAX(cost_map, diff_grad);
        }
        else
        {
            if (function == 3)
            { // beta normalization
                beta = MAX(MAX(1, alpha * c2), coef_variation_tree) / (float)num_nodes_tree;
                diff_grad = arc_cost * pow(grad_adj, (1 / alpha)) * (1 / beta);
                path_cost = MAX(cost_map, diff_grad);
            }
            else
            {
                if (function == 4)
                { // cv normalization
                    beta = MAX(MAX(1, alpha * c2), coef_variation_tree / (float)num_nodes_tree);
                    diff_grad = arc_cost * pow(grad_adj, (1 / alpha)) * (1 / beta);
                    path_cost = MAX(cost_map, diff_grad);
                }
                else
                {
                    if (function == 5)
                    { // sum gradient not norm
                        beta = MAX(MAX(1, alpha * c2), coef_variation_tree);
                        diff_grad = arc_cost + (pow(grad_adj, (1 / alpha)) * (1 / beta));
                        path_cost = MAX(cost_map, diff_grad);
                    }
                    else
                    { // sum gradient beta norm
                        beta = MAX(MAX(1, alpha * c2), coef_variation_tree) / (float)num_nodes_tree;
                        diff_grad = arc_cost + pow(grad_adj, (1 / alpha)) * (1 / beta);
                        path_cost = MAX(cost_map, diff_grad);
                    }
                }
            }
        }
    }

    return path_cost;
}

//=============================================================================
// NodeCoords Functions
//=============================================================================
inline NodeCoords getAdjacentNodeCoords(NodeAdj *adj_rel, NodeCoords coords, int id)
{
    NodeCoords adj_coords;

    adj_coords.x = coords.x + adj_rel->dx[id];
    adj_coords.y = coords.y + adj_rel->dy[id];

    return adj_coords;
}

inline NodeCoords getNodeCoords(int num_cols, int index)
{
    NodeCoords coords;

    coords.x = index % num_cols;
    coords.y = index / num_cols;

    return coords;
}

//=============================================================================
// Float* Functions
//=============================================================================
inline float *meanTreeFeatVector(Tree *tree)
{
    float *mean_feat;

    mean_feat = allocMem(tree->num_feats, sizeof(float));

    for (int i = 0; i < tree->num_feats; i++)
        mean_feat[i] = tree->sum_feat[i] / (float)tree->num_nodes;

    return mean_feat;
}

//=============================================================================
// Double* Functions
//=============================================================================
double *computeGradient(Graph *graph, double *coef_variation_img)
{
    float max_adj_dist, sum_weight;
    float *dist_weight;
    double *grad;
    NodeAdj *adj_rel;
    double sum_grad = 0, sum_grad_2 = 0;
    double variance, mean;
    int num_cols, num_rows, num_nodes, num_feats;
    //double max_grad, min_grad;

    float div, *feats, *adj_feats;

    int i, j, rel_size, adj_index;
    NodeCoords coords, adj_coords;
    double dist;

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    grad = allocMem(num_nodes, sizeof(double));
    adj_rel = create8NeighAdj();

    rel_size = adj_rel->size;
    max_adj_dist = sqrtf(2); // Diagonal distance for 8-neighborhood
    dist_weight = allocMem(adj_rel->size, sizeof(float));
    sum_weight = 0;

    /* private : Cláusula que define que as variáveis definidas em list são duplicadas em cada
thread e o seu acesso passa a ser local (privado) em cada thread. O valor inicial das variáveis
privadas é indefinido (não é iniciado) e o valor final das variáveis originais (depois da região
paralela) também é indefinido.*/

    // firstprivate : Variáveis privadas que são inicializadas quando o código paralelo é iniciado

    /* shared : Cláusula que define sobre as variáveis definidas em list são partilhadas por todos
os threads ficando à responsabilidade do programador garantir o seu correto manuseamento.
Por omissão, as variáveis para as quais não é definido qualquer tipo são consideradas variáveis
partilhadas. */

    //max_grad = -1;
    //min_grad = 20000;

#pragma omp parallel for private(i, div) \
    firstprivate(max_adj_dist, rel_size) \
        shared(dist_weight, adj_rel)     \
            reduction(+                  \
                      : sum_weight)
    // Compute the inverse distance weights (closer --> higher; farther --> lower)
    for (i = 0; i < rel_size; i++)
    {
        // Distance between the adjacent and the center
        div = sqrtf(adj_rel->dx[i] * adj_rel->dx[i] + adj_rel->dy[i] * adj_rel->dy[i]);

        dist_weight[i] = max_adj_dist / div;
        sum_weight += dist_weight[i];
    }

#pragma omp parallel for private(i)    \
    firstprivate(rel_size, sum_weight) \
        shared(dist_weight)
    // Normalize values
    for (i = 0; i < rel_size; i++)
    {
        dist_weight[i] /= sum_weight;
    }

    // Compute the gradients
    for (i = 0; i < num_nodes; i++)
    {
        feats = graph->feats[i];
        coords = getNodeCoords(num_cols, i);

        // For each adjacent node
        for (j = 0; j < rel_size; j++)
        {
            adj_coords = getAdjacentNodeCoords(adj_rel, coords, j);

            if (areValidNodeCoords(num_rows, num_cols, adj_coords))
            {
                //adj_index = getNodeIndex(num_cols, adj_coords);
                adj_index = adj_coords.y * num_cols + adj_coords.x;
                adj_feats = graph->feats[adj_index];

                // Compute L1 Norm between center and adjacent
                dist = taxicabDistance(adj_feats, feats, num_feats);

                // Weight by its distance relevance
                grad[i] += dist * dist_weight[j];
            }
        }
        sum_grad += grad[i];
        sum_grad_2 += grad[i] * grad[i];
    }

    variance = (sum_grad_2 / (float)graph->num_nodes) - ((sum_grad * sum_grad) / ((float)graph->num_nodes * (float)graph->num_nodes));
    mean = sum_grad / (float)graph->num_nodes;

    (*coef_variation_img) = sqrt(MAX(0, variance)) / MAX(0.001, mean);

    freeMem(dist_weight);
    freeNodeAdj(&adj_rel);

    return grad;
}

//=============================================================================
// Float Functions
//=============================================================================
inline float treeVariance(Tree *tree)
{
    float variance;

    variance = (tree->sum_grad_2 / (float)tree->num_nodes) - ((tree->sum_grad * tree->sum_grad) / (((float)tree->num_nodes) * (float)tree->num_nodes));

    return variance;
}

inline float coefTreeVariation(Tree *tree)
{
    double tree_variance;
    double grad_mean, coef_variation_tree;

    tree_variance = (tree->sum_grad_2 / (double)tree->num_nodes) - ((tree->sum_grad * tree->sum_grad) / (((double)tree->num_nodes) * (double)tree->num_nodes));
    grad_mean = tree->sum_grad / (double)tree->num_nodes;
    coef_variation_tree = sqrt(MAX(0, tree_variance)) / MAX(0.00001, grad_mean);

    return coef_variation_tree;
}

//=============================================================================
// Image* Functions
//=============================================================================
Image *runiDISF_scribbles_rem(Graph *graph,                   // input: image graph in lab color space
                              int n_0,                        // input: desired amount of GRID seeds
                              int iterations,                 // input: maximum iterations or final superpixels (if segm_method=1)
                              Image **border_img,             // input/output: empty image to keep the superpixels borders
                              NodeCoords **coords_user_seeds, // input: coords of the scribble pixels
                              int num_markers,                // input: amount of scribbles
                              int *marker_sizes,              // input: amount of pixels in each scribble
                              int function,                   // input: path-cost function {1 : euclidean, 2 : coefficient gradient, 3 : gradient beta norm., 4 : coefficient gradient norm., 5 : sum. coefficient gradient}
                              int all_borders,                // input: flag. If 1, map border_img with the tree borders
                              double c1, double c2,           // input: parameters of path-cost functions 2-5
                              //int sampling_method,            // unused
                              int obj_markers) // input: amount of object scribbles
{
    //clock_t time;
    bool want_borders;
    int stop;
    int iter, num_cols, num_rows, num_nodes, num_feats;
    int *labels_map; // labels_map : mapeia da label 1 para a label 2;
    //int *pred_map;
    double *cost_map, *grad, alpha;
    NodeAdj *adj_rel;
    IntList *seed_set; // passa a ser recebido na função para ser reenviado na próxima interação
    Image *label_img2;
    int *label_img;
    PrioQueue *queue;
    float size, stride, delta, max_seeds;

    stop = 0;
    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    // Init auxiliary structures
    cost_map = allocMem(num_nodes, sizeof(double)); // f
    //pred_map = allocMem(img->num_pixels, sizeof(int));    // P
    adj_rel = create4NeighAdj();
    label_img = allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, 1);
    queue = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    want_borders = border_img != NULL;

    grad = computeGradient(graph, &alpha);

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)num_nodes / ((float)n_0));
    stride = sqrtf(size) + 0.5;
    delta = stride / 2.0;
    delta = (int)delta;
    stride = (int)stride;

    // Compute the max amount of GRID seeds
    max_seeds = ((((float)num_rows - delta) / stride) + 1) * ((((float)num_cols - delta) / stride) + 1);

    // Compute the max amount of all seeds
    for (int i = 0; i < num_markers; i++)
        max_seeds += marker_sizes[i];

    labels_map = allocMem((int)max_seeds, sizeof(int)); // mapeia do indice da árvore para o rótulo da regiao
    
    //time = clock();
    seed_set = gridSampling_scribbles(num_rows, num_cols, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, labels_map, obj_markers);
    /*time = clock() - time;
    printf("seed sampling %.3f\n", ((double)time) / CLOCKS_PER_SEC);*/

    if (c1 <= 0)
        c1 = 0.7;
    if (c2 <= 0)
        c2 = 0.8;

    alpha = MAX(c1, alpha);
    iter = 1;

    // At least a single iteration is performed
    do
    {
        //time = clock();
        int seed_label, num_trees;
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;

        trees = allocMem(seed_set->size, sizeof(Tree *));
        tree_adj = allocMem(seed_set->size, sizeof(IntList *));
        are_trees_adj = allocMem(seed_set->size, sizeof(bool *));

// Assign initial values for all nodes
#pragma omp parallel for firstprivate(num_nodes, want_borders) \
    shared(cost_map, label_img, border_img)
        for (int i = 0; i < num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            //pred_map[i] = -1;
            label_img[i] = -1; // 1 rótulo por árvore

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        seed_label = 0;
        // cria uma árvore para cada pixel de cada marcador
        // porém, cada marcador recebe um rótulo distinto
        // e as demais sementes (GRID) recebem um outro rótulo
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index;
            seed_index = ptr->elem;
            cost_map[seed_index] = 0;
            label_img[seed_index] = seed_label;
            trees[seed_label] = createTree(seed_index, num_feats);
            tree_adj[seed_label] = createIntList();
            are_trees_adj[seed_label] = allocMem(seed_set->size, sizeof(bool));
            seed_label++;
            /*
	        int label = labels_map[seed_label];
            if(label > num_markers){
                NodeCoords coords = getNodeCoords(num_cols,seed_index);
            }*/
            insertPrioQueue(&queue, seed_index);
        }
        /*
        time = clock() - time;
        printf("pre IFT %.3f \t", ((double)time) / CLOCKS_PER_SEC);*/

        //time = clock();
        // For each node within the queue
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree /*, mean_grad_tree*/;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            node_label2 = labels_map[node_label];
            if (node_label2 > obj_markers)
            {
                label_img2->val[node_index][0] = 2;
                node_label2 = 2;
            }
            else
            {
                label_img2->val[node_index][0] = 1;
                node_label2 = 1;
            }

            // We insert the features in the respective tree at this
            // moment, because it is guaranteed that this node will not
            // be inserted ever again.
            insertNodeInTree(graph, node_index, &(trees[node_label]), grad[node_index]);

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label;
                    double path_cost;

                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];
                    path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            //pred_map[adj_index] = node_index;

                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        // relação de adjacência entre árvores
                        if (node_label != adj_label)
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }

                        int adj_label2 = labels_map[adj_label];
                        if (adj_label2 > obj_markers)
                        {
                            adj_label2 = 2;
                        }
                        else
                        {
                            adj_label2 = 1;
                        }

                        // cria bordas entre rótulos (label2)
                        if (want_borders && ((all_borders == 0 && node_label2 != adj_label2) || (all_borders == 1 && node_label != adj_label)))
                        {
                            if (want_borders) // Both depicts a border between their superpixels
                            {
                                (*border_img)->val[node_index][0] = 255;
                                (*border_img)->val[adj_index][0] = 255;
                            }
                        }
                    }
                }
            }
            free(mean_feat_tree);
        }
        /*
        time = clock() - time;
        printf("IFT %.3f \t", ((double)time) / CLOCKS_PER_SEC);*/

        /*** SEED SELECTION ***/
        // Auxiliar var
        num_trees = seed_set->size;
        freeIntList(&seed_set);

        if (stop == 2)
            stop = 1; // se já executou a uma iteracao apos remover todas as sementes grid
        if (iter < iterations && stop == 0)
        {
            //time = clock();
            //printf("Select the most relevant superpixels, iter=%d, iterations=%d \n", iter, iterations);
            // Select the most relevant superpixels
            seed_set = seedRemoval(trees, tree_adj, num_nodes, num_trees, num_markers, obj_markers, labels_map, &stop);
            /*time = clock() - time;
            printf("seed removal %.3f\n", ((double)time) / CLOCKS_PER_SEC);*/
        }

        iter++;                 // next iter
        resetPrioQueue(&queue); // Clear the queue

#pragma omp parallel for firstprivate(num_trees)
        for (int i = 0; i < num_trees; i++)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);

        // o loop termina quando realiza a quantidade definida de iterações
    } while (iter <= iterations && stop != 1);

    freeMem(grad);
    freeMem(label_img);
    freeMem(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);
    freeMem(labels_map);

    return label_img2;
}

// compute iDISF using k-means on the scribble pixels
Image *runiDISF_scribbles_clust_bkp(Graph *graph,                   // input: image graph in lab color space
                                    int n_0,                        // input: desired amount of GRID seeds
                                    int iterations,                 // input: maximum iterations or final superpixels (if segm_method=1)
                                    Image **border_img,             // input/output: empty image to keep the superpixels borders
                                    NodeCoords **coords_user_seeds, // input: coords of the scribble pixels
                                    int num_markers,                // input: amount of scribbles
                                    int *marker_sizes,              // input: amount of pixels in each scribble
                                    int function,                   // input: path-cost function {1 : euclidean, 2 : coefficient gradient, 3 : gradient beta norm., 4 : coefficient gradient norm., 5 : sum. coefficient gradient}
                                    int all_borders,                // input: flag. If 1, map border_img with the tree borders
                                    double c1, double c2,           // input: parameters of path-cost functions 2-5
                                    //int sampling_method,            // unused
                                    int obj_markers) // input: amount of object scribbles
{

    bool want_borders;
    int stop;
    int iter, numTrees, num_cols, num_rows, num_nodes, num_feats;
    int *labels_map; // labels_map : mapeia de tree id para a label de segmentação;
    int *label_seed; // mapeida de seed index para tree id
    //int *pred_map;
    double *cost_map, *grad, alpha;
    NodeAdj *adj_rel;
    IntList *seed_set; // passa a ser recebido na função para ser reenviado na próxima interação
    int *label_img;
    Image *label_img2;
    PrioQueue *queue;
    float size, stride, delta, max_seeds;

    stop = 0;
    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    // Init auxiliary structures
    cost_map = allocMem(graph->num_nodes, sizeof(double)); // f
    //pred_map = allocMem(img->num_pixels, sizeof(int));    // P
    adj_rel = create4NeighAdj();
    label_img = allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, 1);
    queue = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    label_seed = allocMem(num_nodes, sizeof(int));
    want_borders = border_img != NULL;

    grad = computeGradient(graph, &alpha);

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)num_nodes / ((float)n_0));
    stride = sqrtf(size) + 0.5;
    delta = stride / 2.0;
    delta = (int)delta;
    stride = (int)stride;

    // Compute the max amount of GRID seeds
    max_seeds = ((((float)num_rows - delta) / stride) + 1) * ((((float)num_cols - delta) / stride) + 1);

    // Compute the max amount of all seeds
    for (int i = 0; i < num_markers; i++)
        max_seeds += marker_sizes[i];
    numTrees = max_seeds;

    labels_map = allocMem((int)max_seeds, sizeof(int)); // mapeia do indice da árvore para o rótulo da regiao
    seed_set = gridSampling_scribbles_clust_bkp(graph, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, labels_map, obj_markers, label_seed, &numTrees);

    if (c1 <= 0)
        c1 = 0.7;
    if (c2 <= 0)
        c2 = 0.8;

    alpha = MAX(c1, alpha);
    iter = 1;

    // At least a single iteration is performed
    do
    {
        Tree **trees;
        IntList **tree_adj, *nonRootSeeds;
        bool **are_trees_adj;
        int trees_created, new_numTrees;

        trees_created = -1;
        trees = allocMem(numTrees, sizeof(Tree *));
        tree_adj = allocMem(numTrees, sizeof(IntList *));
        are_trees_adj = allocMem(numTrees, sizeof(bool *));
        nonRootSeeds = createIntList();

// Assign initial values for all nodes
#pragma omp parallel for firstprivate(num_nodes, want_borders) \
    shared(cost_map, label_img, border_img)
        for (int i = 0; i < num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            //pred_map[i] = -1;
            label_img[i] = -1; // 1 rótulo por árvore

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        //int obj = 0, fundo = 0, nrobj = 0, nrfundo = 0;
        // cria uma árvore para cada pixel de cada marcador
        // porém, cada marcador recebe um rótulo distinto
        // e as demais sementes (GRID) recebem um outro rótulo
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index, tree_id;
            seed_index = ptr->elem;
            tree_id = label_seed[seed_index] - 1;
            cost_map[seed_index] = 0;
            label_img[seed_index] = tree_id;

            /* cria uma nova arvore para o primeiro pixel de cada label */
            if (tree_id > trees_created)
            {
                trees[tree_id] = createTree(seed_index, num_feats);
                tree_adj[tree_id] = createIntList();
                are_trees_adj[tree_id] = allocMem(numTrees, sizeof(bool));
                trees_created++;
            }
            else /* usa uma arvore ja existente */
            {
                insertIntListTail(&nonRootSeeds, seed_index);
            }
            insertPrioQueue(&queue, seed_index);
            insertNodeInTree(graph, seed_index, &(trees[tree_id]), grad[seed_index]);
        }

        // For each node within the queue
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            node_label2 = labels_map[node_label];

            if (node_label2 > obj_markers)
            {
                label_img2->val[node_index][0] = 2;
                node_label2 = 2;
            }
            else
            {
                label_img2->val[node_index][0] = 1;
                node_label2 = 1;
            }

            // We insert the features in the respective tree at this
            // moment, because it is guaranteed that this node will not
            // be inserted ever again.
            if (label_seed[node_index] == 0)
                insertNodeInTree(graph, node_index, &(trees[node_label]), grad[node_index]);

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label;
                    double path_cost;

                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];
                    path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], graph->num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            //pred_map[adj_index] = node_index;

                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        // relação de adjacência entre árvores
                        if (node_label != adj_label)
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }

                        int adj_label2 = labels_map[adj_label];
                        if (adj_label2 > obj_markers)
                        {
                            adj_label2 = 2;
                        }
                        else
                        {
                            adj_label2 = 1;
                        }

                        // cria bordas entre rótulos (label2)
                        if (want_borders && ((all_borders == 0 && node_label2 != adj_label2) || (all_borders == 1 && node_label != adj_label)))
                        {
                            // Both depicts a border between their superpixels
                            (*border_img)->val[node_index][0] = 255;
                            (*border_img)->val[adj_index][0] = 255;
                        }
                        /////////////////////////////////////////
                    }
                }
            }
            free(mean_feat_tree);
        }

        /*** SEED SELECTION ***/
        // Auxiliar var
        freeIntList(&seed_set);

        if (stop == 2)
            stop = 1;
        if (iter < iterations && stop == 0)
        {
            // Select the most relevant superpixels
            new_numTrees = numTrees;
            seed_set = seedSelection_clust_bkp(trees, tree_adj, graph->num_nodes, &new_numTrees, num_markers, obj_markers, labels_map, label_seed, &stop, nonRootSeeds);
        }

        iter++;                 // next iter
        resetPrioQueue(&queue); // Clear the queue

#pragma omp parallel for firstprivate(numTrees)
        for (int i = 0; i < numTrees; i++)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);
        freeIntList(&nonRootSeeds);

        numTrees = new_numTrees;

        // o loop termina quando realiza a quantidade definida de iterações
        // ou quando deixa de ter rótulos de fundo (condicao para stop)
    } while (iter <= iterations && stop != 1);

    freeMem(grad);
    freeMem(label_img);
    freeMem(label_seed);
    free(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);
    freeMem(labels_map);

    return label_img2;
}

// compute iDISF using k-means on the scribble pixels
Image *runiDISF_scribbles_clust(Graph *graph,                   // input: image graph in lab color space
                                int n_0,                        // input: desired amount of GRID seeds
                                int iterations,                 // input: maximum iterations or final superpixels (if segm_method=1)
                                Image **border_img,             // input/output: empty image to keep the superpixels borders
                                NodeCoords **coords_user_seeds, // input: coords of the scribble pixels
                                int num_markers,                // input: amount of scribbles
                                int *marker_sizes,              // input: amount of pixels in each scribble
                                int function,                   // input: path-cost function {1 : euclidean, 2 : coefficient gradient, 3 : gradient beta norm., 4 : coefficient gradient norm., 5 : sum. coefficient gradient}
                                int all_borders,                // input: flag. If 1, map border_img with the tree borders
                                double c1, double c2,           // input: parameters of path-cost functions 2-5
                                //int sampling_method,            // unused
                                int obj_markers,
                                Image *image) // input: amount of object scribbles
{
    
    //clock_t time;
    bool want_borders;
    int stop;
    int iter, numTrees, num_cols, num_rows, num_nodes, num_feats;
    int *labels_map; // labels_map : mapeia de tree id para a label de segmentação;
    //int *label_seed; // mapeida de seed index para tree id+1
    //int *pred_map;
    double *cost_map, *grad, alpha;
    NodeAdj *adj_rel;
    IntLabeledList *seed_set;    // lista de sementes
    IntLabeledList *nonRootSeeds; // lista de pixels rotulados pelo usuário que não são raíz
    int *label_img;              // label_img[node_index] = tree_id+1;
    Image *label_img2;           // label_img2->val[node_index][0] = label de segmentação
    PrioQueue *queue, *queue_aux;
    //float size, stride, delta, max_seeds;

    stop = 0;
    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    // Init auxiliary structures
    cost_map = allocMem(graph->num_nodes, sizeof(double)); // f
    //pred_map = allocMem(img->num_pixels, sizeof(int));    // P
    adj_rel = create4NeighAdj();
    label_img = allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, 1);
    queue = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    queue_aux = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    //label_seed = allocMem(num_nodes, sizeof(int));
    want_borders = border_img != NULL;
    nonRootSeeds = createIntLabeledList();

    grad = computeGradient(graph, &alpha);
    //time = clock();
    seed_set = gridSampling_scribbles_clust(image, graph, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, obj_markers, &numTrees, &nonRootSeeds);
    /*time = clock() - time;
    printf("gridSampling %.3f\n", ((double)time) / CLOCKS_PER_SEC);*/

    labels_map = allocMem(numTrees, sizeof(int));

    if (c1 <= 0)
        c1 = 0.7;
    if (c2 <= 0)
        c2 = 0.8;

    alpha = MAX(c1, alpha);
    iter = 1;


    // At least a single iteration is performed
    do
    {
        //time = clock();
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;
        int trees_created, new_numTrees;

        trees_created = -1;
        trees = allocMem(numTrees, sizeof(Tree *));
        tree_adj = allocMem(numTrees, sizeof(IntList *));
        are_trees_adj = allocMem(numTrees, sizeof(bool *));

// Assign initial values for all nodes
#pragma omp parallel for firstprivate(num_nodes, want_borders) \
    shared(cost_map, label_img, border_img)
        for (int i = 0; i < num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            //pred_map[i] = -1;
            label_img[i] = -1; // 1 rótulo por árvore

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        //int obj = 0, fundo = 0, nrobj = 0, nrfundo = 0;
        // cria uma árvore para cada pixel de cada marcador
        // porém, cada marcador recebe um rótulo distinto
        // e as demais sementes (GRID) recebem um outro rótulo
        for (IntLabeledCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index, tree_id;
            seed_index = ptr->elem;
            tree_id = ptr->treeId;
            cost_map[seed_index] = 0;
            label_img[seed_index] = tree_id;
            labels_map[tree_id] = ptr->label;

            /* cria uma nova arvore para o primeiro pixel de cada label */
            trees[tree_id] = createTree(seed_index, num_feats);
            tree_adj[tree_id] = createIntList();
            are_trees_adj[tree_id] = allocMem(numTrees, sizeof(bool));
            trees_created++;
            insertPrioQueue(&queue, seed_index);
            insertPrioQueue(&queue_aux, seed_index);
            insertNodeInTree(graph, seed_index, &(trees[tree_id]), grad[seed_index]);
        }

        //int obj = 0, fundo = 0, nrobj = 0, nrfundo = 0;
        // cria uma árvore para cada pixel de cada marcador
        // porém, cada marcador recebe um rótulo distinto
        // e as demais sementes (GRID) recebem um outro rótulo
        for (IntLabeledCell *ptr = nonRootSeeds->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index, tree_id;
            seed_index = ptr->elem;
            tree_id = ptr->treeId;
            cost_map[seed_index] = 0;
            label_img[seed_index] = tree_id;
            
            /* usa uma arvore ja existente */
            insertPrioQueue(&queue, seed_index);
            insertPrioQueue(&queue_aux, seed_index);
            insertNodeInTree(graph, seed_index, &(trees[tree_id]), grad[seed_index]);
        }
        /*
        time = clock() - time;
        printf("pre IFT %.3f \t", ((double)time) / CLOCKS_PER_SEC);*/

        //time = clock();

        // For each node within the queue_aux (only seeds)
        while (!isPrioQueueEmpty(queue_aux))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue_aux);
            removePrioQueueElem(&queue, node_index);

            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            
            if(labels_map[node_label] > obj_markers)
                node_label2 = 2;
            else
                node_label2 = 1;
            label_img2->val[node_index][0] = node_label2;

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label;
                    double path_cost;

                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];
                    path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], graph->num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            label_img2->val[adj_index][0] = node_label2;
                            
                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        // relação de adjacência entre árvores
                        if (node_label != adj_label)
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }

                        int adj_label2;
                        if(labels_map[adj_label] > obj_markers)
                            adj_label2 = 2;
                        else
                            adj_label2 = 1;

                        // cria bordas entre rótulos (label2)
                        if (want_borders && ((all_borders == 0 && node_label2 != adj_label2) || (all_borders == 1 && node_label != adj_label)))
                        {
                            // Both depicts a border between their superpixels
                            (*border_img)->val[node_index][0] = 255;
                            (*border_img)->val[adj_index][0] = 255;
                        }
                        /////////////////////////////////////////
                    }
                }
            }
            free(mean_feat_tree);
        }

        
        // For each node within the queue
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            
            if(labels_map[node_label] > obj_markers)
                node_label2 = 2;
            else
                node_label2 = 1;
            label_img2->val[node_index][0] = node_label2;

            // We insert the features in the respective tree at this
            // moment, because it is guaranteed that this node will not
            // be inserted ever again.
            insertNodeInTree(graph, node_index, &(trees[node_label]), grad[node_index]);

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label;
                    double path_cost;

                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];
                    path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], graph->num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            label_img2->val[adj_index][0] = node_label2;
                            //pred_map[adj_index] = node_index;

                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        // relação de adjacência entre árvores
                        if (node_label != adj_label)
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }

                        int adj_label2;
                        if(labels_map[adj_label] > obj_markers)
                            adj_label2 = 2;
                        else
                            adj_label2 = 1;
                        
                        // cria bordas entre rótulos (label2)
                        if (want_borders && ((all_borders == 0 && node_label2 != adj_label2) || (all_borders == 1 && node_label != adj_label)))
                        {
                            // Both depicts a border between their superpixels
                            (*border_img)->val[node_index][0] = 255;
                            (*border_img)->val[adj_index][0] = 255;
                        }
                        /////////////////////////////////////////
                    }
                }
            }
            free(mean_feat_tree);
        }

        //time = clock() - time;
        //printf("IFT %.3f \t", ((double)time) / CLOCKS_PER_SEC);

        /*** SEED SELECTION ***/
        // Auxiliar var
        new_numTrees = seed_set->size;
        freeIntLabeledList(&seed_set);

        if (stop == 2)
            stop = 1;
        if (iter < iterations && stop == 0)
        {
            // Select the most relevant superpixels
            //time = clock();
            seed_set = seedSelection_clust(trees, tree_adj, graph->num_nodes, &new_numTrees, num_markers, obj_markers, &stop, nonRootSeeds, labels_map);
            //time = clock() - time;
            //printf("seed removal %.3f\n", ((double)time) / CLOCKS_PER_SEC);
        }

        iter++;                 // next iter
        resetPrioQueue(&queue); // Clear the queue
        resetPrioQueue(&queue_aux);

#pragma omp parallel for firstprivate(numTrees) \
        shared(trees, tree_adj, are_trees_adj)
        for (int i = 0; i < numTrees; i++)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);
        
        numTrees = new_numTrees;

        // o loop termina quando realiza a quantidade definida de iterações
        // ou quando deixa de ter rótulos de fundo (condicao para stop)
    } while (iter <= iterations && stop != 1);

    freeMem(grad);
    freeMem(label_img);
    free(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntLabeledList(&seed_set);
    freeIntLabeledList(&nonRootSeeds);
    freePrioQueue(&queue);
    freePrioQueue(&queue_aux);

    return label_img2;
}

Image *runiDISF(Graph *graph, // input: image graph in lab color space
                int n_0,      // input: desired amount of GRID seeds
                int n_f,
                Image **border_img,             // input/output: empty image to keep the superpixels borders
                NodeCoords **coords_user_seeds, // input: coords of the scribble pixels
                int num_user_seeds,             // input: amount of scribbles
                int *marker_sizes,              // input: amount of pixels in each scribble
                int function,                   // input: path-cost function {1 : euclidean, 2 : coefficient gradient, 3 : gradient beta norm., 4 : coefficient gradient norm., 5 : sum. coefficient gradient}
                int all_borders,                // input: flag. If 1, map border_img with the tree borders
                double c1, double c2,           // input: parameters of path-cost functions 2-5
                int num_objmarkers)
{
    bool want_borders;
    int num_rem_seeds, iter, num_cols, num_rows, num_nodes, num_feats;
    //int *pred_map;
    double *cost_map, *grad, alpha;
    NodeAdj *adj_rel;
    IntList *seed_set; // passa a ser recebido na função para ser reenviado na próxima interação
    int *label_img;
    Image *label_img2;
    PrioQueue *queue;
    int scribbled_seeds = 0;

    // mínimo de árvores é a quantidade de pixels marcados de objeto
    //n_f += marker_sizes[0];
    for (int i = 0; i < num_user_seeds; i++)
    {
        n_f += marker_sizes[i];
        scribbled_seeds += marker_sizes[i];
    }

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_feats = graph->num_feats;

    // Init auxiliary structures
    cost_map = allocMem(num_nodes, sizeof(double)); // f
    //pred_map = allocMem(num_nodes, sizeof(int));    // P
    adj_rel = create4NeighAdj();
    label_img = allocMem(num_nodes, sizeof(int));
    label_img2 = createImage(num_rows, num_cols, 1);
    queue = createPrioQueue(num_nodes, cost_map, MINVAL_POLICY);
    want_borders = border_img != NULL;

    grad = computeGradient(graph, &alpha);
    seed_set = gridSampling(num_cols, num_rows, &n_0, coords_user_seeds, num_user_seeds, marker_sizes, grad);

    if (c1 <= 0)
        c1 = 0.7;
    if (c2 <= 0)
        c2 = 0.8;

    alpha = MAX(c1, alpha);
    iter = 1;

    // At least a single iteration is performed
    do
    {
        int seed_label, num_trees, num_maintain;
        int seed_label2;
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;

        trees = allocMem(seed_set->size, sizeof(Tree *));
        tree_adj = allocMem(seed_set->size, sizeof(IntList *));
        are_trees_adj = allocMem(seed_set->size, sizeof(bool *));

// Assign initial values for all nodes
#pragma omp parallel for firstprivate(num_nodes, want_borders) \
    shared(cost_map, label_img, label_img2, border_img)
        for (int i = 0; i < num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            //pred_map[i] = -1;
            label_img[i] = -1;          // 1 rótulo por árvore
            label_img2->val[i][0] = -1; // rotula como será a segmentação de saída

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        // cria uma árvore para cada pixel de cada marcador
        // porém, o primeiro marcador (foreground) recebe um rótulo
        // e os demais marcadores e demais sementes criadas recebem outro rótulo (background)

        for (int i = 0; i < num_objmarkers; i++)
        {
            for (int j = 0; j < marker_sizes[i]; j++)
            {
                //int node_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
                int node_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;
                label_img2->val[node_index][0] = 1;
            }
        }

        for (int i = num_objmarkers; i < num_user_seeds; i++)
        {
            for (int j = 0; j < marker_sizes[i]; j++)
            {
                //int node_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
                int node_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;
                label_img2->val[node_index][0] = 2;
            }
        }

        seed_label = 0;
        seed_label2 = 2;

        // Assign initial values for all the seeds sampled
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        { // cria todas as árvores e as inicializa

            int seed_index;
            seed_index = ptr->elem;
            cost_map[seed_index] = 0;

            label_img[seed_index] = seed_label;
            trees[seed_label] = createTree(seed_index, num_feats);
            tree_adj[seed_label] = createIntList();
            are_trees_adj[seed_label] = allocMem(seed_set->size, sizeof(bool));

            if (label_img2->val[seed_index][0] == -1)
            {
                label_img2->val[seed_index][0] = seed_label2;
            }
            else
            {
                trees[seed_label]->minDist_userSeed = 0;
            }
            if (num_user_seeds == 0)
                trees[seed_label]->minDist_userSeed = 1;

            seed_label++;
            insertPrioQueue(&queue, seed_index);
        }

        // For each node within the queue
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label, node_label2;
            NodeCoords node_coords;
            float *mean_feat_tree;
            double coef_variation_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(num_cols, node_index);
            node_label = label_img[node_index];
            node_label2 = label_img2->val[node_index][0];

            // store the min distance between tree and the most closer object seed
            float min_Dist = trees[node_label]->minDist_userSeed;
            if (min_Dist > 0)
            {
                for (int j = 0; j < marker_sizes[0]; j++)
                {
                    float dist = euclDistanceCoords(node_coords, coords_user_seeds[0][j]);
                    if (dist < min_Dist)
                        trees[node_label]->minDist_userSeed = dist;
                }
            }

            insertNodeInTree(graph, node_index, &(trees[node_label]), grad[node_index]);

            // Speeding purposes
            mean_feat_tree = meanTreeFeatVector(trees[node_label]);
            coef_variation_tree = coefTreeVariation(trees[node_label]);

            // For each adjacent node
            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;
                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                // Is valid?
                if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                {
                    int adj_index, adj_label /*, adj_label2*/;

                    //adj_index = getNodeIndex(num_cols, adj_coords);
                    adj_index = adj_coords.y * num_cols + adj_coords.x;
                    adj_label = label_img[adj_index];

                    // This adjacent was already added to a tree?
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        double path_cost;
                        path_cost = calcPathCost(mean_feat_tree, graph->feats[adj_index], num_feats, cost_map[node_index], trees[node_label]->num_nodes, grad[adj_index], coef_variation_tree, alpha, c2, function);

                        //printf("Can this node be conquered by the current tree?\n");
                        // Can this node be conquered by the current tree?
                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img[adj_index] = node_label;
                            label_img2->val[adj_index][0] = node_label2;
                            //pred_map[adj_index] = node_index;

                            // Update if it is already in the queue
                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else
                    {
                        if (node_label != adj_label) // If they differ, their trees are adjacent
                        {
                            // Were they defined as adjacents?
                            if (!are_trees_adj[node_label][adj_label])
                            {
                                insertIntListTail(&(tree_adj[node_label]), adj_label);
                                are_trees_adj[node_label][adj_label] = true;
                            }

                            if (!are_trees_adj[adj_label][node_label])
                            {
                                insertIntListTail(&(tree_adj[adj_label]), node_label);
                                are_trees_adj[adj_label][node_label] = true;
                            }
                        }
                        if (((all_borders == 0) && (node_label2 != label_img2->val[adj_index][0])) || ((all_borders == 1) && (node_label != adj_label))) // If they differ, their trees are adjacent
                        {
                            if (want_borders) // Both depicts a border between their superpixels
                            {
                                (*border_img)->val[node_index][0] = 255;
                                (*border_img)->val[adj_index][0] = 255;
                            }
                        }
                    }
                }
            }
            free(mean_feat_tree);
        }

        /*** SEED SELECTION ***/
        // Compute the number of seeds to be preserved
        num_maintain = MAX((n_0 + scribbled_seeds) * exp(-iter), n_f);

        // Auxiliar var
        num_trees = seed_set->size;
        freeIntList(&seed_set);

        // Select the most relevant superpixels
        seed_set = selectKMostRelevantSeeds(trees, tree_adj, num_nodes, num_trees, num_maintain, num_user_seeds);

        // Compute the number of seeds to be removed
        num_rem_seeds = num_trees - seed_set->size;
        iter++;
        resetPrioQueue(&queue); // Clear the queue

#pragma omp parallel for firstprivate(num_trees)
        for (int i = 0; i < num_trees; ++i)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);

    } while (num_rem_seeds > 0);

    freeMem(grad);
    freeMem(label_img);
    freeMem(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);

    return label_img2;
}

Image *runLabeledDISF(Graph *graph, int n_0, int n_f, NodeCoords **coords_user_seeds, Image **border_img)
{
    bool want_borders;
    int num_rem_seeds, iter;
    double *cost_map;
    NodeAdj *adj_rel;
    IntList *seed_set;
    Image *label_img;
    //Image *label_img2;
    PrioQueue *queue;

    // Aux
    cost_map = (double *)calloc(graph->num_nodes, sizeof(double));
    // adj_rel = create4NeighAdj();
    adj_rel = create8NeighAdj();
    label_img = createImage(graph->num_rows, graph->num_cols, 1);
    queue = createPrioQueue(graph->num_nodes, cost_map, MINVAL_POLICY);

    want_borders = border_img != NULL;

    seed_set = gridSamplingDISF(graph, n_0);

    iter = 1; // At least a single iteration is performed
    do
    {
        int seed_label, num_trees, num_maintain;
        Tree **trees;
        IntList **tree_adj;
        bool **are_trees_adj;

        trees = (Tree **)calloc(seed_set->size, sizeof(Tree *));
        tree_adj = (IntList **)calloc(seed_set->size, sizeof(IntList *));
        are_trees_adj = (bool **)calloc(seed_set->size, sizeof(bool *));

// Initialize values
#pragma omp parallel for
        for (int i = 0; i < graph->num_nodes; i++)
        {
            cost_map[i] = INFINITY;
            label_img->val[i][0] = -1;

            if (want_borders)
                (*border_img)->val[i][0] = 0;
        }

        seed_label = 0;
        for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
        {
            int seed_index;

            seed_index = ptr->elem;

            cost_map[seed_index] = 0;
            label_img->val[seed_index][0] = seed_label;

            trees[seed_label] = createTree(seed_index, graph->num_feats);
            tree_adj[seed_label] = createIntList();
            are_trees_adj[seed_label] = (bool *)calloc(seed_set->size, sizeof(bool));

            seed_label++;
            insertPrioQueue(&queue, seed_index);
        }

        // IFT algorithm
        while (!isPrioQueueEmpty(queue))
        {
            int node_index, node_label;
            NodeCoords node_coords;
            float *mean_feat_tree;

            node_index = popPrioQueue(&queue);
            node_coords = getNodeCoords(graph->num_cols, node_index);
            node_label = label_img->val[node_index][0];

            // This node won't appear here ever again
            insertNodeInTree(graph, node_index, &(trees[node_label]), 1);

            mean_feat_tree = meanTreeFeatVector(trees[node_label]);

            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;

                adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, i);

                if (areValidNodeCoords(graph->num_rows, graph->num_cols, adj_coords))
                {
                    int adj_index, adj_label;

                    adj_index = getNodeIndex(graph->num_cols, adj_coords);
                    adj_label = label_img->val[adj_index][0];

                    // If it wasn't inserted nor orderly removed from the queue
                    if (queue->state[adj_index] != BLACK_STATE)
                    {
                        double arc_cost, path_cost;

                        arc_cost = euclDistance(mean_feat_tree, graph->feats[adj_index], graph->num_feats);

                        path_cost = MAX(cost_map[node_index], arc_cost);

                        if (path_cost < cost_map[adj_index])
                        {
                            cost_map[adj_index] = path_cost;
                            label_img->val[adj_index][0] = node_label;

                            if (queue->state[adj_index] == GRAY_STATE)
                                moveIndexUpPrioQueue(&queue, adj_index);
                            else
                                insertPrioQueue(&queue, adj_index);
                        }
                    }
                    else if (node_label != adj_label) // Their trees are adjacent
                    {
                        if (want_borders) // Both depicts a border between their superpixels
                        {
                            (*border_img)->val[node_index][0] = 255;
                            (*border_img)->val[adj_index][0] = 255;
                        }

                        if (!are_trees_adj[node_label][adj_label])
                        {
                            insertIntListTail(&(tree_adj[node_label]), adj_label);
                            insertIntListTail(&(tree_adj[adj_label]), node_label);
                            are_trees_adj[adj_label][node_label] = true;
                            are_trees_adj[node_label][adj_label] = true;
                        }
                    }
                }
            }

            free(mean_feat_tree);
        }

        num_maintain = MAX(n_0 * exp(-iter), n_f);

        // Aux
        num_trees = seed_set->size;
        freeIntList(&seed_set);

        seed_set = selectSeedDISF(trees, tree_adj, graph->num_nodes, num_trees, num_maintain);

        num_rem_seeds = num_trees - seed_set->size;

        iter++;
        resetPrioQueue(&queue);

        for (int i = 0; i < num_trees; ++i)
        {
            freeTree(&(trees[i]));
            freeIntList(&(tree_adj[i]));
            free(are_trees_adj[i]);
        }
        free(trees);
        free(tree_adj);
        free(are_trees_adj);
    } while (num_rem_seeds > 0);

    int obj_index = getNodeIndex(graph->num_cols, coords_user_seeds[0][0]);
    int tree_id_obj = label_img->val[obj_index][0];

    for (int i = 0; i < graph->num_nodes; i++)
    {
        if (label_img->val[i][0] == tree_id_obj)
        {
            label_img->val[i][0] = 1;
        }
        else
            label_img->val[i][0] = 2;
        (*border_img)->val[i][0] = 0;
    }

    for (int i = 0; i < graph->num_nodes; i++)
    {
        int node_label = label_img->val[i][0];
        NodeCoords node_coords = getNodeCoords(graph->num_cols, i);

        for (int j = 0; j < adj_rel->size; j++)
        {
            NodeCoords adj_coords;
            adj_coords = getAdjacentNodeCoords(adj_rel, node_coords, j);

            if (areValidNodeCoords(graph->num_rows, graph->num_cols, adj_coords))
            {
                int adj_index, adj_label;
                adj_index = getNodeIndex(graph->num_cols, adj_coords);
                adj_label = label_img->val[adj_index][0];

                if (node_label != adj_label && want_borders)
                {
                    (*border_img)->val[i][0] = 255;
                    (*border_img)->val[adj_index][0] = 255;
                }
            }
        }
    }

    free(cost_map);
    freeNodeAdj(&adj_rel);
    freeIntList(&seed_set);
    freePrioQueue(&queue);

    return label_img;
}

//=============================================================================
// IntList* Functions
//=============================================================================
IntList *gridSampling_scribbles(int num_rows, int num_cols, int *n_0, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, double *grad, int *labels_map, int obj_markers)
{
    float size, stride, delta_x, delta_y;
    int num_seeds, num_nodes;
    bool *label_seed;
    IntList *seed_set;
    NodeAdj *adj_rel;
    int label2;

    num_seeds = (*n_0);
    num_nodes = num_rows * num_cols;
    seed_set = createIntList();
    adj_rel = create8NeighAdj();
    label_seed = allocMem(num_nodes, sizeof(bool));

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)num_nodes / ((float)num_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;
    num_seeds = 0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    // mark all obj seed position true in vector
    for (int i = 0; i < num_markers; i++)
    {
        for (int j = 0; j < marker_sizes[i]; j++)
        {
            int seed_index;

            //seed_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
            seed_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;

            if (!label_seed[seed_index])
            {
                label_seed[seed_index] = true;
                labels_map[num_seeds] = i + 1;
                insertIntListTail(&seed_set, seed_index);
                num_seeds++;
            }
        }
    }

    //label2 = obj_markers + 1;
    label2 = num_markers + 1;
    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index;

                curr_coords.x = x;
                curr_coords.y = y;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if (label_seed[index])
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            j = num_cols + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    //min_grad_index = getNodeIndex(num_cols, curr_coords);
                    min_grad_index = curr_coords.y * num_cols + curr_coords.x;

                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        // Is valid?
                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            //adj_index = getNodeIndex(num_cols, adj_coords);
                            adj_index = adj_coords.y * num_cols + adj_coords.x;

                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    // Select the position with lowest gradient
                    if (label_seed[min_grad_index])
                    {
                        //min_grad_index = getNodeIndex(num_cols, curr_coords);
                        min_grad_index = curr_coords.y * num_cols + curr_coords.x;
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        labels_map[num_seeds] = label2;
                        num_seeds++;
                    }
                    else
                    {
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        labels_map[num_seeds] = label2;
                        num_seeds++;
                    }
                }
            }
        }
    }

    *n_0 = num_seeds;
    freeNodeAdj(&adj_rel);
    freeMem(label_seed);
    return seed_set;
}

IntList *gridSampling_scribbles_clust_bkp(Graph *graph, int *n_0, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, double *grad, int *labels_map, int obj_markers, int *label_seed, int *numTrees)
{
    float size, stride, delta_x, delta_y;
    int num_seeds, tree_id, numFeats;
    IntList *seed_set;
    NodeAdj *adj_rel, *adj_rel4;
    PrioQueue *queue;
    int num_cols, num_rows;
    double *memb_allnodes;
    bool *map_coords; // set if a index is valid / non-duplicate

    num_seeds = (*n_0);
    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    numFeats = graph->num_feats;

    seed_set = createIntList();
    adj_rel = create8NeighAdj();
    adj_rel4 = create4NeighAdj();

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)graph->num_nodes / ((float)num_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;

    num_seeds = (*numTrees);
    num_seeds = 0;
    tree_id = 0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    memb_allnodes = allocMem(num_cols * num_rows, sizeof(double));
    queue = createPrioQueue(num_cols * num_rows, memb_allnodes, MINVAL_POLICY);
    map_coords = allocMem(num_cols * num_rows, sizeof(bool));

    for (int marker = 0; marker < num_markers; marker++)
    {
        int len, /*numClusters,*/ numObjSeeds, real_numObjSeeds, label, i, node_index, *membership;
        float **objects;

        numObjSeeds = marker_sizes[marker];

        /*if (marker < obj_markers) label = marker + 1;
        else label = obj_markers + 1;*/
        label = marker + 1;

        if (numObjSeeds == 1)
        {
            node_index = coords_user_seeds[marker][0].y * num_cols + coords_user_seeds[marker][0].x;
            labels_map[tree_id] = label;
            tree_id++;
            label_seed[node_index] = tree_id;
            insertIntListTail(&seed_set, node_index);
            num_seeds++;
        }
        else
        {

            // allocate space for objects[][] and read all objects
            len = numObjSeeds * (numFeats + 2);
            objects = (float **)malloc(numObjSeeds * sizeof(float *));
            if (objects == NULL)
                printError("gridSampling_scribbles_clust", "Was do not possible to alloc objects.");
            objects[0] = (float *)malloc(len * sizeof(float));
            if (objects == NULL)
                printError("gridSampling_scribbles_clust", "Was do not possible to alloc objects[0].");
            for (int i = 1; i < numObjSeeds; i++)
                objects[i] = objects[i - 1] + numFeats + 2;

            /* start the core computation -------------------------------------------*/
            /* membership: the cluster id for each data object */
            membership = (int *)malloc(numObjSeeds * sizeof(int));
            if (membership == NULL)
                printError("gridSampling_scribbles_clust", "Was do not possible to alloc membership.");

            /* get the colors of the object seeds */
            // não pode ser paralelizado, pois pode acessar o mesmo node_index em iteraçoes diferentes
            real_numObjSeeds = 0;
            for (i = 0; i < numObjSeeds; i++)
            {
                //node_index = getNodeIndex(num_cols, coords_user_seeds[marker][i]);
                node_index = coords_user_seeds[marker][i].y * num_cols + coords_user_seeds[marker][i].x;
                if (!map_coords[node_index])
                {
                    for (int j = 0; j < numFeats; j++)
                        objects[real_numObjSeeds][j] = graph->feats[node_index][j];
                    objects[real_numObjSeeds][numFeats] = (float)(coords_user_seeds[marker][i].x);
                    objects[real_numObjSeeds][numFeats + 1] = (float)(coords_user_seeds[marker][i].y);
                    map_coords[node_index] = true;
                    real_numObjSeeds++;
                }
            }

            /* get the best clusters */
            //numClusters =
            call_kmeans_elbow(objects, real_numObjSeeds, numFeats, membership);

            /* a identificação de um pixel do scribble é o seu índice no grafo */
#pragma omp parallel for private(i)                    \
    firstprivate(real_numObjSeeds, numFeats, num_cols) \
        shared(objects, memb_allnodes, membership)
            for (i = 0; i < real_numObjSeeds; i++)
            {
                //NodeCoords node;
                //node.x = (int)(objects[i][numFeats]);
                //node.y = (int)(objects[i][numFeats+1]);
                //node_index = getNodeIndex(num_cols, node);
                node_index = (int)(objects[i][numFeats + 1]) * num_cols + (int)(objects[i][numFeats]);

                memb_allnodes[node_index] = (double)(membership[i] + 1);
            }

            for (i = 0; i < real_numObjSeeds; i++)
            {
                int cluster;
                //NodeCoords node;
                //node.x = (int)(objects[i][numFeats]);
                //node.y = (int)(objects[i][numFeats+1]);

                //node_index = getNodeIndex(num_cols, node);
                node_index = (int)(objects[i][numFeats + 1]) * num_cols + (int)(objects[i][numFeats]);

                // se o nó nunca entrou na fila, então ele n foi adicionado a um comp.
                if (queue->state[node_index] == WHITE_STATE)
                {
                    // nó raíz
                    insertPrioQueue(&queue, node_index);

                    // map from tree id to segmentation label
                    labels_map[tree_id] = label;
                    cluster = membership[i] + 1;
                    tree_id++;

                    while (!isPrioQueueEmpty(queue))
                    {
                        node_index = popPrioQueue(&queue);
                        label_seed[node_index] = tree_id;
                        insertIntListTail(&seed_set, node_index);
                        num_seeds++;

                        // searching for adjacents
                        for (int k = 0; k < adj_rel4->size; k++)
                        {
                            NodeCoords adj_coords;
                            adj_coords = getAdjacentNodeCoords(adj_rel4, getNodeCoords(num_cols, node_index), k);

                            if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                            {
                                int adj_index, cluster_adj;
                                //adj_index = getNodeIndex(num_cols, adj_coords);
                                adj_index = adj_coords.y * num_cols + adj_coords.x;
                                cluster_adj = (int)memb_allnodes[adj_index];

                                // adiciona os adjacentes de mesmo cluster e evita adicionar o mesmo nó mais de uma vez
                                if (cluster_adj == cluster && queue->state[adj_index] == WHITE_STATE)
                                    insertPrioQueue(&queue, adj_index);
                            }
                        }
                    }
                }
            }

            free(objects[0]);
            free(objects);
            free(membership);
            resetPrioQueue(&queue);
        }
    }

    freePrioQueue(&queue);
    freeNodeAdj(&adj_rel4);
    free(memb_allnodes);
    free(map_coords);

    //int label2 = obj_markers + 1;
    int label2 = num_markers + 1;

    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += (int)stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += (int)stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index, seed_index;

                curr_coords.x = x;
                curr_coords.y = y;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if (label_seed[index])
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            j = num_cols + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    //seed_index = min_grad_index = getNodeIndex(num_cols, curr_coords);
                    seed_index = min_grad_index = curr_coords.y * num_cols + curr_coords.x;

                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        // Is valid?
                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            //adj_index = getNodeIndex(num_cols, adj_coords);
                            adj_index = adj_coords.y * num_cols + adj_coords.x;

                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    labels_map[tree_id] = label2;
                    tree_id++;

                    // Select the position with lowest gradient
                    if (label_seed[min_grad_index] != 0)
                    {
                        label_seed[seed_index] = tree_id;         /* map from seed index to tree id */
                        insertIntListTail(&seed_set, seed_index); // add the seed in fifo
                    }
                    else
                    {
                        label_seed[min_grad_index] = tree_id;         /* map from seed index to tree id */
                        insertIntListTail(&seed_set, min_grad_index); // add the seed in fifo
                    }
                    num_seeds++;
                }
            }
        }
    }

    *n_0 = num_seeds;
    *numTrees = tree_id;
    freeNodeAdj(&adj_rel);

    return seed_set;
}

IntLabeledList *gridSampling_scribbles_clust(Image *image, Graph *graph, int *n_0, NodeCoords **coords_markers, int num_markers, int *marker_sizes, double *grad, int obj_markers, int *numTrees, IntLabeledList **nonRootSeeds_out)
{
    float size, stride, delta_x, delta_y;
    int num_seeds, tree_id;
    IntLabeledList *seed_set, *nonRootSeeds;
    NodeAdj *adj_rel, *adj_rel4;
    PrioQueue *queue;
    int num_cols, num_rows;
    double *M;
    int normalize = 0;
    //clock_t time;

    const int maxClusters = 10;
    double *laplacian; // laplacian matrix
    double *eigenvalues;
    double *eigenvectors;
    double *D;
    float **objects;
    int *membership;
    int max_num_coords = 0;
    int i;

    nonRootSeeds = (*nonRootSeeds_out);

    num_seeds = (*n_0);
    num_cols = image->num_cols;
    num_rows = image->num_rows;

    seed_set = createIntLabeledList();
    adj_rel = create8NeighAdj();
    adj_rel4 = create4NeighAdj();

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)(num_cols * num_rows) / ((float)num_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;

    tree_id = 0;
    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    M = allocMem(num_cols * num_rows, sizeof(double));
    queue = createPrioQueue(num_cols * num_rows, M, MINVAL_POLICY);
    num_seeds = 0;

    for (i=0; i < num_markers; i++)
        max_num_coords = MAX(max_num_coords, marker_sizes[i]);
    
    laplacian = (double *)calloc(max_num_coords * max_num_coords, sizeof(double));

    eigenvalues = (double *)malloc((maxClusters+1) * sizeof(double));
    eigenvectors = (double *)malloc(max_num_coords * (maxClusters+1) * sizeof(double));        

    objects = (float **)malloc(max_num_coords * sizeof(float *));
    if (objects == NULL)
        printError("clusterPoints_normalized", "Was do not possible to alloc objects.");

    membership = (int *)malloc(max_num_coords * sizeof(int));
    if (membership == NULL)
        printError("gridSampling_scribbles_clust", "Was do not possible to alloc membership.");

#pragma omp parallel for private(i) \
    firstprivate(maxClusters, max_num_coords) \
    shared(objects)
    for (i = 0; i < max_num_coords; i++)
        objects[i] = (float *)malloc(maxClusters * sizeof(float));
    

    for (i = 0; i < num_markers; i++)
    {
        //time = clock();
        int num_coords;
        int n_eigenValues/*, numClusters*/;
        int n_features;
        int j;

        num_coords = marker_sizes[i]; // get the total number of marker coords

        // Quantidade minima de clusters: 2
        // Quantidade maxima de eigenvalues: num_points - 1 -> para k=2 sao necessarios, pelo menos, 3 pontos
        // Quantidade minima de eigenvalues: k+1
        if (num_coords < 3) 
        {
            insertIntLabeledListTail(&seed_set, getNodeIndex(num_cols, coords_markers[i][0]), i+1, tree_id);
            tree_id++;
            num_seeds++;
        }
        else
        {
            n_eigenValues = MIN(maxClusters+1, num_coords - 1); // spectra framework has a limitation of maximum num_coords-1 eigenvalues/eigenvectors
            //laplacian = (double *)calloc(num_coords * num_coords, sizeof(double));

            switch (normalize)
            {
            case 0:
                laplacian = laplacian_unnormalized(coords_markers[i], num_coords, max_num_coords, image, graph);
                break;
            case 1:
                laplacian = laplacian_normalized_sym(coords_markers[i], num_coords, max_num_coords, image);
                break;
            case 2:
                laplacian = laplacian_normalized_rw(coords_markers[i], num_coords, max_num_coords, image);
                break;
            default:
                printError("clusterPoints", "Invalid normalize option.");
                break;
            }
            
            //time = clock() - time;
            //printf("\n laplacian %.3f \t", ((double)time) / CLOCKS_PER_SEC);
            //time = clock();
            

            // spectra returns from higher to small eigenvalue ordering)
            // eigenvectors : an matrix (in a unique vector) num_coords x n_eigenValues,
            // in which each line is a point and its columns are the features
            smallest_eigenvalues(laplacian, num_coords, n_eigenValues, eigenvalues, eigenvectors);

            
            //time = clock() - time;
            //printf("eigenvalues and vectors %.3f \t", ((double)time) / CLOCKS_PER_SEC);
            //time = clock();
            

            n_features = n_eigenValues - 1; // we do not use the last eigenvector (whose with the smallest eigenvalue -- always 0)
            
            /*
            printf("\n\nEIGENVALUES: ");
            for (int j = 0; j < n_eigenValues; j++) printf("%f \t", eigenvalues[j]);
        
            printf("\n\nEIGENVECTORS: ");
            for (int j = 0; j < num_coords * n_eigenValues; j++)
            {
                if (j % n_eigenValues == 0) printf("\n");
                printf("%f ", eigenvectors[j]);
            }
            */
            
            switch (normalize)
            {
            case 0:
            #pragma omp parallel for private(j) \
                firstprivate(num_coords, n_features, n_eigenValues) \
                shared(objects, eigenvectors)
                for (j = 0; j < num_coords; j++)
                {
                    for (int k = 0; k < n_features; k++)
                    {
                        objects[j][k] = (float)(eigenvectors[j * n_eigenValues + (n_features - 1 - k)]);
                        // the object j is the positions [j,{k_0, .., k_max}] of the eigenvectors matrix
                        // we also invert the columns (features) positions to obtain an ascending order concerning to its eigenvalues
                    }
                }
                break;
            case 1:
                D = (double *)calloc(num_coords, sizeof(double)); // store the sum of eigenvectors[i][k]^2 for k=[0,..,n_eigenValues] to normalize it
            #pragma omp parallel for private(j) \
                firstprivate(n_eigenValues) \
                shared(eigenvectors, D)
                for (j = 0; j < num_coords; j++)
                {
                    double tmp = 0;
                    for (int k = 0; k < n_eigenValues; k++)
                    {
                        double tmp2 = eigenvectors[j * n_eigenValues + k];
                        tmp += tmp2 * tmp2;
                    }
                    D[j] = sqrt(tmp);
                }

                //printf("\n\n NORMALIZED EIGENVECTORS: \n");
            #pragma omp parallel for private(j) \
                firstprivate(num_coords, n_features, n_eigenValues) \
                shared(objects, eigenvectors, D)
                for (j = 0; j < num_coords; j++)
                {
                    for (int k = 0; k < n_features; k++)
                    {
                        objects[j][k] = (float)(eigenvectors[j * n_eigenValues + (n_features - 1 - k)] / D[j]);
                        //printf("%f ", objects[j][k]/D[j]);
                        // the object j is the positions [j,{k_0, .., k_max}] of the eigenvectors matrix
                        // we also invert the columns (features) positions to obtain an ascending order concerning to its eigenvalues
                    }
                    //printf("\n");
                }
                free(D);
                break;
            case 2:
            #pragma omp parallel for private(j) \
                firstprivate(num_coords, n_features, n_eigenValues) \
                shared(objects, eigenvectors)
                for (j = 0; j < num_coords; j++)
                {
                    for (int k = 0; k < n_features; k++)
                    {
                        objects[j][k] = (float)(eigenvectors[j * n_eigenValues + (n_features - 1 - k)]);
                        //printf("%f ", objects[j][k]);
                        // the object j is the positions [j,{k_0, .., k_max}] of the eigenvectors matrix
                        // we also invert the columns (features) positions to obtain an ascending order concerning to its eigenvalues
                    }
                    //printf("\n");
                }
                break;
            default:
                printError("clusterPoints", "Invalid normalize option.");
                break;
            }

            /*
            time = clock() - time;
            printf("copying values %.3f \t", ((double)time) / CLOCKS_PER_SEC);
            time = clock();
            */

            //numClusters = call_kmeans_elbow(objects, num_coords, n_features, membership);
            /*numClusters = */call_kmeans_silhouette(objects, num_coords, n_features, membership);
            //printf("\n numClusters = %d\n", numClusters);

            
            //time = clock() - time;
            //printf("kmeans %.3f \t", ((double)time) / CLOCKS_PER_SEC);
            //time = clock();
            

            /* a identificação de um pixel do scribble é o seu índice no grafo */
    #pragma omp parallel for private(j)                    \
        firstprivate(num_coords, num_cols) \
        shared(coords_markers, membership, M)
            for (j = 0; j < num_coords; j++)
            {
                M[getNodeIndex(num_cols, coords_markers[i][j])] = (double)(membership[j] + 1);
            }

            for (j = 0; j < num_coords; j++)
            {
                int cluster;
                int node_index = getNodeIndex(num_cols, coords_markers[i][j]);

                // se o nó nunca entrou na fila, então ele n foi adicionado a um comp.
                if (queue->state[node_index] == WHITE_STATE)
                {
                    // nó raíz
                    insertPrioQueue(&queue, node_index);

                    // map from tree id to segmentation label
                    int label = i+1;
                    cluster = membership[j]+1;
                    insertIntLabeledListTail(&seed_set, node_index, label, tree_id);
                    tree_id++;
                    num_seeds++;

                    while (!isPrioQueueEmpty(queue))
                    {
                        node_index = popPrioQueue(&queue);
                        
                        // searching for adjacents
                        for (int k = 0; k < adj_rel4->size; k++)
                        {
                            NodeCoords adj_coords;
                            adj_coords = getAdjacentNodeCoords(adj_rel4, getNodeCoords(num_cols, node_index), k);

                            if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                            {
                                int adj_index, cluster_adj;
                                //adj_index = getNodeIndex(num_cols, adj_coords);
                                adj_index = adj_coords.y * num_cols + adj_coords.x;
                                cluster_adj = (int)(M[adj_index]);

                                // adiciona os adjacentes de mesmo cluster e evita adicionar o mesmo nó mais de uma vez
                                if (cluster_adj == cluster && queue->state[adj_index] == WHITE_STATE){
                                    insertPrioQueue(&queue, adj_index);
                                    insertIntLabeledListTail(&nonRootSeeds, adj_index, label, tree_id-1);
                                    num_seeds++;
                                }
                            }
                        }
                    }
                }
            }
            
            //time = clock() - time;
            //printf("connected components %.3f \n", ((double)time) / CLOCKS_PER_SEC);
            
            resetPrioQueue(&queue);
        }
    }
    
    free(eigenvectors);
    free(eigenvalues);
    free(laplacian);

#pragma omp parallel for private(i)                    \
    firstprivate(max_num_coords) \
    shared(objects)
    for (i = 0; i < max_num_coords; i++)
        free(objects[i]);

    free(objects);
    free(membership);
    freePrioQueue(&queue);
    freeNodeAdj(&adj_rel4);

    //int label2 = obj_markers + 1;
    int label2 = num_markers + 1;

    //time = clock();
    
    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += (int)stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += (int)stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index, seed_index;

                curr_coords.x = x;
                curr_coords.y = y;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if ((int)(M[index]) != 0)
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            j = num_cols + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    seed_index = min_grad_index = getNodeIndex(num_cols, curr_coords);

                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        // Is valid?
                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            adj_index = getNodeIndex(num_cols, adj_coords);
                            
                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    // Select the position with lowest gradient
                    if ((int)(M[min_grad_index]) != 0)
                    {
                        insertIntLabeledListTail(&seed_set, seed_index, label2, tree_id); // add the seed in fifo
                    }
                    else
                    {
                        insertIntLabeledListTail(&seed_set, min_grad_index, label2, tree_id); // add the seed in fifo
                    }
                    tree_id++;
                    num_seeds++;
                }
            }
        }
    }

    
    //time = clock() - time;
    //printf("grid sampling %.3f \n", ((double)time) / CLOCKS_PER_SEC);
    

    (*nonRootSeeds_out) = nonRootSeeds;
    *n_0 = num_seeds;
    *numTrees = tree_id;
    freeNodeAdj(&adj_rel);
    free(M);

    return seed_set;
}

IntLabeledList *gridSampling_scribbles_1clust(Image *image, Graph *graph, int *n_0, NodeCoords **coords_markers, int num_markers, int *marker_sizes, double *grad, int obj_markers, int *numTrees, IntLabeledList **nonRootSeeds_out)
{
    float size, stride, delta_x, delta_y;
    int num_seeds, tree_id;
    IntLabeledList *seed_set, *nonRootSeeds;
    NodeAdj *adj_rel, *adj_rel4;
    PrioQueue *queue;
    int num_cols, num_rows;
    double *M;
    //int normalize = 0;
    //clock_t time;

    int i;

    nonRootSeeds = (*nonRootSeeds_out);

    num_seeds = (*n_0);
    num_cols = image->num_cols;
    num_rows = image->num_rows;

    seed_set = createIntLabeledList();
    adj_rel = create8NeighAdj();
    adj_rel4 = create4NeighAdj();

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)(num_cols * num_rows) / ((float)num_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;

    tree_id = 0;
    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    M = allocMem(num_cols * num_rows, sizeof(double));
    queue = createPrioQueue(num_cols * num_rows, M, MINVAL_POLICY);
    num_seeds = 0;

    for (i = 0; i < num_markers; i++)
    {
        //time = clock();
        int num_coords;
        int j;

        num_coords = marker_sizes[i]; // get the total number of marker coords

        if (num_coords == 1)
        {
            insertIntLabeledListTail(&seed_set, getNodeIndex(num_cols, coords_markers[i][0]), i+1, tree_id);
            tree_id++;
            num_seeds++;
        }
        else
        {
            
            /*
            time = clock() - time;
            printf("copying values %.3f \t", ((double)time) / CLOCKS_PER_SEC);
            time = clock();
            */

        #pragma omp parallel for private(j)     \
            firstprivate(num_coords, num_cols)  \
            shared(coords_markers, M)
            for (j = 0; j < num_coords; j++)
            {
                M[getNodeIndex(num_cols, coords_markers[i][j])] = 1;
            }
            
            for (j = 0; j < num_coords; j++)
            {
                int cluster;
                int node_index = getNodeIndex(num_cols, coords_markers[i][j]);

                // se o nó nunca entrou na fila, então ele n foi adicionado a um comp.
                if (queue->state[node_index] == WHITE_STATE)
                {
                    // nó raíz
                    insertPrioQueue(&queue, node_index);

                    // map from tree id to segmentation label
                    int label = i+1;
                    cluster = 1;
                    insertIntLabeledListTail(&seed_set, node_index, label, tree_id);
                    tree_id++;
                    num_seeds++;

                    while (!isPrioQueueEmpty(queue))
                    {
                        node_index = popPrioQueue(&queue);
                        
                        // searching for adjacents
                        for (int k = 0; k < adj_rel4->size; k++)
                        {
                            NodeCoords adj_coords;
                            adj_coords = getAdjacentNodeCoords(adj_rel4, getNodeCoords(num_cols, node_index), k);

                            if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                            {
                                int adj_index, cluster_adj;
                                adj_index = getNodeIndex(num_cols, adj_coords);
                                //adj_index = adj_coords.y * num_cols + adj_coords.x;
                                cluster_adj = (int)(M[adj_index]);

                                // adiciona os adjacentes de mesmo cluster e evita adicionar o mesmo nó mais de uma vez
                                if (cluster_adj == cluster && queue->state[adj_index] == WHITE_STATE){
                                    insertPrioQueue(&queue, adj_index);
                                    insertIntLabeledListTail(&nonRootSeeds, adj_index, label, tree_id-1);
                                    num_seeds++;
                                }
                            }
                        }
                    }
                }
            }
            /*
            time = clock() - time;
            printf("connected components %.3f \n", ((double)time) / CLOCKS_PER_SEC);
            */
            resetPrioQueue(&queue);
        }
    }

    //printf("tree id = %d \n", tree_id);
    //printf("num_seeds = %d \n", num_seeds);

    freePrioQueue(&queue);
    freeNodeAdj(&adj_rel4);

    //int label2 = obj_markers + 1;
    int label2 = num_markers + 1;

    //time = clock();
    
    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += (int)stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += (int)stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index, seed_index;

                curr_coords.x = x;
                curr_coords.y = y;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if ((int)(M[index]) != 0)
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            j = num_cols + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    seed_index = min_grad_index = getNodeIndex(num_cols, curr_coords);

                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        // Is valid?
                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            adj_index = getNodeIndex(num_cols, adj_coords);
                            
                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    // Select the position with lowest gradient
                    if ((int)(M[min_grad_index]) != 0)
                    {
                        insertIntLabeledListTail(&seed_set, seed_index, label2, tree_id); // add the seed in fifo
                    }
                    else
                    {
                        insertIntLabeledListTail(&seed_set, min_grad_index, label2, tree_id); // add the seed in fifo
                    }
                    tree_id++;
                    num_seeds++;
                }
            }
        }
    }

    /*
    time = clock() - time;
    printf("grid sampling %.3f \n", ((double)time) / CLOCKS_PER_SEC);
    */

    (*nonRootSeeds_out) = nonRootSeeds;
    *n_0 = num_seeds;
    *numTrees = tree_id;
    freeNodeAdj(&adj_rel);
    free(M);

    return seed_set;
}


// used in iDISF
IntList *gridSampling(int num_cols, int num_rows, int *n_0, NodeCoords **coords_user_seeds, int num_user_seeds, int *marker_sizes, double *grad)
{
    float size, stride, delta_x, delta_y;
    int num_nodes, num_seeds;
    IntList *seed_set;
    NodeAdj *adj_rel;
    bool *label_seed;

    // marcadores em linha não podem ter os pixels mudando de lugar para a posição com menor gradiente
    // porque isso pode desconectar a linha
    // a solução encontrada é só mudar os marcadores unitários (que são pontos)
    // mas é preciso verificar se a nova posição não é outra semente

    num_seeds = (*n_0);
    num_nodes = num_cols * num_rows;
    seed_set = createIntList();
    label_seed = allocMem(num_nodes, sizeof(bool));

    // Compute the approximate superpixel size and stride
    size = 0.5 + (float)(num_nodes / ((float)num_seeds + (float)num_user_seeds));
    stride = sqrtf(size) + 0.5;
    delta_x = delta_y = stride / 2.0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    adj_rel = create8NeighAdj();
    int count = 0;

    // change the user seed to a seed adj with min. gradient
    for (int i = 0; i < num_user_seeds; i++)
    {
        // add all point of the scribbles
        for (int j = 0; j < marker_sizes[i]; j++)
        {
            int node_index = coords_user_seeds[i][j].y * num_cols + coords_user_seeds[i][j].x;
            if (!label_seed[node_index])
            {
                label_seed[node_index] = true;
                insertIntListTail(&seed_set, node_index);
            }
        }
    }

    // Iterate through the nodes coordinates
    if (*n_0 > 0)
    {
        for (int y = (int)delta_y; y < num_rows; y += stride)
        {
            for (int x = (int)delta_x; x < num_cols; x += stride)
            {
                NodeCoords curr_coords;
                bool isUserSeed;
                int min_grad_index, node_index;

                curr_coords.x = x;
                curr_coords.y = y;

                // check if is a user seed or is near to
                isUserSeed = false;
                for (int i = MAX(0, y - (int)delta_y); i <= MIN(num_rows, y + (int)delta_y); i++)
                {
                    for (int j = MAX(0, x - (int)delta_x); j <= MIN(num_cols, x + (int)delta_x); j++)
                    {
                        int index = i * num_cols + j;
                        if (label_seed[index])
                        {
                            isUserSeed = true;
                            i = num_rows + 1;
                            break;
                        }
                    }
                }

                if (!isUserSeed)
                {
                    //min_grad_index = getNodeIndex(num_cols, curr_coords);
                    node_index = min_grad_index = curr_coords.y * num_cols + curr_coords.x;

                    // For each adjacent node
                    for (int i = 0; i < adj_rel->size; i++)
                    {
                        NodeCoords adj_coords;
                        adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                        if (areValidNodeCoords(num_rows, num_cols, adj_coords))
                        {
                            int adj_index;
                            //adj_index = getNodeIndex(num_cols, adj_coords);
                            adj_index = adj_coords.y * num_cols + adj_coords.x;

                            // The gradient in the adjacent is minimum?
                            if (grad[adj_index] < grad[min_grad_index])
                                min_grad_index = adj_index;
                        }
                    }

                    // Select the position with lowest gradient
                    if (label_seed[min_grad_index])
                    {
                        //is_seed[getNodeIndex(num_cols, curr_coords)] = true;
                        label_seed[node_index] = true;
                        insertIntListTail(&seed_set, node_index);
                        count++;
                    }
                    else
                    {
                        label_seed[min_grad_index] = true;
                        insertIntListTail(&seed_set, min_grad_index);
                        count++;
                    }
                }
            }
        }
    }

    *n_0 = count;
    freeMem(label_seed);
    freeNodeAdj(&adj_rel);
    return seed_set;
}

// used in DISF
IntList *gridSamplingDISF(Graph *graph, int num_seeds)
{
    float size, stride, delta_x, delta_y;
    double *grad;
    bool *is_seed;
    IntList *seed_set;
    NodeAdj *adj_rel;

    seed_set = createIntList();
    is_seed = (bool *)calloc(graph->num_nodes, sizeof(bool));

    // Approximate superpixel size
    size = 0.5 + (float)(graph->num_nodes / (float)num_seeds);
    stride = sqrtf(size) + 0.5;

    delta_x = delta_y = stride / 2.0;

    if (delta_x < 1.0 || delta_y < 1.0)
        printError("gridSampling", "The number of samples is too high");

    double variation;
    grad = computeGradient(graph, &variation);
    adj_rel = create8NeighAdj();

    for (int y = (int)delta_y; y < graph->num_rows; y += stride)
    {
        for (int x = (int)delta_x; x < graph->num_cols; x += stride)
        {
            int min_grad_index;
            NodeCoords curr_coords;

            curr_coords.x = x;
            curr_coords.y = y;

            min_grad_index = getNodeIndex(graph->num_cols, curr_coords);

            for (int i = 0; i < adj_rel->size; i++)
            {
                NodeCoords adj_coords;

                adj_coords = getAdjacentNodeCoords(adj_rel, curr_coords, i);

                if (areValidNodeCoords(graph->num_rows, graph->num_cols, adj_coords))
                {
                    int adj_index;

                    adj_index = getNodeIndex(graph->num_cols, adj_coords);

                    if (grad[adj_index] < grad[min_grad_index])
                        min_grad_index = adj_index;
                }
            }

            is_seed[min_grad_index] = true;
        }
    }

    for (int i = 0; i < graph->num_nodes; i++)
        if (is_seed[i]) // Assuring unique values
            insertIntListTail(&seed_set, i);

    free(grad);
    free(is_seed);
    freeNodeAdj(&adj_rel);

    return seed_set;
}

// used in iDISF
IntList *selectKMostRelevantSeeds(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain, int num_user_seeds)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue;

    tree_prio = allocMem(num_trees, sizeof(double));
    rel_seeds = createIntList();
    queue = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    // For each tree
    for (int i = 0; i < num_trees; i++)
    {
        if (trees[i]->minDist_userSeed == 0 && num_user_seeds > 0)
        {
            // Compute the superpixel relevance
            tree_prio[i] = INFINITY;
        }
        else
        {
            double area_prio, grad_prio;
            float *mean_feat_i;

            // Compute the area relevance
            area_prio = ((float)trees[i]->num_nodes * trees[i]->minDist_userSeed) / ((float)num_nodes);

            // Initial values for the computation of gradient relevance
            grad_prio = INFINITY;
            mean_feat_i = meanTreeFeatVector(trees[i]); // Speeding purposes

            // For each adjacent tree
            for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
            {
                int adj_tree_id;
                float *mean_feat_j;
                double dist;

                adj_tree_id = ptr->elem;
                mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);

                // Compute the L2 norm between trees
                dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

                // Get the minimum gradient value
                grad_prio = MIN(grad_prio, dist);

                free(mean_feat_j);
            }

            // Compute the superpixel relevance
            tree_prio[i] = area_prio * grad_prio;
            free(mean_feat_i);
        }
        insertPrioQueue(&queue, i);
    }

    // While it is possible to get relevant seeds
    for (int i = 0; i < num_maintain && !isPrioQueueEmpty(queue); i++)
    {
        int tree_id, root_index;

        tree_id = popPrioQueue(&queue);
        root_index = trees[tree_id]->root_index;
        insertIntListTail(&rel_seeds, root_index);
    }

    // The rest is discarted
    freePrioQueue(&queue);
    freeMem(tree_prio);
    return rel_seeds;
}

IntList *selectSeedDISF(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue;

    tree_prio = (double *)calloc(num_trees, sizeof(double));
    rel_seeds = createIntList();
    queue = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    for (int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio;
        float *mean_feat_i;

        area_prio = trees[i]->num_nodes / (float)num_nodes;

        grad_prio = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]);

        for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id;
            float *mean_feat_j;
            double dist;

            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);

            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            grad_prio = MIN(grad_prio, dist);

            free(mean_feat_j);
        }

        tree_prio[i] = area_prio * grad_prio;

        insertPrioQueue(&queue, i);

        free(mean_feat_i);
    }

    for (int i = 0; i < num_maintain && !isPrioQueueEmpty(queue); i++)
    {
        int tree_id, root_index;

        tree_id = popPrioQueue(&queue);
        root_index = trees[tree_id]->root_index;

        insertIntListTail(&rel_seeds, root_index);
    }

    freePrioQueue(&queue); // The remaining are discarded
    free(tree_prio);

    return rel_seeds;
}

inline int addSeed(int root_index, int label, IntList *rel_seeds, int *new_labels_map, int index_label)
{
    insertIntListTail(&rel_seeds, root_index);
    new_labels_map[index_label] = label;
    index_label++;
    return index_label;
}

inline int superpixelSelectionType1(IntCell *tree_adj_head, int tree_id, int root_index, int *labels_map, int num_markers, IntList *rel_seeds, int *new_labels_map, int index_label)
{
    // se algum vizinho for objeto, não remove a semente
    bool adj_obj = false;
    for (IntCell *ptr = tree_adj_head; ptr != NULL; ptr = ptr->next)
    {
        int adj_tree_id;
        adj_tree_id = ptr->elem;

        /* adj obj or excluded seed */
        if (labels_map[adj_tree_id] <= num_markers)
        {
            adj_obj = true;
            break;
        }
    }
    if (!adj_obj)
        labels_map[tree_id] = 0;
    else // mantém a semente
        index_label = addSeed(root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);

    return index_label;
}

// used in iDISF scribbles
IntList *seedRemoval(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_markers, int num_objmarkers, int *new_labels_map, int *stop)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue1, *queue2, *queue3;
    double mean_relevance13, mean_relevance24; // usaremos como medida de limiar
    int *labels_map;
    int index_label;
    float sum_prio_13, sum_prio_2_13, std_deviation_13;
    float sum_prio_24, sum_prio_2_24, std_deviation_24;
    int type1 = 0, type2 = 0, type3 = 0, type4 = 0;

    index_label = 0;
    mean_relevance13 = 0.0;
    sum_prio_13 = 0;
    sum_prio_2_13 = 0;
    std_deviation_13 = 0;
    mean_relevance24 = 0.0;
    sum_prio_24 = 0;
    sum_prio_2_24 = 0;
    std_deviation_24 = 0;

    tree_prio = allocMem(num_trees, sizeof(double));
    rel_seeds = createIntList();
    labels_map = allocMem(num_trees, sizeof(int));

    // inicia as filas
    queue1 = createPrioQueue(num_trees, tree_prio, MINVAL_POLICY);
    queue3 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);
    queue2 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    // For each tree
    for (int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio, grad_prio_mix;
        float *mean_feat_i;
        //int root_index;
        int adjacents_type; // 0:nothing , 1:only background with background seed , 2:mixed with background seed , 3:only foreground with foreground seed, 4:mixed with foreground seed
        bool background_seed, gridSeed;
        int label;

        //root_index = trees[i]->root_index;
        label = new_labels_map[i];
        labels_map[i] = label;
        background_seed = false;
        gridSeed = false;

        adjacents_type = 0;

        // Compute the area relevance
        area_prio = (double)trees[i]->num_nodes / (double)num_nodes;

        // Initial values for the computation of gradient relevance
        grad_prio = INFINITY;
        grad_prio_mix = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]); // Speeding purposes

        if (label > num_objmarkers)
            background_seed = true;
        if (label > num_markers)
            gridSeed = true;

        // For each adjacent tree
        for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id, adj_label;
            float *mean_feat_j;
            double dist;
            bool adj_background;

            adj_background = false;
            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);
            adj_label = new_labels_map[adj_tree_id];

            // Compute the L2 norm between trees
            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            if (adj_label > num_objmarkers) // labels_map : indice indica labels e valor indica labels2
                adj_background = true;

            // Get the minimum gradient value
            if ((background_seed && !adj_background) || (!background_seed && adj_background))
                grad_prio_mix = MIN(grad_prio_mix, dist);
            else
                grad_prio = MIN(grad_prio, dist);

            if (adjacents_type != 2 && adjacents_type != 4)
            {
                /* os tipos com seed background são 1 e 2. */
                if (background_seed)
                {
                    if (!adj_background)
                        adjacents_type = 2;
                    else
                        adjacents_type = 1;
                }
                /* os tipos com seed foreground são 3 e 4. */
                else
                {
                    if (adj_background)
                        adjacents_type = 4;
                    else
                        adjacents_type = 3;
                }
            }
            free(mean_feat_j);
        }

        // Compute the superpixel relevance
        switch (adjacents_type)
        {
        case 1:
            tree_prio[i] = area_prio * grad_prio;
            type1++;
            if (gridSeed)
                insertPrioQueue(&queue1, i);
            else
                index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;

        case 2:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type2++;
            if (gridSeed)
                insertPrioQueue(&queue2, i);
            else
                index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;

        case 3:
            tree_prio[i] = area_prio * grad_prio;
            type3++;
            //insertPrioQueue(&queue3, i);
            index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;

        case 4:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type4++;
            index_label = addSeed(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;
        }
        free(mean_feat_i);
    }

    std_deviation_13 = sqrtf(fabs(sum_prio_2_13 - 2 * sum_prio_13 + ((sum_prio_13 * sum_prio_13) / (float)type1 + type3)) / (float)type1 + type3);
    mean_relevance13 = sum_prio_13 / (float)(type1 + type3);
    mean_relevance13 += std_deviation_13;

    std_deviation_24 = sqrtf(fabs(sum_prio_2_24 - 2 * sum_prio_24 + ((sum_prio_24 * sum_prio_24) / (float)type2 + type4)) / (float)type2 + type4);
    mean_relevance24 = sum_prio_24 / (float)(type2 + type4);
    mean_relevance24 += std_deviation_24;

    while (!isPrioQueueEmpty(queue1))
    {
        int tree_id;

        tree_id = popPrioQueue(&queue1);
        if (tree_prio[tree_id] >= mean_relevance13)
        {
            index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
            while (!isPrioQueueEmpty(queue1))
            {
                tree_id = popPrioQueue(&queue1);
                index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
            }
        }
        else
        {
            index_label = superpixelSelectionType1(tree_adj[tree_id]->head, tree_id, trees[tree_id]->root_index, labels_map, num_objmarkers, rel_seeds, new_labels_map, index_label);
        }
    }

    int prio = mean_relevance13;
    while (!isPrioQueueEmpty(queue3) && prio >= mean_relevance13)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue3);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance13)
            index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
    }

    prio = mean_relevance24;
    while (!isPrioQueueEmpty(queue2) && prio >= mean_relevance24)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue2);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance24)
            index_label = addSeed(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label);
    }

    bool background = false, foreground = false, gridSeeds = false;
    if (num_markers == 1)
        background = true;
    for (int i = 0; i < index_label; i++)
    {
        if (new_labels_map[i] <= num_objmarkers)
            foreground = true;
        else
        {
            if (new_labels_map[i] > num_markers)
                gridSeeds = true;
            else
                background = true;
        }
        if (foreground && background && gridSeeds)
            break;
    }
    if (num_markers == 1)
        background = false;
    if (!gridSeeds)
        *stop = 2;
    if (!foreground || (!background && !gridSeeds))
        *stop = 1;

    /*
    bool background = false, foreground = false;
    for (int i = 0; i < index_label; i++)
    {
        if (new_labels_map[i] <= num_objmarkers) foreground = true;
        else background = true;
        if (foreground && background) break;
    }
    if (!foreground || !background) *stop = 1;
    */
    freePrioQueue(&queue1);
    freePrioQueue(&queue3);
    freePrioQueue(&queue2);
    freeMem(tree_prio);
    freeMem(labels_map);
    return rel_seeds;
}

inline int addSeed_clust(int root_index, int label, IntLabeledList *rel_seeds, int *new_labels_map, int index_label, int *labels_trees, int old_tree_id)
{
    
    new_labels_map[index_label] = label;
    insertIntLabeledListTail(&rel_seeds, root_index, label, index_label);
    index_label++;
    labels_trees[old_tree_id] = index_label;
    return index_label;
}

inline int addSeed_clust_bkp(int root_index, int label, IntList *rel_seeds, int *new_labels_map, int *label_seed, int index_label)
{
    insertIntListTail(&rel_seeds, root_index);
    new_labels_map[index_label] = label;
    index_label++;
    label_seed[root_index] = index_label;
    return index_label;
}

inline int superpixelSelectionType1_clust(IntCell *tree_adj_head, int tree_id, int root_index, int *labels_map, int num_markers, IntLabeledList *rel_seeds, int *new_labels_map, int index_label, int* labels_trees)
{
    // se algum vizinho for objeto, não remove a semente
    bool adj_obj = false;
    for (IntCell *ptr = tree_adj_head; ptr != NULL; ptr = ptr->next)
    {
        int adj_tree_id;

        adj_tree_id = ptr->elem;
        /* adj obj or excluded seed */
        if (labels_map[adj_tree_id] <= num_markers)
        {
            adj_obj = true;
            break;
        }
    }
    if (!adj_obj)
    {
        labels_map[tree_id] = 0;
    }
    else // mantém a semente
        index_label = addSeed_clust(root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label, labels_trees, tree_id);

    return index_label;
}

inline int superpixelSelectionType1_clust_bkp(IntCell *tree_adj_head, int tree_id, int root_index, int *labels_map, int num_markers, IntList *rel_seeds, int *new_labels_map, int index_label, int *label_seed)
{
    // se algum vizinho for objeto, não remove a semente
    bool adj_obj = false;
    for (IntCell *ptr = tree_adj_head; ptr != NULL; ptr = ptr->next)
    {
        int adj_tree_id;

        adj_tree_id = ptr->elem;
        /* adj obj or excluded seed */
        if (labels_map[adj_tree_id] <= num_markers)
        {
            adj_obj = true;
            break;
        }
    }
    if (!adj_obj)
    {
        labels_map[tree_id] = 0;
        label_seed[root_index] = 0;
    }
    else // mantém a semente
        index_label = addSeed_clust_bkp(root_index, labels_map[tree_id], rel_seeds, new_labels_map, label_seed, index_label);

    return index_label;
}

// used in iDISF scribbles with clustering
IntList *seedSelection_clust_bkp(Tree **trees, IntList **tree_adj, int num_nodes, int *numTrees, int num_markers, int num_objmarkers, int *new_labels_map, int *label_seed, int *stop, IntList *seed_set)
{
    double *tree_prio;
    IntList *rel_seeds;
    PrioQueue *queue1, *queue2, *queue3;
    double mean_relevance13, mean_relevance24; // usaremos como medida de limiar
    int *labels_map;
    int index_label;
    int num_trees;
    //float sum_prio, sum_prio_2, std_deviation;
    float sum_prio_13, sum_prio_2_13, std_deviation_13;
    float sum_prio_24, sum_prio_2_24, std_deviation_24;
    int type1 = 0, type2 = 0, type3 = 0, type4 = 0;

    index_label = 0;
    //mean_relevance = 0.0;  sum_prio = 0; sum_prio_2 = 0; std_deviation = 0;
    mean_relevance13 = 0.0;
    sum_prio_13 = 0;
    sum_prio_2_13 = 0;
    std_deviation_13 = 0;
    mean_relevance24 = 0.0;
    mean_relevance24 = 0.0;
    sum_prio_24 = 0;
    sum_prio_2_24 = 0;
    std_deviation_24 = 0;
    num_trees = (*numTrees);

    tree_prio = allocMem(num_trees, sizeof(double));
    rel_seeds = createIntList();
    labels_map = allocMem(num_trees, sizeof(int));

    // inicia as filas
    queue1 = createPrioQueue(num_trees, tree_prio, MINVAL_POLICY);
    queue2 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);
    queue3 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    // For each tree
    for (int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio, grad_prio_mix;
        float *mean_feat_i;
        int root_index;
        int adjacents_type = 0; // 0:nothing , 1:only background with background seed , 2:mixed with background seed , 3:only foreground with foreground seed, 4:mixed with foreground seed
        bool background_seed, gridSeeds;
        int label;

        root_index = trees[i]->root_index;
        label = new_labels_map[i];
        labels_map[i] = label;
        background_seed = false;
        adjacents_type = 0;
        gridSeeds = false;

        // Compute the area relevance
        area_prio = (double)trees[i]->num_nodes / (double)num_nodes;

        // Initial values for the computation of gradient relevance
        grad_prio = INFINITY;
        grad_prio_mix = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]); // Speeding purposes

        if (label > num_objmarkers)
            background_seed = true;
        if (label > num_markers)
            gridSeeds = true;

        // For each adjacent tree
        for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id, adj_label;
            float *mean_feat_j;
            double dist;
            bool adj_background;

            adj_background = false;
            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);
            adj_label = new_labels_map[adj_tree_id];

            // Compute the L2 norm between trees
            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            if (adj_label > num_objmarkers) // labels_map : indice indica labels e valor indica labels2
                adj_background = true;

            if ((background_seed && !adj_background) || (!background_seed && adj_background))
                grad_prio_mix = MIN(grad_prio_mix, dist);
            else // Get the minimum gradient value
                grad_prio = MIN(grad_prio, dist);

            if (adjacents_type != 2 && adjacents_type != 4)
            {
                /* os tipos com seed background são 1 e 2. */
                if (background_seed)
                {
                    if (!adj_background)
                        adjacents_type = 2;
                    else
                        adjacents_type = 1;
                }
                /* os tipos com seed foreground são 3 e 4. */
                else
                {
                    if (adj_background)
                        adjacents_type = 4;
                    else
                        adjacents_type = 3;
                }
            }
            free(mean_feat_j);
        }

        // Compute the superpixel relevance
        switch (adjacents_type)
        {
        case 1:
            tree_prio[i] = area_prio * grad_prio;
            type1++;
            if (gridSeeds)
                insertPrioQueue(&queue1, i);
            else
                index_label = addSeed_clust_bkp(root_index, labels_map[i], rel_seeds, new_labels_map, label_seed, index_label);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;
        case 2:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type2++;
            if (gridSeeds)
                insertPrioQueue(&queue2, i);
            else
                index_label = addSeed_clust_bkp(root_index, labels_map[i], rel_seeds, new_labels_map, label_seed, index_label);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;
        case 3:
            tree_prio[i] = area_prio * grad_prio;
            type3++;
            //insertPrioQueue(&queue3, i);
            index_label = addSeed_clust_bkp(root_index, labels_map[i], rel_seeds, new_labels_map, label_seed, index_label);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;
        case 4:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type4++;
            index_label = addSeed_clust_bkp(root_index, labels_map[i], rel_seeds, new_labels_map, label_seed, index_label);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;
        }
        free(mean_feat_i);
    }

    //std_deviation = sqrtf(fabs(sum_prio_2 - 2 * sum_prio + ((sum_prio * sum_prio) / (float)num_trees)) / (float)num_trees);
    //mean_relevance = sum_prio / (float)num_trees;
    //mean_relevance += std_deviation;

    std_deviation_13 = sqrtf(fabs(sum_prio_2_13 - 2 * sum_prio_13 + ((sum_prio_13 * sum_prio_13) / (float)type1 + type3)) / (float)type1 + type3);
    mean_relevance13 = sum_prio_13 / (float)(type1 + type3);
    mean_relevance13 += std_deviation_13;

    std_deviation_24 = sqrtf(fabs(sum_prio_2_24 - 2 * sum_prio_24 + ((sum_prio_24 * sum_prio_24) / (float)type2 + type4)) / (float)type2 + type4);
    mean_relevance24 = sum_prio_24 / (float)(type2 + type4);
    mean_relevance24 += std_deviation_24;

    while (!isPrioQueueEmpty(queue1))
    {
        int tree_id;
        int root_index;

        tree_id = popPrioQueue(&queue1);
        if (tree_prio[tree_id] >= mean_relevance13)
        {
            index_label = addSeed_clust_bkp(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, label_seed, index_label);
            while (!isPrioQueueEmpty(queue1))
            {
                tree_id = popPrioQueue(&queue1);
                index_label = addSeed_clust_bkp(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, label_seed, index_label);
            }
        }
        else
        {
            root_index = trees[tree_id]->root_index;
            index_label = superpixelSelectionType1_clust_bkp(tree_adj[tree_id]->head, tree_id, root_index, labels_map, num_objmarkers, rel_seeds, new_labels_map, index_label, label_seed);
        }
    }

    int prio = mean_relevance13;
    while (!isPrioQueueEmpty(queue3) && prio >= mean_relevance13)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue3);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance13)
            index_label = addSeed_clust_bkp(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, label_seed, index_label);
    }
    while (!isPrioQueueEmpty(queue3))
    {
        int tree_id;
        tree_id = popPrioQueue(&queue3);
        label_seed[trees[tree_id]->root_index] = 0;
    }

    prio = mean_relevance24;
    while (!isPrioQueueEmpty(queue2) && prio >= mean_relevance24)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue2);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance24)
            index_label = addSeed_clust_bkp(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, label_seed, index_label);
        else
            label_seed[trees[tree_id]->root_index] = 0;
    }

    while (!isPrioQueueEmpty(queue2))
    {
        int tree_id;
        tree_id = popPrioQueue(&queue2);
        label_seed[trees[tree_id]->root_index] = 0;
    }

    /*
    while (!isPrioQueueEmpty(queue2))
    {
        int tree_id;
        tree_id = popPrioQueue(&queue2);
        index_label = addSeed_clust(trees[tree_id]->root_index, 1, rel_seeds, new_labels_map, label_seed, index_label);
    }*/

    (*numTrees) = index_label;

    for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
    {
        int seed_index, tree_id, new_Tree_id;

        seed_index = ptr->elem;
        tree_id = label_seed[seed_index] - 1;
        new_Tree_id = label_seed[trees[tree_id]->root_index];
        label_seed[seed_index] = new_Tree_id;

        if (new_Tree_id != 0)
            insertIntListTail(&rel_seeds, seed_index);
    }

    bool background = false, foreground = false, gridSeeds = false;
    if (num_markers == 1)
        background = true;
    for (int i = 0; i < index_label; i++)
    {
        if (new_labels_map[i] <= num_objmarkers)
            foreground = true;
        else
        {
            if (new_labels_map[i] > num_markers)
                gridSeeds = true;
            else
                background = true;
        }
        if (foreground && background && gridSeeds)
            break;
    }
    if (num_markers == 1)
        background = false;
    if (!gridSeeds)
        *stop = 2;
    if (!foreground || (!background && !gridSeeds))
        *stop = 1;

    freePrioQueue(&queue1);
    freePrioQueue(&queue2);
    freePrioQueue(&queue3);
    freeMem(tree_prio);
    freeMem(labels_map);
    return rel_seeds;
}


IntLabeledList *seedSelection_clust(Tree **trees, IntList **tree_adj, int num_nodes, int *numTrees, int num_markers, int num_objmarkers, int *stop, IntLabeledList *nonRootSeeds, int *new_labels_map)
{
    double *tree_prio;
    IntLabeledList *rel_seeds;
    PrioQueue *queue1, *queue2, *queue3;
    double mean_relevance13, mean_relevance24; // usaremos como medida de limiar
    int *labels_map; // mapeia do tree_id anterior para o label
    int *labels_trees; // mapeia do tree_id anterior para o próximo +1
    int index_label;
    int num_trees;
    //float sum_prio, sum_prio_2, std_deviation;
    float sum_prio_13, sum_prio_2_13, std_deviation_13;
    float sum_prio_24, sum_prio_2_24, std_deviation_24;
    int type1 = 0, type2 = 0, type3 = 0, type4 = 0;

    index_label = 0;
    mean_relevance13 = 0.0;
    sum_prio_13 = 0;
    sum_prio_2_13 = 0;
    std_deviation_13 = 0;
    mean_relevance24 = 0.0;
    mean_relevance24 = 0.0;
    sum_prio_24 = 0;
    sum_prio_2_24 = 0;
    std_deviation_24 = 0;
    num_trees = (*numTrees);

    tree_prio = allocMem(num_trees, sizeof(double));
    rel_seeds = createIntLabeledList();
    labels_map = allocMem(num_trees, sizeof(int));
    labels_trees = allocMem(num_trees, sizeof(int));

    // inicia as filas
    queue1 = createPrioQueue(num_trees, tree_prio, MINVAL_POLICY);
    queue2 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);
    queue3 = createPrioQueue(num_trees, tree_prio, MAXVAL_POLICY);

    // For each tree
    for (int i = 0; i < num_trees; i++)
    {
        double area_prio, grad_prio, grad_prio_mix;
        float *mean_feat_i;
        int adjacents_type = 0; // 0:nothing , 1:only background with background seed , 2:mixed with background seed , 3:only foreground with foreground seed, 4:mixed with foreground seed
        bool background_seed, gridSeeds;
        int label;

        label = new_labels_map[i];
        labels_map[i] = label;
        background_seed = false;
        adjacents_type = 0;
        gridSeeds = false;

        // Compute the area relevance
        area_prio = (double)trees[i]->num_nodes / (double)num_nodes;

        // Initial values for the computation of gradient relevance
        grad_prio = INFINITY;
        grad_prio_mix = INFINITY;
        mean_feat_i = meanTreeFeatVector(trees[i]); // Speeding purposes

        if (label > num_objmarkers)
            background_seed = true;
        if (label > num_markers)
            gridSeeds = true;

        // For each adjacent tree
        for (IntCell *ptr = tree_adj[i]->head; ptr != NULL; ptr = ptr->next)
        {
            int adj_tree_id, adj_label;
            float *mean_feat_j;
            double dist;
            bool adj_background;

            adj_background = false;
            adj_tree_id = ptr->elem;
            mean_feat_j = meanTreeFeatVector(trees[adj_tree_id]);
            adj_label = new_labels_map[adj_tree_id];

            // Compute the L2 norm between trees
            dist = euclDistance(mean_feat_i, mean_feat_j, trees[i]->num_feats);

            if (adj_label > num_objmarkers) // labels_map : indice indica labels e valor indica labels2
                adj_background = true;

            // Get the minimum gradient value
            if ((background_seed && !adj_background) || (!background_seed && adj_background))
                grad_prio_mix = MIN(grad_prio_mix, dist);
            else 
                grad_prio = MIN(grad_prio, dist);

            if (adjacents_type != 2 && adjacents_type != 4)
            {
                /* os tipos com seed background são 1 e 2. */
                if (background_seed)
                {
                    if (!adj_background) adjacents_type = 2;
                    else adjacents_type = 1;
                }
                /* os tipos com seed foreground são 3 e 4. */
                else
                {
                    if (adj_background) adjacents_type = 4;
                    else adjacents_type = 3;
                }
            }
            free(mean_feat_j);
        }

        // Compute the superpixel relevance
        switch (adjacents_type)
        {
        case 1:
            tree_prio[i] = area_prio * grad_prio;
            type1++;
            if (gridSeeds)
                insertPrioQueue(&queue1, i);
            else
                index_label = addSeed_clust(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label, labels_trees, i);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;

        case 2:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type2++;
            if (gridSeeds)
                insertPrioQueue(&queue2, i);
            else
                index_label = addSeed_clust(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label, labels_trees,i);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;

        case 3:
            tree_prio[i] = area_prio * grad_prio;
            type3++;
            //insertPrioQueue(&queue3, i);
            index_label = addSeed_clust(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label, labels_trees,i);
            sum_prio_13 += tree_prio[i];
            sum_prio_2_13 += (tree_prio[i] * tree_prio[i]);
            break;

        case 4:
            tree_prio[i] = /*area_prio **/ grad_prio_mix;
            type4++;
            index_label = addSeed_clust(trees[i]->root_index, labels_map[i], rel_seeds, new_labels_map, index_label, labels_trees,i);
            sum_prio_24 += tree_prio[i];
            sum_prio_2_24 += (tree_prio[i] * tree_prio[i]);
            break;
        }
        free(mean_feat_i);
    }

    std_deviation_13 = sqrtf(fabs(sum_prio_2_13 - 2 * sum_prio_13 + ((sum_prio_13 * sum_prio_13) / (float)type1 + type3)) / (float)type1 + type3);
    mean_relevance13 = sum_prio_13 / (float)(type1 + type3);
    mean_relevance13 += std_deviation_13;

    std_deviation_24 = sqrtf(fabs(sum_prio_2_24 - 2 * sum_prio_24 + ((sum_prio_24 * sum_prio_24) / (float)type2 + type4)) / (float)type2 + type4);
    mean_relevance24 = sum_prio_24 / (float)(type2 + type4);
    mean_relevance24 += std_deviation_24;

    while (!isPrioQueueEmpty(queue1))
    {
        int tree_id;

        tree_id = popPrioQueue(&queue1);
        if (tree_prio[tree_id] >= mean_relevance13)
        {
            index_label = addSeed_clust(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label, labels_trees, tree_id);
            while (!isPrioQueueEmpty(queue1))
            {
                tree_id = popPrioQueue(&queue1);
                index_label = addSeed_clust(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label, labels_trees, tree_id);
            }
        }
        else
        {
            index_label = superpixelSelectionType1_clust(tree_adj[tree_id]->head, tree_id, trees[tree_id]->root_index, labels_map, num_objmarkers, rel_seeds, new_labels_map, index_label, labels_trees);
        }
    }

    int prio = mean_relevance13;
    while (!isPrioQueueEmpty(queue3) && prio >= mean_relevance13)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue3);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance13)
            index_label = addSeed_clust(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label, labels_trees, tree_id);
    }

    prio = mean_relevance24;
    while (!isPrioQueueEmpty(queue2) && prio >= mean_relevance24)
    {
        int tree_id;
        tree_id = popPrioQueue(&queue2);
        prio = tree_prio[tree_id];

        if (prio >= mean_relevance24)
            index_label = addSeed_clust(trees[tree_id]->root_index, labels_map[tree_id], rel_seeds, new_labels_map, index_label, labels_trees, tree_id);
    }

    (*numTrees) = index_label;

    for (IntLabeledCell *ptr = nonRootSeeds->head; ptr != NULL; ptr = ptr->next)
    {
        int tree_id;
        tree_id = labels_trees[ptr->treeId]-1;

        if (tree_id != 0)
        {
            ptr->treeId = tree_id;
            ptr->label = new_labels_map[tree_id];
        }
    }
    
    bool background = false, foreground = false, gridSeeds = false;
    if (num_markers == 1)
        background = true;
    for (int i = 0; i < index_label; i++)
    {
        if (new_labels_map[i] <= num_objmarkers)
            foreground = true;
        else
        {
            if (new_labels_map[i] > num_markers)
                gridSeeds = true;
            else
                background = true;
        }
        if (foreground && background && gridSeeds)
            break;
    }
    if (num_markers == 1)
        background = false;
    if (!gridSeeds)
        *stop = 2;
    if (!foreground || (!background && !gridSeeds))
        *stop = 1;

    freePrioQueue(&queue1);
    freePrioQueue(&queue2);
    freePrioQueue(&queue3);
    freeMem(tree_prio);
    freeMem(labels_map);
    freeMem(labels_trees);
    return rel_seeds;
}


//=============================================================================
// Void
//=============================================================================
inline void insertNodeInTree(Graph *graph, int index, Tree **tree, double grad)
{
    (*tree)->num_nodes++;

    for (int i = 0; i < graph->num_feats; i++)
        (*tree)->sum_feat[i] += graph->feats[index][i];

    (*tree)->sum_grad += grad;
    (*tree)->sum_grad_2 += (grad * grad);
}
