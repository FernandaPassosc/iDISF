/**
* Dynamic and Iterative Spanning Forest
* 
* @date September, 2019
*/
#ifndef DISF_H
#define DISF_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include "Utils.h"
#include "Image.h"
#include "Color.h"
#include "IntList.h"
#include "IntLabeledList.h"
#include "PrioQueue.h"
#include <omp.h>
//#include "spectralClust.h"

//=============================================================================
// Definitions
//=============================================================================
#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))
#define ROUND(x) ((x < 0)?(int)(x-0.5):(int)(x+0.5))

//=============================================================================
// Constants
//=============================================================================
const static float GAUSSIAN_3x3[9] = {4.0/16.0,// Center
                                      2.0/16.0,// Center-Left
                                      2.0/16.0,// Center-Right
                                      2.0/16.0,// Top-Center
                                      2.0/16.0,// Bottom-Center
                                      1.0/16.0,// Bottom-Left
                                      1.0/16.0,// Top-Right
                                      1.0/16.0,// Top-Left
                                      1.0/16.0};// Bottom-Right

//=============================================================================
// Structures
//=============================================================================
/**
* Node 2D coordinates
*/
typedef struct
{
    int x, y;
} NodeCoords;

/**
* Adjacency relation between the nodes of the graph
*/ 
typedef struct
{
    int size;
    // An adjacent coordinate can be obtained by simply adding the variations
    // along the x-axis and y-axis (dx and dy, respectively). Example: the adjacent
    // in the left can be obtained if dx[i] = -1, and dy[i] = 0, for an i < size
    int *dx, *dy;
} NodeAdj;

/**
* Abstract representation of an optimum-path tree
*/
typedef struct
{
    int root_index, num_nodes, num_feats;
    float minDist_userSeed; 
    // For speeding purposes, it is preferable to have such vector for the summation
    // of the pixel's features. 
    float *sum_feat;

    // For speeding purposes, it is preferable to have such vector for the summation
    // of the grad's pixels.     
    float sum_grad;   // sum (grad)
    float sum_grad_2; // sum (grad^2)
    //int type;
} Tree;

/**
* Image Graph
*/
typedef struct
{
    int num_cols, num_rows, num_feats, num_nodes;
    // Each node whose index is i < num_nodes, contain num_feats of features, which
    // can be obtained through feats[i].
    float **feats;
} Graph;


int* clusterPoints(Image *image, Graph *graph, NodeCoords **coords_markers, int *marker_sizes, int num_markers, int *numTotalClusters, int normalize);
int* clusterPoints_normalized(Image *image, Graph *graph, NodeCoords **coords_markers, int *marker_sizes, int num_markers, int *numTotalClusters);
int* clusterPoints_normalized2(Image *image, Graph *graph, NodeCoords **coords_markers, int *marker_sizes, int num_markers, int *numTotalClusters);

//=============================================================================
// Bool Functions
//=============================================================================
/**
* Evaluates if the coordinates of a given NodeCoords object are within the 
* domains of the image graph
*/
bool areValidNodeCoords(int num_rows, int num_cols, NodeCoords coords);

//=============================================================================
// Int Functions
//=============================================================================
/**
* Converts the coordinates of a given NodeCoords object into an index. Warning!
* It does not verify if the coordinates given are valid!
*/
int getNodeIndex(int num_cols, NodeCoords coords);

//=============================================================================
// Double Functions
//=============================================================================
/**
* Computes the L2 Norm (a.k.a. Euclidean Distance) between two feature vectors
* of same dimensionality
*/
double euclDistance(float *feat1, float *feat2, int num_feats);

/**
* Computes the L2 Norm (a.k.a. Euclidean Distance) between two features
*/
double euclDistanceValues(float feat1, float feat2);

/**
* Computes the L2 Norm (a.k.a. Euclidean Distance) between two NodeCoords
*/
double euclDistanceCoords(NodeCoords feat1, NodeCoords feat2);

/**
* Computes the L1 Norm (a.k.a. Taxicab Distance) between two feature vectors 
* of same dimensionality
*/
double taxicabDistance(float *feat1, float *feat2, int num_feats);

/*
Compute the path cost
*/
double calcPathCost(float *mean_feat_tree, float *feats, int num_feats, double cost_map, int num_nodes_tree, double grad_adj, double coef_variation_tree, double alpha, double c2, int function);

//=============================================================================
// NodeCoords Functions
//=============================================================================
/**
* Get the coordinates of the id-th adjacent pixel determined by the adjacency
* relation considered. Warning! It does not evaluate whether the id given is 
* valid
*/
NodeCoords getAdjacentNodeCoords(NodeAdj *adj_rel, NodeCoords coords, int id);

/**
* Gets the coordinates of a pixel at the given index in the image graph. Warning!
* It does note evaluate whether the index is valid!
*/
NodeCoords getNodeCoords(int num_cols, int index);

//=============================================================================
// Float* Functions
//=============================================================================
/**
* Computes the mean feature vector of a given tree
*/
float* meanTreeFeatVector(Tree *tree);

/**
* Computes the mean gradient of a given tree
*/
float meanTreeGradVector(Tree *tree);

//=============================================================================
// Double* Functions
//=============================================================================
/**
* Computes the image gradient of the graph. It performs a summation of the
* of the ;weighted differences between a center pixel, and its adjacents.
*/
//double *computeGradient(Graph *graph);
double *computeGradient(Graph *graph, double *coef_variation_img);
//=============================================================================
// NodeAdj* Functions
//=============================================================================
/**
* Creates a 4-neighborhood adjacency relation
*/
NodeAdj *create4NeighAdj();

/**
* Creates an 8-neighborhood adjacency relation
*/
NodeAdj *create8NeighAdj();

//=============================================================================
// Graph* Functions
//=============================================================================
/**
* Creates an image graph given the image in parameter. It considers an 4-adjacency
* relation between the nodes, and converts the image's features (expecting RGB)
* into the L*a*b* colorspace.
*/
Graph *createGraph(Image *img);

//=============================================================================
// Tree* Functions
//=============================================================================
/**
* Creates an empty tree rooted at the given index. Warning! It does not add the
* index features into the tree!
*/
Tree *createTree(int root_index, int num_feats);

//=============================================================================
// Image* Functions
//=============================================================================
/**
* Extracts the superpixels using the Dynamic and Iterative Spanning Forest 
* algorithm. Given an initial number of N_0 seeds, it removes iteratively the 
* most irrelevant ones, until the number N_f of superpixels is achieved. It is 
* possible to obtain the border map by simply creating an Image whose width and
* height are the same as the original image, but the number of channels is 1. 
* However, if no border map is desired, simply define the respective border image 
* object as NULL. Warning! It does not verify if N_0 >> N_f!
*/
//Image *runDISF(Image *img, int n_0, int n_f, Image **border_img, NodeCoords* coords_user_seeds, int num_user_seeds);
Image *runiDISF(Graph *graph, int n_0, int n_f, Image **border_img, NodeCoords** coords_user_seeds, int num_user_seeds, int *marker_sizes, int function, int all_borders, double c1, double c2, int num_objmarkers);
Image *runiDISF_scribbles_rem(Graph *graph, int n_0, int iterations, Image **border_img, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int function, int all_borders, double c1, double c2, /*int sampling_method,*/ int obj_markers);
Image *runiDISF_scribbles_clust_bkp(Graph *graph, int n_0, int iterations, Image **border_img, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int function, int all_borders, double c1, double c2, /*int sampling_method,*/ int obj_markers);
Image *runiDISF_scribbles_clust(Graph *graph, int n_0, int iterations, Image **border_img, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int function, int all_borders, double c1, double c2, /*int sampling_method,*/ int obj_markers, Image *image);
Image *runLabeledDISF(Graph *graph, int n_0, int n_f, NodeCoords **coords_user_seeds, Image **border_img);

//=============================================================================
// IntList* Functions
//=============================================================================
/**
* Performs a grid sampling in the image graph, in order to achieve an approximate
* number of seeds (given in parameter), and returns a list of seed pixels indexes. 
* Please, be aware that the real number of seeds can be very different from expected.
*/
IntList *gridSampling(int num_cols, int num_rows, int *num_seeds, NodeCoords** coords_user_seeds, int num_user_seeds, int *marker_sizes, double *grad);
IntList *gridSampling_scribbles(int num_rows, int num_cols, int *n_0, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, double *grad, int *labels_map, int obj_markers);
IntList *gridSampling_scribbles_clust_bkp(Graph *graph, int *n_0, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, double *grad, int *labels_map, int obj_markers, int* label_seed, int* numTrees);
IntLabeledList *gridSampling_scribbles_clust(Image *image, Graph *graph, int *n_0, NodeCoords **coords_markers, int num_markers, int *marker_sizes, double *grad, int obj_markers, int *numTrees, IntLabeledList **nonRootSeeds_out);
IntLabeledList *gridSampling_scribbles_1clust(Image *image, Graph *graph, int *n_0, NodeCoords **coords_markers, int num_markers, int *marker_sizes, double *grad, int obj_markers, int *numTrees, IntLabeledList **nonRootSeeds_out);
IntList *gridSamplingDISF(Graph *graph, int num_seeds);

/**
* Selects the seeds which generated the K most relevant superpixels, according to
* their area and gradient (defined by the tree-adjacency relation given), and returns
* their root pixel indexes on a list. Warning! It does not verify if the number to 
* maintain is greater than the current number of trees!
*/
//IntList *selectKMostRelevantSeeds(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain, Graph *graph);
IntList *selectKMostRelevantSeeds(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain, int num_user_seeds);
IntList *seedSelection_clust_bkp(Tree **trees, IntList **tree_adj, int num_nodes, int *num_trees, int num_markers, int num_objmarkers, int *new_labels_map, int *label_seed, int *stop, IntList *seed_set);
IntLabeledList *seedSelection_clust(Tree **trees, IntList **tree_adj, int num_nodes, int *numTrees, int num_markers, int num_objmarkers, int *stop, IntLabeledList *nonRootSeeds, int *new_labels_map);
IntList *seedRemoval(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_markers, int num_objmarkers, int *new_labels_map, int *stop);
IntList *selectSeedDISF(Tree **trees, IntList **tree_adj, int num_nodes, int num_trees, int num_maintain);

//=============================================================================
// Void Functions
//=============================================================================
/**
* Deallocates the memory reserved for the adjacency relation given in parameter
*/ 
void freeNodeAdj(NodeAdj **adj_rel);

/**
* Deallocates the memory reserved for the tree given in parameter
*/ 
void freeTree(Tree **tree);

/**
* Deallocates the memory reserved for the image graph given in parameter
*/ 
void freeGraph(Graph **graph);

/**
* Inserts a node into the given tree. Thus, the node's features are added to
* the tree's summation vector, and the tree's size is increased by one. Warning!
* It does not verify whether such node was already inserted in this, or any other
* tree!
*/ 
void insertNodeInTree(Graph *graph, int index, Tree **tree, double grad);


#ifdef __cplusplus
}
#endif

#endif
