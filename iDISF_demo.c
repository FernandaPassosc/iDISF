/**
* Interactive Segmentation based on Dynamic and Iterative Spanning Forest (C)
*
* @date September, 2020
*/

//=============================================================================
// Includes
//=============================================================================
#include "Image.h"
#include "iDISF.h"
#include "Utils.h"
//#include <stdio.h>
//#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//=============================================================================
// Prototypes
//=============================================================================
void usage(char *argv);
Image *loadImage(const char *filepath);
void writeImagePGM(Image *img, char *filepath);
void writeImagePPM(Image *img, char *filepath);
Image *overlayBorders(Image *img, Image *border_img);

void usage(char *argv)
{
    printf("\n Usage    : %s [options]\n", argv); // 5 args
    printf("\n Options  : \n");
    printf("           --i : Input image\n");
    printf("           --n0 : Number of init GRID seeds (>= 0)\n");
    printf("           --it : Iterations/Number of final superpixels\n");
    printf("           --f : Path-cost function. {1:color distance, 2:gradient-cost, 3:beta norm, 4:cv tree norm, 5:sum gradient-cost, 6: sum beta norm}\n");
    printf("\n Optional : \n");
    printf("           --xseeds : Object seed x coord\n");
    printf("           --yseeds : Object seed y coord\n");
    printf("           --file <scribbles.txt>: File name with the pixel coordinates of the scribbles\n");
    printf("           WARNING: Use --xseeds --yseeds OR --file\n");
    printf("           --inverse : Inverse the pixel coordinates of the scribbles. Can use any char with this flag to activate it. \n");
    printf("           --saveSeeds <seeds.txt>: Save all pixels of the scribbles and all seeds sampled in the specificated file. \n");
    printf("           --draw <image.pgm>: Save an pgm image with all seeds (grid or from scribbles). \n");
    printf("           --c1 : Used in path-cost functions 2-5. Interval: [0.1,1.0] \n");
    printf("           --c2 : Used in path-cost functions 2-5. Interval: [0.1,1.0] \n");
    printf("           --max_markers : Define the number of scribbles that will be used. (Default: The number of scribbles in scribbles file)\n");
    printf("           --obj_markers : Define the number of scribbles that will be labeled as object. (Default: all scribbles are labeled as object)\n");
    printf("           --o <path/image>: Define the image name and its path.\n");
    printf("\n Seeds file format : \n");
    printf("          number of scribbles\\n \n");
    printf("          number of pixels in the scribble\\n \n");
    printf("          x_coord;y_coord\\n \n");
    printf("          obs: the last coord don't have \"\\n\" \n");
}

/* This function its used to write a txt file with all seeds */
void writeSeeds(Image *img, int n_0, NodeCoords **coords_user_seeds, int num_user_seeds, int *marker_sizes, char *seedsFile)
{
    IntList *seed_set;
    Graph *graph;
    double *grad;

    double coef_variation_img;
    graph = createGraph(img);
    grad = computeGradient(graph, &coef_variation_img);
    seed_set = gridSampling(graph->num_cols, graph->num_rows, &n_0, coords_user_seeds, num_user_seeds, marker_sizes, grad);
    ////// WRITE THE SEEDS IN FILE
    FILE *file = fopen(seedsFile, "w");
    int *is_seed = (int *)allocMem(graph->num_nodes, sizeof(int));

    int total_pixels_marked = 0;

    for (int i = 0; i < num_user_seeds; i++)
    {
        for (int j = 0; j < marker_sizes[i]; j++)
            total_pixels_marked++;
    }

    if (file == NULL)
    {
        printError("Main:writeSeeds", "Was do not possible to write file %s", seedsFile);
    }

    fprintf(file, "%d;%d\n", num_user_seeds, n_0);

    for (int i = 0; i < num_user_seeds; i++)
    {
        fprintf(file, "%d\n", marker_sizes[i]);
        for (int j = 0; j < marker_sizes[i]; j++)
        {
            fprintf(file, "%d;%d\n", coords_user_seeds[i][j].x, coords_user_seeds[i][j].y);
            is_seed[getNodeIndex(graph->num_cols, coords_user_seeds[i][j])] = 1;
        }
    }

    //fprintf(file, "GRID\n");
    for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
    {
        int seed_index = ptr->elem;
        if (is_seed[seed_index] == 0)
        {
            NodeCoords GRIDseeds = getNodeCoords(graph->num_cols, seed_index);
            fprintf(file, "%d;%d\n", GRIDseeds.x, GRIDseeds.y);
        }
    }

    fclose(file);
    freeMem(is_seed);
    freeMem(grad);
    freeIntList(&seed_set);
    //freeGraph(graph);
}


void draw_all_seeds(Image *img, Graph *graph, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int obj_markers, int n_0, char *fileName)
{
    IntList *seed_set;
    Image *label_img;
    int num_cols, num_rows, num_nodes, num_channels;
    //int n_0 = 0;
    int normval;
    int *labels_map, *label_seed;
    int num_pixels, i, j;
    double *grad, alpha;
    float size, stride, delta, max_seeds;

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_channels = img->num_channels;
    num_pixels = img->num_pixels;

    label_seed = (int *)allocMem(num_nodes, sizeof(int));
    label_img = createImage(img->num_rows, img->num_cols, 3);
    grad = computeGradient(graph, &alpha);
    normval = getNormValue(img);

    // Compute the approximate superpixel size and stride
    size = 0.5 + ((float)num_nodes / ((float)n_0));
    stride = sqrtf(size) + 0.5;
    delta = stride / 2.0;
    delta = (int)delta;
    stride = (int)stride;

    max_seeds = ((((float)num_rows - delta) / stride) + 1) * ((((float)num_cols - delta) / stride) + 1);

    for (int i = 0; i < num_markers; i++)
        max_seeds += marker_sizes[i];

    labels_map = (int *)allocMem((int)max_seeds, sizeof(int));

    seed_set = gridSampling_scribbles(num_rows, num_cols, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, labels_map, 4);
    //numTrees = seed_set->size;

    int seed_label = 0;
    for (IntCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
    {
        int seed_index;
        seed_index = ptr->elem;
        label_seed[seed_index] = labels_map[seed_label];
        seed_label++;
    }
    freeIntList(&seed_set);

    float colors[num_markers + 1][3];
    // object scribbles are cyan
    for (int i = 0; i < obj_markers; i++)
    {
        // cyan
        colors[i][0] = 0.0;
        colors[i][1] = 1.0;
        colors[i][2] = 1.0;

        // yellow
        //colors[i][0] = 1.0;
        //colors[i][1] = 0.9;
        //colors[i][2] = 0.0;
    }
    // background scribbles are red
    for (int i = obj_markers; i < num_markers - 1; i++)
    {
        // red
        colors[i][0] = 1.0;
        colors[i][1] = 0.0;
        colors[i][2] = 0.0;
    }
    // grid seeds are blue
    //colors[num_markers][0] = 0.0;
    //colors[num_markers][1] = 0.0;
    //colors[num_markers][2] = 1.0;

    // yellow
    colors[num_markers - 1][0] = 1.0;
    colors[num_markers - 1][1] = 0.9;
    colors[num_markers - 1][2] = 0.0;

    float radius = 5.5;

    for (i = 0; i < num_pixels; i++)
    {
        int label = label_seed[i];
        if (label != 0)
        {
            int color[3];
            for (j = 0; j < 3; j++)
            {
                color[j] = colors[label - 1][j] * normval;
                label_img->val[i][j] = color[j];
            }
            if (label >= num_markers || marker_sizes[label - 1] == 1 /*|| label <= num_markers*/)
            {
                // center
                NodeCoords coords = getNodeCoords(num_cols, i);

                // draw a circle
                for (int y = MAX(coords.y - radius, 0); y <= MIN(coords.y + radius, num_rows); y++)
                {
                    for (int x = MAX(coords.x - radius, 0); x <= MIN(coords.x + radius, num_cols); x++)
                    {
                        if (sqrtf((y - coords.y) * (y - coords.y) + (x - coords.x) * (x - coords.x)) <= radius)
                        {
                            NodeCoords point;
                            point.x = x;
                            point.y = y;
                            for (j = 0; j < 3; j++)
                            {
                                label_img->val[getNodeIndex(num_cols, point)][j] = color[j];
                            }
                        }
                    }
                }
            }
        }
        else
        {
            if (label_img->val[i][0] == 0 && label_img->val[i][1] == 0 && label_img->val[i][2] == 0)
            {
                for (j = 0; j < 3; j++)
                {
                    if (num_channels == 1) // It will convert the image to PPM
                        label_img->val[i][j] = img->val[i][0];
                    else
                        label_img->val[i][j] = img->val[i][j];
                }
            }
        }
    }

    freeMem(grad);
    freeMem(label_seed);
    writeImagePPM(label_img, fileName);
    freeImage(&label_img);
    freeMem(labels_map);
}


void draw_scribbles_1clust(Image *img, Graph *graph, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int obj_markers, char *fileName)
{
    IntLabeledList *seed_set;    // lista de sementes
    IntLabeledList *nonRootSeeds; // lista de pixels rotulados pelo usuário que não são raíz
    int *label_img;              // label_img[node_index] = tree_id+1;
    Image *label_img2;           // label_img2->val[node_index][0] = label de segmentação
    int num_cols, num_rows, num_nodes, num_channels;
    int n_0 = 0/*, normval*/;
    int num_pixels, i, j, numTrees;
    double *grad, alpha;
    
    srand(time(0));

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_channels = img->num_channels;
    num_pixels = img->num_pixels;

    label_img = (int*)allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, num_channels);
    grad = computeGradient(graph, &alpha);
    //normval = getNormValue(img);
    nonRootSeeds = createIntLabeledList();

    seed_set = gridSampling_scribbles_1clust(img, graph, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, obj_markers, &numTrees, &nonRootSeeds);
    
    numTrees = seed_set->size;

    /*
    int colors[numTrees][3];
#pragma omp parallel for private(i, j) \
    firstprivate(numTrees)             \
    shared(colors)
    for (i = 0; i < numTrees; i++)
    {
        for (j = 0; j < 3; j++)
            colors[i][j] = rand() % normval;
    }
    */
    int colors[10][3] = {{0,0,255},{255,0,0},{0,255,0},{255,0,255},{255,255,0},{0,255,255},{128,255,0},{128,0,255},{255,0,128},{255,128,0}};


    for (IntLabeledCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
    {
        label_img[ptr->elem] = ptr->treeId+1;
    }

    for (IntLabeledCell *ptr = nonRootSeeds->head; ptr != NULL; ptr = ptr->next)
    {
        label_img[ptr->elem] = ptr->treeId+1;
    }
    printf("seed_set: %d \n", seed_set->size);
    printf("nonRootSeeds: %d \n", nonRootSeeds->size);

#pragma omp parallel for private(i, j)              \
    firstprivate(num_channels, num_pixels) \
        shared(label_img2, label_img, img, colors)
    for (i = 0; i < num_pixels; i++)
    {
        for (j = 0; j < 3; j++)
        {
            int label = label_img[i];
            if (label != 0)
                label_img2->val[i][j] = colors[(label - 1)%10][j];

            else if (num_channels == 1) // It will convert the image to PPM
                label_img2->val[i][j] = img->val[i][0];
            else
                label_img2->val[i][j] = img->val[i][j];
        }
    }
    freeMem(grad);
    freeMem(label_img);
    writeImagePPM(label_img2, fileName);
    freeImage(&label_img2);
    freeIntLabeledList(&seed_set);
    freeIntLabeledList(&nonRootSeeds);
}


void draw_scribbles_clust(Image *img, Graph *graph, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int obj_markers, char *fileName)
{
    IntLabeledList *seed_set;    // lista de sementes
    IntLabeledList *nonRootSeeds; // lista de pixels rotulados pelo usuário que não são raíz
    int *label_img;              // label_img[node_index] = tree_id+1;
    Image *label_img2;           // label_img2->val[node_index][0] = label de segmentação
    int num_cols, num_rows, num_nodes, num_channels;
    int n_0 = 0, normval;
    int num_pixels, i, j, numTrees;
    double *grad, alpha;
    
    srand(time(0));

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_channels = img->num_channels;
    num_pixels = img->num_pixels;

    label_img = (int*)allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, num_channels);
    grad = computeGradient(graph, &alpha);
    normval = getNormValue(img);
    nonRootSeeds = createIntLabeledList();

    seed_set = gridSampling_scribbles_clust(img, graph, &n_0, coords_user_seeds, num_markers, marker_sizes, grad, obj_markers, &numTrees, &nonRootSeeds);
    
    numTrees = seed_set->size;

    
    int colors[numTrees][3];
#pragma omp parallel for private(i, j) \
    firstprivate(numTrees)             \
    shared(colors)
    for (i = 0; i < numTrees; i++)
    {
        for (j = 0; j < 3; j++)
            colors[i][j] = rand() % normval;
    }
    
    //int colors[10][3] = {{0,0,255},{255,0,0},{0,255,0},{255,0,255},{255,255,0},{0,255,255},{128,255,0},{128,0,255},{255,0,128},{255,128,0}};


    for (IntLabeledCell *ptr = seed_set->head; ptr != NULL; ptr = ptr->next)
    {
        label_img[ptr->elem] = ptr->treeId+1;
    }

    for (IntLabeledCell *ptr = nonRootSeeds->head; ptr != NULL; ptr = ptr->next)
    {
        label_img[ptr->elem] = ptr->treeId+1;
    }
    printf("seed_set: %d \n", seed_set->size);
    printf("nonRootSeeds: %d \n", nonRootSeeds->size);

#pragma omp parallel for private(i, j)              \
    firstprivate(num_channels, num_pixels, normval) \
        shared(label_img2, label_img, img, colors)
    for (i = 0; i < num_pixels; i++)
    {
        for (j = 0; j < 3; j++)
        {
            int label = label_img[i];
            if (label != 0)
                label_img2->val[i][j] = colors[label - 1][j];

            else if (num_channels == 1) // It will convert the image to PPM
                label_img2->val[i][j] = img->val[i][0];
            else
                label_img2->val[i][j] = img->val[i][j];
        }
    }
    freeMem(grad);
    freeMem(label_img);
    writeImagePPM(label_img2, fileName);
    freeImage(&label_img2);
    freeIntLabeledList(&seed_set);
    freeIntLabeledList(&nonRootSeeds);
}


void draw_scribbles_clust_bkp(Image *img, Graph *graph, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int obj_markers, char *fileName)
{
    Image *label_img;
    int num_cols, /*num_rows,*/ num_nodes, num_channels;
    int normval;
    int *labels;
    int numTotalCusters;
    int *label_seed;
    int num_pixels, i, j;
    
    srand(time(0));

    num_cols = graph->num_cols;
    //num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_channels = img->num_channels;
    num_pixels = img->num_pixels;

    labels = clusterPoints(img, graph, coords_user_seeds, marker_sizes, num_markers, &numTotalCusters, 0);
    //labels = clusterPoints_normalized(img, graph, coords_user_seeds, marker_sizes, num_markers, &numTotalCusters);

    label_seed = (int *)allocMem(num_nodes, sizeof(int));
    label_img = createImage(img->num_rows, img->num_cols, 3); // create an empty image
    normval = getNormValue(img);
    printf("normval=%d \n", normval);

    int index = 0;
    for(int i=0; i < num_markers; i++)
    {
        for(int j=0; j < marker_sizes[i]; j++)
        {
            int node_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
            label_seed[node_index] = labels[index]+1;
            index++;
        }
    }
    
    /*
    int colors[numTotalCusters][3];
#pragma omp parallel for private(i, j) \
    firstprivate(numTotalCusters)             \
        shared(colors)
    for (i = 0; i < numTotalCusters; i++)
    {
        for (j = 0; j < 3; j++)
            colors[i][j] = rand() % normval;
    }
    */
    int colors[10][3] = {{0,0,255},{255,0,0},{0,255,0},{255,0,255},{255,255,0},{0,255,255},{128,255,0},{128,0,255},{255,0,128},{255,128,0}};

#pragma omp parallel for private(i, j)              \
    firstprivate(num_channels, num_pixels, normval) \
        shared(label_img, label_seed, img, colors)
    for (i = 0; i < num_pixels; i++)
    {
        for (j = 0; j < 3; j++)
        {
            int label = label_seed[i];
            if (label != 0)
                label_img->val[i][j] = colors[(label - 1)%10][j];

            else if (num_channels == 1) // It will convert the image to PPM
                label_img->val[i][j] = img->val[i][0];
            else
                label_img->val[i][j] = img->val[i][j];
        }
    }
    freeMem(label_seed);
    writeImagePPM(label_img, fileName);
    freeImage(&label_img);
}


void draw_markers(Image *img, Graph *graph, NodeCoords **coords_user_seeds, int num_markers, int *marker_sizes, int obj_markers, char *fileName)
{
    int *label_img;              // label_img[node_index] = tree_id+1;
    Image *label_img2;           // label_img2->val[node_index][0] = label de segmentação
    int num_cols, num_rows, num_nodes, num_channels;
    int /*n_0 = 0,*/ normval;
    int num_pixels, i, j, numTrees;
    
    srand(time(0));

    num_cols = graph->num_cols;
    num_rows = graph->num_rows;
    num_nodes = graph->num_nodes;
    num_channels = img->num_channels;
    num_pixels = img->num_pixels;

    label_img = (int*)allocMem(num_nodes, sizeof(int)); // f
    label_img2 = createImage(num_rows, num_cols, num_channels);
    normval = getNormValue(img);
    numTrees = num_markers;

    int colors[numTrees][3];
#pragma omp parallel for private(i, j) \
    firstprivate(numTrees)             \
    shared(colors)
    for (i = 0; i < numTrees; i++)
    {
        for (j = 0; j < num_channels; j++)
            colors[i][j] = rand() % normval;
    }

    for (int i=0; i < num_markers; i++)
    {
        for(int j=0; j < marker_sizes[i]; j++)
        {
            int node_index = getNodeIndex(num_cols, coords_user_seeds[i][j]);
            label_img[node_index] = i+1;
        }
    }

#pragma omp parallel for private(i, j)              \
    firstprivate(num_channels, num_pixels, normval) \
        shared(label_img2, label_img, img, colors)
    for (i = 0; i < num_pixels; i++)
    {
        for (j = 0; j < 3; j++)
        {
            int label = label_img[i];
            if (label != 0)
                label_img2->val[i][j] = colors[label - 1][j];

            else if (num_channels == 1) // It will convert the image to PPM
                label_img2->val[i][j] = img->val[i][0];
            else
                label_img2->val[i][j] = img->val[i][j];
        }
    }
    freeMem(label_img);
    writeImagePPM(label_img2, fileName);
    freeImage(&label_img2);
}

/**
 * Reads in a text file the coordinates of the marked pixels
 * @param fileMarkers in : char[255] -- a file name
 * @param max_markers in : int -- the maximum number of markers
 * @param marker_sizes out : &int[num_user_seeds] -- the size of each marker
 * @param coords_markers out : &NodeCoords[num_user_seeds][marker_sizes]
 * @param inverse in : 0 or 1 -- indicates if the coordinates are inverted
 * @return : number of markers 
 */
int readMarkersFile(char fileMarkers[],
                    int max_markers,
                    int **marker_sizes_out,
                    NodeCoords ***coords_markers_out,
                    int inverse)
{
    int num_user_seeds = 0;
    NodeCoords **coords_markers;
    int *marker_sizes;

    FILE *file = fopen(fileMarkers, "r");

    if (file == NULL)
        printError("readFile", ("Was do not possible to read the seeds file"));

    if (fscanf(file, "%d\n", &num_user_seeds) < 1)
        printError("readFile", "Invalid file format");

    marker_sizes = (int *)allocMem(num_user_seeds, sizeof(int));
    coords_markers = (NodeCoords **)allocMem(num_user_seeds, sizeof(NodeCoords *));

    if (max_markers < 1)
        max_markers = num_user_seeds;

    for (int i = 0; i < MIN(num_user_seeds, max_markers); i++)
    {
        if (fscanf(file, "%d\n", &(marker_sizes[i])) < 1)
            printError("readFile", "Invalid file format");

        coords_markers[i] = (NodeCoords *)allocMem(marker_sizes[i], sizeof(NodeCoords));

        for (int j = 0; j < marker_sizes[i]; j++)
        {
            if (!(i == num_user_seeds - 1 && j == marker_sizes[i] - 1))
            {
                if (inverse == 0)
                {
                    if (fscanf(file, "%d;%d\n", &coords_markers[i][j].x, &coords_markers[i][j].y) != 2 ||
                        coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                    {
                        printf("coordsx=%d, coordsy=%d \n", coords_markers[i][j].x, coords_markers[i][j].y);
                        printError("readFile", "Invalid coords values in file");
                    }
                }
                else
                {
                    // CASO AS COORDENADAS ESTEJAM TROCADAS
                    if (fscanf(file, "%d;%d\n", &coords_markers[i][j].y, &coords_markers[i][j].x) != 2 ||
                        coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                        printError("readFile", "Invalid coords values in file");
                }
            }
        }
    }

    int i = num_user_seeds - 1;
    int j = marker_sizes[i] - 1;

    if (num_user_seeds <= max_markers)
    {
        if (inverse == 0)
        {
            if (fscanf(file, "%d;%d", &coords_markers[i][j].x, &coords_markers[i][j].y) != 2 ||
                coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                printError("readFile", "Invalid coords values in file2");
        }
        else
        {
            // CASO AS COORDENADAS ESTEJAM TROCADAS
            if (fscanf(file, "%d;%d", &coords_markers[i][j].y, &coords_markers[i][j].x) != 2 ||
                coords_markers[i][j].x < 0 || coords_markers[i][j].y < 0)
                printError("readFile", "Invalid coords values in file2");
        }
    }
    else
    {
        num_user_seeds = max_markers;
    }

    fclose(file);

    (*coords_markers_out) = coords_markers;
    (*marker_sizes_out) = marker_sizes;

    return num_user_seeds;
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char *argv[])
{
    // input args
    char imagePath[255];
    int num_init_seeds, iterations;
    char fileObjSeeds[255];
    int function;
    int all_borders = 0;
    char fileSeeds[256];
    double c1, c2;
    int sampling_method;
    int max_markers = -1;
    int obj_markers = 1;
    int segm = 1;
    int i;

    // structures used
    Image *img, *border_img, *label_img, *ovlay_img;
    NodeCoords **coords_user_seeds;
    Graph *graph;
    clock_t time;

    // others
    int num_user_seeds, *marker_sizes = NULL;
    char *pch, *imageName;
    char ovlayName[255], labelsName[255], bordersName[255];
    char xseeds[255], yseeds[255], n0[255], it[255], f[255];
    char inverse[255];
    char all[255];
    char c1_char[256], c2_char[256];
    char output[256];
    char sampling[256];
    char lists[256];
    char maxMarkers[256];
    char objMarkers[256];
    char clust[256];
    char fileName[256];
    char segmChar[256];

    // get arguments
    parseArgs(argv, argc, (char *)"--i", imagePath);
    parseArgs(argv, argc, (char *)"--n0", n0);
    parseArgs(argv, argc, (char *)"--it", it);
    parseArgs(argv, argc, (char *)"--xseeds", xseeds);
    parseArgs(argv, argc, (char *)"--yseeds", yseeds);
    parseArgs(argv, argc, (char *)"--file", fileObjSeeds);
    parseArgs(argv, argc, (char *)"--f", f);
    parseArgs(argv, argc, (char *)"--inverse", inverse);
    parseArgs(argv, argc, (char *)"--all", all);
    parseArgs(argv, argc, (char *)"--saveSeeds", fileSeeds);
    parseArgs(argv, argc, (char *)"--c1", c1_char);
    parseArgs(argv, argc, (char *)"--c2", c2_char);
    parseArgs(argv, argc, (char *)"--o", output);
    parseArgs(argv, argc, (char *)"--s", sampling);
    parseArgs(argv, argc, (char *)"--l", lists);
    parseArgs(argv, argc, (char *)"--clust", clust);
    parseArgs(argv, argc, (char *)"--max_markers", maxMarkers);
    parseArgs(argv, argc, (char *)"--obj_markers", objMarkers);
    parseArgs(argv, argc, (char *)"--draw", fileName);
    parseArgs(argv, argc, (char *)"--segm", segmChar);
    num_init_seeds = atoi(n0);
    iterations = atoi(it);
    function = atoi(f);

    if (strcmp(segmChar, "-") != 0)
    {
        int tmp = atoi(segmChar);
        if (tmp > 0 && tmp <= 4)
            segm = tmp;
    }

    // Validation of user's inputs
    if (strcmp(imagePath, "-") == 0 || strcmp(n0, "-") == 0 || strcmp(it, "-") == 0 || num_init_seeds < 0 ||
        (iterations <= 0 && segm != 3) || iterations < 0 ||
        (strcmp(xseeds, "-") != 0 && strcmp(yseeds, "-") != 0 && strcmp(fileObjSeeds, "-") != 0) ||
        (strcmp(xseeds, "-") == 0 && strcmp(yseeds, "-") != 0) || (strcmp(xseeds, "-") != 0 && strcmp(yseeds, "-") == 0) ||
        (strcmp(f, "-") == 0) || function <= 0 || function > 6)
    {
        usage(argv[0]);
        printError("main", "Too many/few parameters!");
    }

    if (strcmp(all, "-") != 0)
        all_borders = 1;

    if (strcmp(sampling, "-") != 0)
        sampling_method = atoi(sampling);
    else
        sampling_method = 0;

    if (strcmp(maxMarkers, "-") != 0)
        max_markers = atoi(maxMarkers);

    // Load image and get the user-defined params
    img = loadImage(imagePath);

    // Get object seeds coords in a txt file
    if (strcmp(fileObjSeeds, "-") != 0)
    {
        num_user_seeds = readMarkersFile(fileObjSeeds, max_markers, &marker_sizes, &coords_user_seeds, strcmp(inverse, "-"));
    }
    else
    {
        // get one point object seed
        if (strcmp(xseeds, "-") != 0 && strcmp(yseeds, "-") != 0)
        {
            num_user_seeds = 1;
            marker_sizes = (int *)allocMem(num_user_seeds, sizeof(int));
            marker_sizes[0] = 1;
            coords_user_seeds = (NodeCoords **)allocMem(num_user_seeds, sizeof(NodeCoords *));
            coords_user_seeds[0] = (NodeCoords *)allocMem(1, sizeof(NodeCoords));
            coords_user_seeds[0][0].x = atoi(xseeds);
            coords_user_seeds[0][0].y = atoi(yseeds);
        }
        else
        {
            // dont have object seed
            num_user_seeds = 0;
            coords_user_seeds = NULL;
        }
    }

    // Get only the image name
    pch = strtok(imagePath, "/");
    imageName = pch;
    while (pch != NULL)
    {
        imageName = pch;
        pch = strtok(NULL, "/");
    }
    pch = strtok(imageName, ".");
    imageName = pch;

    // Create auxiliary data structures
    border_img = createImage(img->num_rows, img->num_cols, 1);
    graph = createGraph(img);

    if (strcmp(fileSeeds, "-") != 0)
    {
        writeSeeds(img, num_init_seeds, coords_user_seeds, num_user_seeds, marker_sizes, fileSeeds);

    #pragma omp parallel for \
        private(i) \
        firstprivate(num_user_seeds) \
        shared(coords_user_seeds)
        for (i = 0; i < num_user_seeds; ++i)
            free(coords_user_seeds[i]);
        free(coords_user_seeds);
        freeGraph(&graph);
        return 0;
    }

    if (strcmp(objMarkers, "-") != 0)
        obj_markers = MIN(num_user_seeds, MAX(1, atoi(objMarkers)));
    else
        obj_markers = num_user_seeds;

    if (strcmp(c1_char, "-") != 0)
        c1 = atof(c1_char);
    else
        c1 = 0.7;

    if (strcmp(c2_char, "-") != 0)
        c2 = atof(c2_char);
    else
        c2 = 0.8;

    if (strcmp(fileName, "-") != 0)
    {
        draw_scribbles_clust(img, graph, coords_user_seeds, num_user_seeds, marker_sizes, obj_markers, fileName);
        //draw_scribbles_clust_bkp(img, graph, coords_user_seeds, num_user_seeds, marker_sizes, obj_markers, fileName);
        //draw_all_seeds(img, graph, coords_user_seeds, num_user_seeds, marker_sizes, obj_markers, num_init_seeds, fileName);
        //draw_scribbles_1clust(img, graph, coords_user_seeds, num_user_seeds, marker_sizes, obj_markers, fileName);
        //draw_markers(img, graph, coords_user_seeds, num_user_seeds, marker_sizes, obj_markers, fileName);

#pragma omp parallel for \
        private(i) \
        firstprivate(num_user_seeds) \
        shared(coords_user_seeds)
        for (i = 0; i < num_user_seeds; ++i)
            freeMem(coords_user_seeds[i]);

        freeGraph(&graph);
        freeImage(&img);
        freeMem(coords_user_seeds);
        return 0;
    }

    //clusterPoints(img, graph, coords_user_seeds, marker_sizes, num_user_seeds, obj_markers);

    if (segm == 1)
    {
        time = clock();
        label_img = runiDISF_scribbles_rem(graph, num_init_seeds, iterations, &border_img, coords_user_seeds, num_user_seeds, marker_sizes, function, all_borders, c1, c2, obj_markers);
        time = clock() - time;
    }
    else
    {
        if (segm == 2)
        {
            time = clock();
            label_img = runiDISF_scribbles_clust(graph, num_init_seeds, iterations, &border_img, coords_user_seeds, num_user_seeds, marker_sizes, function, all_borders, c1, c2, obj_markers, img);
            time = clock() - time;
        }
        else
        {
            if (segm == 3)
            {
                time = clock();
                label_img = runiDISF(graph, num_init_seeds, iterations, &border_img, coords_user_seeds, num_user_seeds, marker_sizes, function, all_borders, c1, c2, obj_markers);
                time = clock() - time;
            }
            else
            {
                //printf("running DISF \n");
                time = clock();
                label_img = runLabeledDISF(graph, num_init_seeds, iterations, coords_user_seeds, &border_img);
                time = clock() - time;
            }
        }
    }

    //int num_pixels;
    //num_pixels = label_img->num_pixels;
    //#pragma omp parallel for firstprivate(num_pixels)
    //for (int i = 0; i < num_pixels; i++)
    //    label_img->val[i][0]++;

    freeGraph(&graph);

    // Overlay superpixel's borders into a copy of the original image
    ovlay_img = overlayBorders(img, border_img);
    freeImage(&img);

    printf("%.3f\n\n", ((double)time) / CLOCKS_PER_SEC);

    // read the image name to write in results
    if (strcmp(output, "-") != 0)
    {
        sprintf(ovlayName, "%s_ovlay.ppm", output);
        sprintf(labelsName, "%s_labels.pgm", output);
        sprintf(bordersName, "%s_borders.pgm", output);
    }
    else
    {
        sprintf(ovlayName, "%s_n0-%d_it-%d_c1-%.1f_c2-%.1f_f%d_s%d_segm%d_ovlay.ppm", imageName, num_init_seeds, iterations, c1, c2, function, sampling_method, segm);
        sprintf(labelsName, "%s_n0-%d_it-%d_c1-%.1f_c2-%.1f_f%d_s%d_segm%d_labels.pgm", imageName, num_init_seeds, iterations, c1, c2, function, sampling_method, segm);
        sprintf(bordersName, "%s_n0-%d_it-%d_c1-%.1f_c2-%.1f_f%d_s%d_segm%d_borders.pgm", imageName, num_init_seeds, iterations, c1, c2, function, sampling_method, segm);
    }

    // Save the segmentation results
    writeImagePPM(ovlay_img, ovlayName);
    writeImagePGM(label_img, labelsName);
    writeImagePGM(border_img, bordersName);

// Free
#pragma omp parallel for \
    private(i) \
    firstprivate(num_user_seeds) \
    shared(coords_user_seeds) 
    for (i = 0; i < num_user_seeds; ++i)
    {
        freeMem(coords_user_seeds[i]);
    }
    freeImage(&label_img);
    freeImage(&border_img);
    freeImage(&ovlay_img);
    freeMem(coords_user_seeds);
}

//=============================================================================
// Image* Functions
//=============================================================================

Image *loadImage(const char *filepath)
{
    int num_channels, num_rows, num_cols, num_nodes;
    int i, j;
    unsigned char *data;
    Image *new_img;

    data = stbi_load(filepath, &num_cols, &num_rows, &num_channels, 0);

    if (data == NULL)
        printError("loadImage", "Could not load the image <%s>", filepath);

    new_img = createImage(num_rows, num_cols, num_channels);
    num_nodes = num_rows * num_cols;

#pragma omp parallel for private(i, j)  \
    firstprivate(num_nodes, num_channels) \
    shared(new_img, data)
    for (i = 0; i < num_nodes; i++)
    {
        new_img->val[i] = (int *)calloc(num_channels, sizeof(int));

        for (j = 0; j < num_channels; j++)
        {
            new_img->val[i][j] = data[i * num_channels + j];
        }
    }

    stbi_image_free(data);

    return new_img;
}

Image *overlayBorders(Image *img, Image *border_img)
{
    const float BORDER_sRGB[] = {0.0, 1.0, 1.0}; // Cyan

    int normval, i, j, num_channels, img_num_channels, num_pixels;
    Image *ovlay_img;

    normval = getNormValue(img);
    ovlay_img = createImage(img->num_rows, img->num_cols, 3);
    num_channels = ovlay_img->num_channels;
    img_num_channels = img->num_channels;
    num_pixels = img->num_pixels;

#pragma omp parallel for private(i, j)  \
    firstprivate(num_channels, num_pixels, img_num_channels, normval) \
        shared(ovlay_img, border_img, img, BORDER_sRGB)
    for (i = 0; i < num_pixels; i++)
    {
        for (j = 0; j < num_channels; j++)
        {
            if (border_img->val[i][0] != 0)
                ovlay_img->val[i][j] = BORDER_sRGB[j] * normval;

            else if (img_num_channels == 1) // It will convert the image to PPM
                ovlay_img->val[i][j] = img->val[i][0];
            else
                ovlay_img->val[i][j] = img->val[i][j];
        }
    }

    return ovlay_img;
}

//=============================================================================
// Void Functions
//=============================================================================
void writeImagePPM(Image *img, char *filepath)
{

    int max_val, min_val;
    FILE *fp;

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);

    fp = fopen(filepath, "wb");

    if (fp == NULL)
        printError("writeImagePPM", "Could not open the file %s", filepath);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PPM file
    if (max_val < 256 && min_val >= 0)
    {
        unsigned char *rgb;

        rgb = (unsigned char *)allocMem(img->num_channels, sizeof(unsigned char));

        for (int i = 0; i < img->num_pixels; i++)
        {
            for (int c = 0; c < img->num_channels; c++)
                rgb[c] = img->val[i][c];

            fwrite(rgb, 1, img->num_channels, fp);
        }

        freeMem(rgb);
    }
    // 16-bit PPM file
    else if (max_val < 65536 && min_val >= 0)
    {
        unsigned short *rgb;

        rgb = (unsigned short *)allocMem(img->num_channels, sizeof(unsigned short));

        for (int i = 0; i < img->num_pixels; i++)
        {
            for (int c = 0; c < img->num_channels; c++)
                rgb[c] = ((img->val[i][c] & 0xff) << 8) | ((unsigned short)img->val[i][c] >> 8);

            fwrite(rgb, 2, img->num_channels, fp);
        }

        freeMem(rgb);
    }
    else
        printError("writeImagePPM", "Invalid max and/or min vals %d, %d", max_val, min_val);

    fclose(fp);
}

void writeImagePGM(Image *img, char *filepath)
{
    int max_val, min_val, i;
    FILE *fp;
    int num_pixels;

    num_pixels = img->num_pixels;

    fp = fopen(filepath, "wb");

    if (fp == NULL)
        printError("writeImagePGM", "Could not open the file <%s>", filepath);

    max_val = getMaximumValue(img, -1);
    min_val = getMinimumValue(img, -1);

    fprintf(fp, "P5\n");
    fprintf(fp, "%d %d\n", img->num_cols, img->num_rows);
    fprintf(fp, "%d\n", max_val);

    // 8-bit PGM file
    if (max_val < 256 && min_val >= 0)
    {
        unsigned char *data;

        data = (unsigned char *)calloc(num_pixels, sizeof(unsigned char));

    #pragma omp parallel for private(i)  \
        firstprivate(num_pixels) \
        shared(data, img)
        for (i = 0; i < num_pixels; i++)
            data[i] = (unsigned char)img->val[i][0];

        fwrite(data, sizeof(unsigned char), num_pixels, fp);

        free(data);
    }
    // 16-bit PGM file
    else if (max_val < 65536 && min_val >= 0)
    {
        unsigned short *data;

        data = (unsigned short *)calloc(num_pixels, sizeof(unsigned short));

    #pragma omp parallel for private(i)  \
        firstprivate(num_pixels) \
        shared(data, img)
        for (i = 0; i < num_pixels; i++)
            data[i] = (unsigned short)img->val[i][0];

        for (i = 0; i < num_pixels; i++)
        {
            int high, low;

            high = ((data[i]) & 0x0000FF00) >> 8;
            low = (data[i]) & 0x000000FF;

            fputc(high, fp);
            fputc(low, fp);
        }

        free(data);
    }
    else
        printError("writeImagePGM", "Invalid min/max spel values <%d,%d>", min_val, max_val);

    fclose(fp);
}
