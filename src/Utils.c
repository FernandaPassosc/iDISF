#include "Utils.h"
#include <string.h>

/*
float** getDistinctRGBColors(int n)
{
    float colors[n][3];
    // H : 0 - 359  S : 0 - 1 L : 0 - 1
    // each color has 30 degrees in H, so 360/15 = 12 pure colors
    // for S and V, we divide each one in 10 pieces.

    for(int i=0; i < n+1; i++){
        h = i%
    }
    return colors;
}
*/

//=============================================================================
// Void Functions
//=============================================================================
void freeMem(void* data)
{
    if(data != NULL) 
        free(data);
}

void printError(const char* function_name, const char* message, ...)
{
    va_list args;
    char full_msg[4096];

    va_start(args, message);
    vsprintf(full_msg, message, args);
    va_end(args);

    fprintf(stderr, "\nError in %s:\n%s!\n", function_name, full_msg);
    fflush(stdout);
    exit(0);
}

void printWarning(const char *function_name, const char *message, ...)
{
    va_list args;
    char full_msg[4096];

    va_start(args, message);
    vsprintf(full_msg, message, args);
    va_end(args);

    fprintf(stdout, "\nWarning in %s:\n%s!\n", function_name, full_msg);
}


/*
Find a string key in arguments and storage in argument variable
Input: argv and argc received in main, stringKey to match, argument variable to store match value
Output: argument variable with the match value or NULL
*/
void parseArgs(char *argv[], int argc, char *stringKey, char *argument){
    for(int i=1; i < argc-1; i++){
        // if both strings are identical
        if(strcmp(argv[i], stringKey) == 0){
            strcpy(argument, argv[i+1]);
            return;
        }
    }
    sprintf(argument, "-");
}


//=============================================================================
// Void* Functions
//=============================================================================
void* allocMem(size_t size, size_t size_bytes)
{
    void *data;

    data = calloc(size, size_bytes);

    if(data == NULL)
        printError("allocMemory", "Could not allocate memory");

    return data;
}