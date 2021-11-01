/**
* Utilities
*
* @date September, 2019
*/
#ifndef UTILS_H
#define UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Includes
//=============================================================================
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

//=============================================================================
// Void Functions
//=============================================================================
/**
* Safely frees the allocated memory
*/
void freeMem(void* data);

/**
* Prints an error message and aborts the program.
*/
void printError(const char* function_name, const char* message, ...);

/**
* Prints a warning message and does not abort the program.
*/
void printWarning(const char* function_name, const char* message, ...);

/**
* Find a string key in arguments and storage in argument variable
*/
void parseArgs(char *argv[], int argc, char *stringKey, char *argument);

//=============================================================================
// Void* Functions
//=============================================================================
/**
* Tries to allocate the memory and verifies if it was sucessful. If
* not, it will abort the program.
*/
void* allocMem(size_t size, size_t size_bytes);


#ifdef __cplusplus
}
#endif

#endif