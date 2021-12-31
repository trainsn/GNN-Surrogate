#pragma once
#ifndef _DEF_H
#define _DEF_H

// needed for malloc
#include <cstdlib>

// NetCDF error handling
#define NC_SAFE_CALL(call) {\
  int retval = call;\
  if (retval != 0) {\
      fprintf(stderr, "[NetCDF Error] %s, in file '%s', line %i.\n", nc_strerror(retval), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
  }\
}

// PNetCDF error handling
#define PNC_SAFE_CALL(call) {\
  int retval = call;\
  if (retval != 0) {\
      fprintf(stderr, "[PNetCDF Error] %s, in file '%s', line %i.\n", ncmpi_strerror(retval), __FILE__, __LINE__); \
      exit(EXIT_FAILURE); \
  }\
}

// CUDA error handling
#define CUDA_SAFE_CALL(call) {\
  cudaError_t err = call;\
  if (cudaSuccess != err) {\
    fprintf(stderr, "[CUDA Error] %s, in file '%s', line%i.\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    exit(EXIT_FAILURE); \
  }\
}

// OpenGL error handling
#define RESET_GLERROR()\
{\
  while (glGetError() != GL_NO_ERROR) {}\
}

#define CHECK_GLERROR()\
{\
  GLenum err = glGetError();\
  if (err != GL_NO_ERROR) {\
    const GLubyte *errString = gluErrorString(err);\
    qDebug("[%s line %d] GL Error: %s\n",\
            __FILE__, __LINE__, errString);\
  }\
}

// mem alloc for 2D arrays
#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)

#endif
