#pragma once

/* /////////////////////////////////////////////////////
*	Specify the compiler type (for now the supported
*	compilers are Visual C++, NVCC)
* /////////////////////////////////////////////////////
*/

#define IMPACTS_COMPILER_UNKNOWN	0x0000

// VC++ defines
#define IMPACTS_COMPILER_VC2012		0x1000
#define IMPACTS_COMPILER_VC2013		0x1010
#define IMPACTS_COMPILER_VC2015		0x1020

// CUDA defines
#define IMPACTS_COMPILER_CUDA		0x2000

// CUDA
#ifdef __CUDACC__
#	if !defined(CUDA_VERSION) && !defined(IMPACTS_FORCE_CUDA)
#		include <cuda_runtime.h>
#	endif
#	if CUDA_VERSION < 7050
#		error "This implementation of Dividing Cubes requires CUDA v7.5 or higher"
#	else
#		define IMPACTS_COMPILER IMPACTS_COMPILER_CUDA
#	endif

// Visual C++
#elif defined(_MSC_VER)
#	if _MSC_VER < 1700
#		error "This implementation of Dividing Cubes requires Visual C++ 2012 or higher"
#	elif _MSC_VER == 1700
#		define IMPACTS_COMPILER IMPACTS_COMPILER_VC2012
#	elif _MSC_VER == 1800
#		define IMPACTS_COMPILER IMPACTS_COMPILER_VC2013
#	elif _MSC_VER == 1900
#		define IMPACTS_COMPILER IMPACTS_COMPILER_VC2015
#	else
#		error "Unknown version of Visual C++"
#	endif

#else
#	define IMPACTS_COMPILER IMPACTS_COMPILER_UNKNOWN

#endif

#ifndef IMPACTS_COMPILER
#	error "Unsupported compiler"
#endif


/* /////////////////////////////////////////////////////
*	Specify function qualifiers or declarations, i.e., whether the function
*	(or method) is supported for both CPU and GPU
* /////////////////////////////////////////////////////
*/

#if IMPACTS_COMPILER & IMPACTS_COMPILER_CUDA	// & means whether IMPACTS_COMPILER == IMPACTS_COMPILER_CUDA or not
#	define IMPACTS_CUDA_FUNC_DECL __device__ __host__
#else
#	define IMPACTS_CUDA_FUNC_DECL
#endif

#define IMPACTS_FUNC_DECL IMPACTS_CUDA_FUNC_DECL