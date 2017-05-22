/***************************************************************************
Copyright (c) 2013-2016, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/

/**************************************************************************************
* 2016/03/25 Werner Saar (wernsaar@googlemail.com)
* 	 BLASTEST 		: OK
* 	 CTEST			: OK
* 	 TEST			: OK
*	 LAPACK-TEST		: OK
**************************************************************************************/

#define HAVE_KERNEL_8 1

static void dscal_kernel_8( BLASLONG n, FLOAT *alpha, FLOAT *x) __attribute__ ((noinline));

static void dscal_kernel_8( BLASLONG n, FLOAT *alpha, FLOAT *x)
{


	BLASLONG i = n;
	BLASLONG o16 = 16;
	BLASLONG o32 = 32;
	BLASLONG o48 = 48;
	BLASLONG o64 = 64;
	BLASLONG o80 = 80;
	BLASLONG o96 = 96;
	BLASLONG o112 = 112;
	FLOAT *x1=x;
	FLOAT *x2=x+1;
	BLASLONG pre = 384;

	__asm__  __volatile__
	(

        "lxsdx          33, 0, %3                           \n\t"
        "xxspltd        32, 33, 0                           \n\t"
        "addi           %1, %1, -8                          \n\t"

	"dcbt		%2, %4				    \n\t"

	"lxvd2x		40, 0, %2			    \n\t"
	"lxvd2x		41, %5, %2			    \n\t"
	"lxvd2x		42, %6, %2			    \n\t"
	"lxvd2x		43, %7, %2			    \n\t"
	"lxvd2x		44, %8, %2			    \n\t"
	"lxvd2x		45, %9, %2			    \n\t"
	"lxvd2x		46, %10, %2			    \n\t"
	"lxvd2x		47, %11, %2			    \n\t"

	"addi		%2, %2, 128			    \n\t"

	"addic.		%0 , %0	, -16  	 	             \n\t"
	"ble		2f		             	     \n\t"

	".align 5				            \n\t"
	"1:				                    \n\t"

	"dcbt		%2, %4				    \n\t"

	"xvmuldp	48, 40, 32		    	    \n\t"
	"xvmuldp	49, 41, 32		    	    \n\t"
	"lxvd2x		40, 0, %2			    \n\t"
	"lxvd2x		41, %5, %2			    \n\t"
	"xvmuldp	50, 42, 32		    	    \n\t"
	"xvmuldp	51, 43, 32		    	    \n\t"
	"lxvd2x		42, %6, %2			    \n\t"
	"lxvd2x		43, %7, %2			    \n\t"
	"xvmuldp	52, 44, 32		    	    \n\t"
	"xvmuldp	53, 45, 32		    	    \n\t"
	"lxvd2x		44, %8, %2			    \n\t"
	"lxvd2x		45, %9, %2			    \n\t"
	"xvmuldp	54, 46, 32		    	    \n\t"
	"xvmuldp	55, 47, 32		    	    \n\t"
	"lxvd2x		46, %10, %2			    \n\t"
	"lxvd2x		47, %11, %2			    \n\t"

	"stxvd2x	48, 0, %1			    \n\t"
	"stxvd2x	49, %5, %1			    \n\t"
	"stxvd2x	50, %6, %1			    \n\t"
	"stxvd2x	51, %7, %1			    \n\t"
	"stxvd2x	52, %8, %1			    \n\t"
	"stxvd2x	53, %9, %1			    \n\t"
	"stxvd2x	54, %10, %1			    \n\t"
	"stxvd2x	55, %11, %1			    \n\t"

	"addi		%1, %1, 128			    \n\t"
	"addi		%2, %2, 128			    \n\t"

	"addic.		%0 , %0	, -16  	 	             \n\t"
	"bgt		1b		             	     \n\t"

	"2:						     \n\t"

	"xvmuldp	48, 40, 32		    	    \n\t"
	"xvmuldp	49, 41, 32		    	    \n\t"
	"xvmuldp	50, 42, 32		    	    \n\t"
	"xvmuldp	51, 43, 32		    	    \n\t"
	"xvmuldp	52, 44, 32		    	    \n\t"
	"xvmuldp	53, 45, 32		    	    \n\t"
	"xvmuldp	54, 46, 32		    	    \n\t"
	"xvmuldp	55, 47, 32		    	    \n\t"

	"stxvd2x	48, 0, %1			    \n\t"
	"stxvd2x	49, %5, %1			    \n\t"
	"stxvd2x	50, %6, %1			    \n\t"
	"stxvd2x	51, %7, %1			    \n\t"
	"stxvd2x	52, %8, %1			    \n\t"
	"stxvd2x	53, %9, %1			    \n\t"
	"stxvd2x	54, %10, %1			    \n\t"
	"stxvd2x	55, %11, %1			    \n\t"

	:
        : 
          "r" (i),	// 0	
	  "r" (x2),  	// 1
          "r" (x1),     // 2
          "r" (alpha),  // 3
          "r" (pre),    // 4
	  "r" (o16),	// 5
	  "r" (o32),	// 6
	  "r" (o48),    // 7
          "r" (o64),    // 8
          "r" (o80),    // 9
          "r" (o96),    // 10
          "r" (o112)    // 11
	: "cr0", "%0", "%2" , "%1", "memory"
	);

} 


static void dscal_kernel_8_zero( BLASLONG n, FLOAT *alpha, FLOAT *x) __attribute__ ((noinline));

static void dscal_kernel_8_zero( BLASLONG n, FLOAT *alpha, FLOAT *x)
{


	BLASLONG i = n;
	BLASLONG o16 = 16;
	BLASLONG o32 = 32;
	BLASLONG o48 = 48;
	BLASLONG o64 = 64;
	BLASLONG o80 = 80;
	BLASLONG o96 = 96;
	BLASLONG o112 = 112;
	FLOAT *x1=x;
	FLOAT *x2=x+1;
	BLASLONG pre = 384;

	__asm__  __volatile__
	(

	"xxlxor		32 , 32 , 32			    \n\t"
        "addi           %1, %1, -8                          \n\t"


	".align 5				            \n\t"
	"1:				                    \n\t"

	"stxvd2x	32, 0, %1			    \n\t"
	"stxvd2x	32, %5, %1			    \n\t"
	"stxvd2x	32, %6, %1			    \n\t"
	"stxvd2x	32, %7, %1			    \n\t"
	"stxvd2x	32, %8, %1			    \n\t"
	"stxvd2x	32, %9, %1			    \n\t"
	"stxvd2x	32, %10, %1			    \n\t"
	"stxvd2x	32, %11, %1			    \n\t"

	"addi		%1, %1, 128			    \n\t"

	"addic.		%0 , %0	, -16  	 	             \n\t"
	"bgt		1b		             	     \n\t"

	"2:						     \n\t"

	:
        : 
          "r" (i),	// 0	
	  "r" (x2),  	// 1
          "r" (x1),     // 2
          "r" (alpha),  // 3
          "r" (pre),    // 4
	  "r" (o16),	// 5
	  "r" (o32),	// 6
	  "r" (o48),    // 7
          "r" (o64),    // 8
          "r" (o80),    // 9
          "r" (o96),    // 10
          "r" (o112)    // 11
	: "cr0", "%0", "%2" , "%1", "memory"
	);

} 


