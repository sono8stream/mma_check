/******************************************************************************
*                                                                             *
* module name       :MMALIB                                                   *
*                                                                             *
* module descripton :Matrix Multiply Accelerator Library module for C7x+MMA   *
*                                                                             *
* Copyright (C) 2017-2018 Texas Instruments Incorporated - http://www.ti.com/ *
* ALL RIGHTS RESERVED                                                         *
*                                                                             *
******************************************************************************/

/**
******************************************************************************
*  @file     MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_idat.h
*
*  @brief    File to have static test data related structure
*
*  @version  0.1 - Feb 2018 : Initial Version 
*            
*******************************************************************************
*/

#ifndef MMALIB_CNN_CONVOLVE_COL_SMALLNO_IXX_IXX_OXX_IDAT_H_
#define MMALIB_CNN_CONVOLVE_COL_SMALLNO_IXX_IXX_OXX_IDAT_H_

#include <mmalib.h>


// include test infrastructure provided by MMALIB
#include "../../test/MMALIB_test.h"

typedef struct {
   uint8_t testPattern;   /* 0: constant, 1: sequential, 2: random, 3: static array, 4: file, etc */
   void  *staticKernel;
   void  *staticBias;
   void  *staticIn;
   uint8_t *staticShiftValues;
   int32_t *staticBiasBValues;
   void  *staticOut;
   void  *reserved0;
   void  *reserved1;
   int32_t kernelWidth;
   int32_t kernelHeight;
   int32_t strideX;
   int32_t strideY;
   int32_t dilationX;
   int32_t dilationY;
   int32_t Lc;
   int32_t Lr;
   int32_t numInChannels;
   int32_t numOutChannels;
   int32_t numGroupsPerKernel;
   uint8_t qShift;
   int8_t  shiftMethod;
   int32_t expectedStatusCode;
   int32_t biasB;
   int32_t numBiasVals;
   uint8_t  activationType;
   uint32_t dataTypeA;
   uint32_t dataTypeB;
   uint32_t dataTypeC;
   int32_t outputDataLocation; // 0 -> HEAP (probably L2SRAM), 1 -> MSMC
   int32_t numReps;
   uint32_t testID;
} MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_testParams_t;

void MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_getTestParams(MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_testParams_t * *params, int32_t *numTests);

#endif /* define MMALIB_CNN_CONVOLVE_COL_SMALLNO_IXX_IXX_OXX_IDAT_H_ */

/* ======================================================================== */
/*  End of file:  MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_idat.h                */
/* ======================================================================== */

