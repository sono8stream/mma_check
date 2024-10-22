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
*  @file     MMALIB_CNN_convolve_row_ixX_ixX_oxX_d.c
*
*  @brief    File to demnostrate usage of kernel
*
*  @version  0.1 - Jan 2018 : Initial Version - [Asheesh Bhardwaj]
*  @version  0.2 - Feb 2018 : Support for 8 bit [Asheesh Bhardwaj]
*  @version  0.3 - March 2018: Support for 16 bit, templates and 3x3 stride by 2 [Asheesh Bhardwaj]
*
*******************************************************************************
*/

// include MMALIB
#include <mmalib.h>

#include "MMALIB_CNN_convolve_row_ixX_ixX_oxX_d.h"

// include test infrastructure provided by MMALIB
#include "../../test/MMALIB_test.h"

// include test data for this kernel
#include "MMALIB_CNN_convolve_row_ixX_ixX_oxX_idat.h"

__attribute__((section(".msmcData"), aligned(64))) int8_t msmcBuffer[MMALIB_L3_RESULTS_BUFFER_SIZE];
__attribute__((section(".ddrData"), aligned(64))) int8_t ddrBuffer[2048 * 1024];

int16_t volatile volatileSum = 0; // use volatile to keep compiler from removing this operation

/* Test Driver Routine */
int MMALIB_CNN_convolve_row_ixX_ixX_oxX_d(uint32_t *pProfile, uint8_t LevelOfFeedback)
{
   int32_t tpi; /* test parameter index */
   int32_t fail = 0;
   uint64_t archCycles, estCycles;
   uint32_t num_pts = 0;
   /* Test Parameters */
   convolve_row_ixX_ixX_oxX_testParams_t *prm;

   convolve_row_ixX_ixX_oxX_getTestParams(&prm, &test_cases);

   MMALIB_DEBUGPRINTFN(1, "Number of test cases in this run: %d\n", test_cases);

   MMALIB_kernelHandle handle;
   int32_t handleSize;

   MMALIB_CNN_convolve_row_ixX_ixX_oxX_InitArgs kerInitArgs;
   MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecInArgs kerExecInArgs;
   MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs;

   /* Initialize profiling */
   TI_profile_init("convolve_row_ixX_ixX_oxX");

   /* Run each test case */
   for (tpi = 0; tpi < test_cases; tpi++)
   {
      // initialize status flags
      int32_t status_nat_vs_opt = TI_TEST_KERNEL_FAIL;      // test status : natural C vs. optimized C
      int32_t status_opt_vs_ref = TI_TEST_KERNEL_FAIL;      // test status : optimized C vs. static reference
      MMALIB_STATUS currTestStatus = MMALIB_STATUS_NOT_SET; /* Test current testcase status */

      uint32_t inp0Size, inp1Size, outSize;

      int32_t inWidth = prm[tpi].inWidth;
      int32_t inputBlockWidth = inWidth + prm[tpi].pad;
      int32_t inChOffset = prm[tpi].inChOffset;
      int32_t numInChannels = 1;//prm[tpi].numInChannels;
      int32_t numOutChannels = 1;//prm[tpi].numOutChannels;
      int32_t numOfOutputChKerBuf = 1;//prm[tpi].numOfOutputChKerBuf;
      int32_t subMChannels = 1;//prm[tpi].subMChannels;
      int32_t kernelWidth = prm[tpi].kernelWidth;
      int32_t kernelHeight = prm[tpi].kernelHeight;
      int32_t dilationWidth = prm[tpi].dilationWidth;
      int32_t dilationHeight = prm[tpi].dilationHeight;
      int32_t validColsIn = prm[tpi].validColsIn;
      int32_t strideWidth = prm[tpi].strideWidth;
      int32_t strideHeight = prm[tpi].strideHeight;
      int32_t kDim = 9;//prm[tpi].kDim;
      int32_t pitchA = 64;
      int32_t pitchC = prm[tpi].pitchC;
      uint8_t dataTypeA = prm[tpi].dataTypeA;
      uint8_t dataTypeB = prm[tpi].dataTypeB;
      uint8_t dataTypeC = prm[tpi].dataTypeC;
      uint8_t activationType = prm[tpi].activationType;
      int32_t pad = prm[tpi].pad;
      int32_t shift = 0;
      int32_t bias = prm[tpi].biasB;
      int32_t expectedStatusCode = prm[tpi].expectedStatusCode;
      uint8_t mode = prm[tpi].mode;
      int32_t circularOffset = prm[tpi].circularOffset;
      int32_t maxHeight = prm[tpi].maxHeight;
      int32_t totalN = prm[tpi].totalN;
      int32_t subN = prm[tpi].subN;
      int32_t validColsPerRowIn = prm[tpi].validColsPerRowIn;
      int32_t validRowsIn = prm[tpi].validRowsIn;
      int32_t outputPitchPerRow = prm[tpi].outputPitchPerRow;
      int32_t inputPitchPerRow = prm[tpi].inputPitchPerRow;
      int32_t validColsInlast = prm[tpi].validColsInlast;
      int32_t validRowsInlast = prm[tpi].validRowsInlast;
      int32_t validColsPerRowInlast = prm[tpi].validColsPerRowInlast;
      int32_t numGroupsPerKernel = 1;//prm[tpi].numGroupsPerKernel;
      int32_t MCounter = numOutChannels / subMChannels;
      MCounter = (numOutChannels % subMChannels == 0) ? MCounter - 1 : MCounter;

      int32_t testNum = prm[tpi].testID;
      int32_t numBytes = 1;
      int32_t MMA_SIZE = MMALIB_MMA_SIZE_8_BIT;

      if (dataTypeC == MMALIB_INT8 || dataTypeC == MMALIB_UINT8)
      {
         numBytes = 1;
         MMA_SIZE = MMALIB_MMA_SIZE_8_BIT;
      }
      else if (dataTypeC == MMALIB_INT16 || dataTypeC == MMALIB_UINT16)
      {
         numBytes = 2;
         MMA_SIZE = MMALIB_MMA_SIZE_16_BIT;
      }
      else
      {
         // TODO
      }

      // Function returns the memory for the handle to be allocated
      handleSize = MMALIB_CNN_convolve_row_ixX_ixX_oxX_getHandleSize(&kerInitArgs);
      MMALIB_DEBUGPRINTFN(1, "\nAfter MMALIB_CNN Handle Size before %d\n", handleSize);
      // Predicate buffer size is returned for allocating appropriate memory by application
      if (strideWidth == 1 && strideHeight == 1)
      {
         int32_t defaultPredSize = MMALIB_CNN_seamPredicateRegistersSizeDefault();
         int32_t actualPredSize = MMALIB_CNN_seamPredicateRegistersSize(inWidth, pad, maxHeight, MMA_SIZE, numOutChannels, subMChannels);

         if (actualPredSize > defaultPredSize)
         {
            handleSize = handleSize + actualPredSize - defaultPredSize;
         }
      }

      MMALIB_DEBUGPRINTFN(1, "\nAfter MMALIB_CNN Handle Size after %d\n", handleSize);
      handle = malloc(handleSize);

      int32_t strideShiftW = 0;
      if (strideWidth == 2)
         strideShiftW = 1;
      if (strideWidth == 4)
         strideShiftW = 2;

      int32_t strideShiftH = 0;
      if (strideHeight == 2)
         strideShiftH = 1;
      if (strideHeight == 4)
         strideShiftH = 2;

      // validColsOut calculated for non strided convolution
      int32_t validColsOut =
          ((validColsIn - inputBlockWidth * (kernelHeight * dilationHeight - 1)) >>
           strideShiftH) >>
          strideShiftW;
      int32_t validColsPerRow = 0;
      /* Compute buffer sizes */
      inp0Size = numOfOutputChKerBuf * pitchA * numGroupsPerKernel;
      inp1Size = numInChannels * inChOffset * numGroupsPerKernel * numBytes;
      outSize = numOutChannels * numGroupsPerKernel * pitchC * (totalN / subN);

      /* Allocate buffers for each test vector */
      MMALIB_DEBUGPRINTFN(1,
                          "pitchA %d, inChOffset %d, pitchC %d, inp0Size %d, inp1Size %d, outSize %d numGroupsPerKernel %d\n",
                          pitchA, inChOffset, pitchC, inp0Size, inp1Size, outSize, numGroupsPerKernel);

      MMALIB_DEBUGPRINTFN(1, "numOfOutputChKerBuf %d, numInChannels %d, numOutChannels %d, totalN %d, subN %d\n",
                          numOfOutputChKerBuf, numInChannels, numOutChannels, totalN, subN);

      MMALIB_bufParams2D_t src0_addr; // paramsWgt
      MMALIB_bufParams2D_t src1_addr; // input
      MMALIB_bufParams2D_t src2_addr; // bias
      MMALIB_bufParams3D_t dst_addr;
      int32_t numBiasVals = kDim - kernelWidth * kernelHeight * numInChannels;
      // Allocate coefficient buffers
      src0_addr.dim_x = kDim;
      src0_addr.dim_y = numOfOutputChKerBuf * numGroupsPerKernel;
      src0_addr.stride_y = pitchA;
      src0_addr.data_type = dataTypeA;

      // allocate input Channels
      src1_addr.dim_x = inChOffset;
      src1_addr.dim_y = numInChannels * numGroupsPerKernel;
      src1_addr.stride_y = inChOffset;
      src1_addr.data_type = dataTypeB;

      // allocate Bias per output channel
      src2_addr.dim_x = numBiasVals;
      src2_addr.dim_y = numOutChannels;
      src2_addr.stride_y = numBiasVals * numBytes;
      src2_addr.data_type = dataTypeA;

      // allocate output Channels
      dst_addr.dim_x = pitchC / numBytes;
      dst_addr.dim_y = numOutChannels;
      dst_addr.stride_y = pitchC;
      dst_addr.dim_z = numGroupsPerKernel;
      dst_addr.stride_z = numOutChannels * pitchC; // numOutChannels*pitchC = pitchG
      dst_addr.data_type = dataTypeC;

      // Assign to Init Args and Interface file
      kerInitArgs.Fc = kernelWidth;
      kerInitArgs.Fr = kernelHeight;
      kerInitArgs.dilationX = dilationWidth;
      kerInitArgs.dilationY = dilationHeight;
      kerInitArgs.strideX = strideWidth;
      kerInitArgs.strideY = strideHeight;
      kerInitArgs.validColsIn = validColsIn;
      kerInitArgs.subMChannels = subMChannels;
      kerInitArgs.inWidth = inWidth;
      kerInitArgs.pad = pad;
      kerInitArgs.maxHeight = maxHeight;
      kerInitArgs.inChOffset = inChOffset;
      kerInitArgs.shift = shift;
      kerInitArgs.No = numOutChannels;
      kerInitArgs.bias = bias;
      kerInitArgs.activationType = activationType;
      kerInitArgs.mode = mode;
      kerInitArgs.validColsPerRowIn = validColsPerRowIn;
      kerInitArgs.validRowsIn = validRowsIn;
      kerInitArgs.outputPitchPerRow = outputPitchPerRow;
      kerInitArgs.inputPitchPerRow = inputPitchPerRow;
      kerInitArgs.numGroupsPerKernel = numGroupsPerKernel;
      kerInitArgs.numBiasVals = numBiasVals;

      kerExecInArgs.bufferReset = 0;

      MMALIB_CNN_convolve_row_ixX_ixX_oxX_reorderWeights_Args reorderWeights;

      reorderWeights.dataType = src1_addr.data_type;
      reorderWeights.Ni = src1_addr.dim_y;
      reorderWeights.No = kerInitArgs.No;
      reorderWeights.Fr = kerInitArgs.Fr;
      reorderWeights.Fc = kerInitArgs.Fc;
      reorderWeights.strideX = kerInitArgs.strideX;
      reorderWeights.strideY = kerInitArgs.strideY;
      reorderWeights.dilationX = kerInitArgs.dilationX;
      reorderWeights.dilationY = kerInitArgs.dilationY;
      reorderWeights.numBiasVals = kerInitArgs.numBiasVals;
      reorderWeights.subMChannels = kerInitArgs.subMChannels;
      reorderWeights.numGroupsPerKernel = kerInitArgs.numGroupsPerKernel;
      reorderWeights.inWidth = kerInitArgs.inWidth;
      reorderWeights.pad = kerInitArgs.pad;
      reorderWeights.validColsIn = kerInitArgs.validColsIn;

      kerInitArgs.weightReorderFlag = MMALIB_CNN_convolve_row_reorderWeightsFlag(&reorderWeights);

      int32_t weightBufferSize;
      if (kerInitArgs.weightReorderFlag == 1)
      {
         weightBufferSize = MMALIB_CNN_convolve_row_reorderWeightsBufferSize(&src0_addr, &reorderWeights, &kerInitArgs);
      }
      else
      {
         weightBufferSize = 0;
      }

      int8_t *src0 = (int8_t*)0x64810000;

      int8_t *src1 =  (int8_t*)0x64820000;

      int8_t *dst = (int8_t*)0x64840000;

      int32_t biasSize = numOutChannels * numBiasVals * numBytes;
      int8_t *src2 = (int8_t*)0x64850000;
      
      // for debug
      {
         int iter=0;
         for(;iter<validColsIn;iter++){
            src1[iter]=iter%10;
         }
      }

      /* Only run the test if the buffer allocations fit in the heap */
      if (src0 && src1 && dst)
      {
         /* Fill input arrays according to desired test pattern */
         if (kerInitArgs.weightReorderFlag == 1 && prm[tpi].staticKernel != NULL)
         {
            TI_fillBuffer(prm[tpi].testPattern,
                          (uint8_t)255,
                          src2,
                          prm[tpi].staticKernel,
                          numOutChannels * numBiasVals,
                          1,
                          numOutChannels * numBiasVals,
                          numBytes,
                          testPatternString);

            MMALIB_CNN_convolve_row_reorderWeights(prm[tpi].staticKernel, src0, src2, &src0_addr, &src2_addr, &reorderWeights, &kerInitArgs);
         }
         else
         {
            TI_fillBuffer(prm[tpi].testPattern,
                          (uint8_t)255,
                          src0,
                          prm[tpi].staticKernel,
                          kDim,
                          numOfOutputChKerBuf * numGroupsPerKernel,
                          pitchA,
                          numBytes,
                          testPatternString);
         }

         /* This for creating the predicate buffers */
         int32_t totalBytes;
         if (kerInitArgs.strideX == 1)
         {
            totalBytes = MMALIB_CNN_generateFillSeamPredicateRegisters(
                handle, inWidth, pad, maxHeight, MMA_SIZE, numOutChannels, subMChannels);
         }

         MMALIB_DEBUGPRINTFN(1, "After MMALIB_CNN_generateFillSeamPredicateRegisters totalBytes %d\n", totalBytes);

         /* Initialize kernel */
         kerInitArgs.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
         TI_profile_start(TI_PROFILE_KERNEL_INIT);
         MMALIB_asm(" MARK 0");
         currTestStatus = MMALIB_CNN_convolve_row_ixX_ixX_oxX_init(
             handle, &src0_addr, &src1_addr, &dst_addr, &kerInitArgs);

         MMALIB_asm(" MARK 1");
         TI_profile_stop();

         /* Test optimized kernel */
         MMALIB_DEBUGPRINTFN(1, "Entering:MMALIB_CNN_convolve_row_ixX_ixX_oxX with currTestStatus = %d\n", currTestStatus);

         int32_t iterN = 0;
         int32_t validOutputRows;

         // subN is number of input pixels processed in one call and used to calcuate the starting address of feature map buffers
         // total N is total number of input pixels to be processed for a entire feature map
         // The test code is limited due memory available since there is no background data transfer
         // Verify upto two or 3 calls to kernel
         if (currTestStatus == MMALIB_SUCCESS)
         {
            int8_t *src1_Iter = src1;

            kerExecInArgs.subMChannels = subMChannels;

            // validColsInLast, validColsPerRowInlast, validRowsInlast should be kept same as validColsIn as in init phase.
            // This parameter is only for cases when last call parameters are different than what is initialized.
            kerExecInArgs.validColsIn = validColsInlast;
            kerExecInArgs.validColsPerRowIn = validColsPerRowInlast;
            kerExecInArgs.validRowsIn = validRowsInlast;
            kerExecInArgs.pad = pad;

            MMALIB_DEBUGPRINTFN(1, "src1_Iter %p dst_iter %p dst %p src0 %p\n", src1_Iter, dst_iter, dst, src0);
            MMALIB_DEBUGPRINTFN(1, "subMChannels %d, validColsIn %d MCount %d NCount %d subN %d\n",
                                 subMChannels, validColsIn, MCount, NCount, subN);
            
            // for debug
            int iter=0;
            for(;iter<kDim;iter++){
               src0[iter]=1;
            }
            
            long long startTsc=__TSC;
            TI_profile_start(TI_PROFILE_KERNEL_OPT);
            MMALIB_asm(" MARK 2");
            currTestStatus = MMALIB_CNN_convolve_row_ixX_ixX_oxX_exec (
                  handle, src0, src1, dst, &kerExecInArgs, &kerExecOutArgs);
            MMALIB_asm(" MARK 3");
            TI_profile_stop();                  
            long long endTsc=__TSC;
            long long elapsed=endTsc-startTsc;
            validColsOut = kerExecOutArgs.validColsOut;
            validColsPerRow = kerExecOutArgs.validColsPerRowOut;

            MMALIB_DEBUGPRINTFN(1, "OptC: valid cols out %d itenN %d\n", kerExecOutArgs.validColsOut, iterN);
            iterN++;
               
            __SE0_CLOSE();
            __SE1_CLOSE();

            validOutputRows = kerExecOutArgs.validRowsOut;
         }
      }
   }

   pProfile[3 * tpi] = (int32_t)TI_profile_get_cycles(TI_PROFILE_KERNEL_OPT);
   pProfile[3 * tpi + 1] =
         (int32_t)TI_profile_get_cycles(TI_PROFILE_KERNEL_OPT_WARM);
   pProfile[3 * tpi + 2] =
         (int32_t)TI_profile_get_cycles(TI_PROFILE_KERNEL_OPT_WARMWRB);
}

int test_main(uint32_t *pProfile)
{
   if (TI_cache_init())
   {
      TI_memError("MMALIB_CNN_convolve_row_ixX_ixX_oxX");
      return 1;
   }
   else
   {
      return MMALIB_CNN_convolve_row_ixX_ixX_oxX_d(&pProfile[0], 0);
   }
}

/* Main call for individual test projects */

int mainConv()
{
   int fail = 1;

   uint32_t profile[256 * 3];

   MMALIB_TEST_init();

   fail = test_main(&profile[0]);

#if !defined(NO_PRINTF)
   if (fail == 0)
      printf("Test Pass!\n");
   else
      printf("Test Fail!\n");

   int i;
   for (i = 0; i < test_cases; i++)
   {
      printf("Test %4d: Cold Cycles = %8d, Warm Cycles = %8d, Warm Cycles WRB = %8d\n",
             i,
             profile[3 * i],
             profile[3 * i + 1],
             profile[3 * i + 2]);
   }
#endif

   return fail;
}
/* ======================================================================== */
/*  End of file:  MMALIB_CNN_convolve_row_ixX_ixX_oxX_d.c                   */
/* ======================================================================== */
