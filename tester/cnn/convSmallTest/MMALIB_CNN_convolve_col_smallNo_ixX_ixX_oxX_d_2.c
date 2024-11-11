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

/********************************************************************************
 *  @file     MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_d.c
 *
 *  @brief    File to demnostrate usage of kernel
 *
 *  @version  0.1 - Feb 2018 : Initial Version
 *
 ********************************************************************************/

// include MMALIB
#include <mmalib.h>

#include "MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_d.h"

// include test infrastructure provided by MMALIB
#include "../../test/MMALIB_test.h"

// include test data for this kernel
#include "MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_idat.h"

#define FALSE 0
#define TRUE 1
#define FORCE_EVEN_GROUPS FALSE

__attribute__((section(".msmcData"), aligned(64))) int8_t msmcBuffer[MMALIB_L3_RESULTS_BUFFER_SIZE];

__attribute__((section(".ddrData"), aligned(64))) int8_t ddrBuffer[2048 * 1024];

int16_t volatile volatileSum = 0; // use volatile to keep compiler from removing this operation, move to global to prevent compiler warning of unused variable

int MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_d(uint32_t *pProfile, uint8_t LevelOfFeedback)
{
   int32_t tpi;      /* test parameter index */
   int32_t fail = 0; /* fail flag */
   int32_t repCount; /* number of times to repeat a test; useful for random data and random sizes (set to 1 for static) */
   int32_t numReps;

   // variables for calculating performance estimates
   uint64_t archCycles, estCycles;
   uint32_t numPts = 0;

   /* Test Parameters */
   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_testParams_t *prm;
   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_testParams_t currPrm;
   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_getTestParams(&prm, &test_cases);

   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_InitArgs kerInitArgs;
   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_ExecInArgs kerExecInArgs;
   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs_ci;
   MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs_cn;

   /* Initialize profiling */
   TI_profile_init("MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX");

   /* Run each test case */
   for (tpi = 0; tpi < test_cases; tpi++)
   {
      /* Number of repetitions for the test */
      currPrm = prm[tpi];

      int32_t handleSize = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_getHandleSize(&kerInitArgs);
      MMALIB_kernelHandle handle = (MMALIB_kernelHandle)malloc(handleSize);

      /* Initialize status flags */
      MMALIB_STATUS currTestStatus = MMALIB_STATUS_NOT_SET; /* Test current testcase status*/

      uint32_t inp0Size, inp1Size, outSize;
      // HACK ALERT - setting this parameter allows testing for non-continguous groups for all devices except C7100, which requires contiguous groups; hack here because it is not implemented in the idat yet
      // assumed to represent a number of extra rows in the memory storage of the input feature map
      int32_t gapRowsBetweenGroups = 0;

      kerInitArgs.Ni = currPrm.numInChannels;
      kerInitArgs.No = currPrm.numOutChannels;
      kerInitArgs.Fr = currPrm.kernelWidth;
      kerInitArgs.Fc = currPrm.kernelHeight;
      kerInitArgs.strideX = currPrm.strideX;
      kerInitArgs.strideY = currPrm.strideY;
      kerInitArgs.dilationX = currPrm.dilationX;
      kerInitArgs.dilationY = currPrm.dilationY;
      int32_t eFc = kerInitArgs.dilationX * (kerInitArgs.Fc - 1) + 1; // equivalent filter width for dilation
      int32_t eFr = kerInitArgs.dilationY * (kerInitArgs.Fr - 1) + 1; // equivalent filter height for dilation
      uint8_t dataTypeA = currPrm.dataTypeA;
      uint8_t dataTypeB = currPrm.dataTypeB;
      uint8_t dataTypeC = currPrm.dataTypeC;
      kerInitArgs.activationType = currPrm.activationType;
      kerInitArgs.shift = currPrm.qShift;
      kerInitArgs.shiftMethod = currPrm.shiftMethod;
      kerInitArgs.bias = currPrm.biasB;
      kerInitArgs.biasDataType = dataTypeB;
      kerInitArgs.numBiasVals = currPrm.numBiasVals;
      kerInitArgs.featureWidth = currPrm.Lc;
      kerInitArgs.blockFeatureHeight = currPrm.Lr;

      kerInitArgs.blockFeaturePitch = kerInitArgs.featureWidth * MMALIB_sizeof(dataTypeB) * kerInitArgs.Ni;
      kerInitArgs.columnOffset = MMALIB_MMA_SIZE_8_BIT / MMALIB_sizeof(dataTypeB) * 2 * kerInitArgs.strideX; // should make this an idat.c parameter
      kerInitArgs.inPairOffset = MMALIB_MMA_SIZE_8_BIT / MMALIB_sizeof(dataTypeB) * kerInitArgs.strideX;     // should make this an idat.c parameter
      kerInitArgs.inChOffset = kerInitArgs.featureWidth;                                                     // temporary until it's implemented as an idat.c parameter
      kerInitArgs.outPairOffset = MMALIB_MMA_SIZE_8_BIT / MMALIB_sizeof(dataTypeC);
      kerInitArgs.numGroupsPerKernel = currPrm.numGroupsPerKernel;
      int32_t Mr = MMALIB_ceilingDiv(currPrm.Lr - eFr + 1, currPrm.strideY);
      int32_t Mc = MMALIB_ceilingDiv(currPrm.Lc - eFc + 1, currPrm.strideX);

      MMALIB_bufParams2D_t weights_addr; // filter coefficients in packed, natural order
      MMALIB_bufParams2D_t bias_addr;    // bias values in packed, natural order

      MMALIB_bufParams2D_t src0_addr;            // filter coefficients
      MMALIB_bufParams2D_t src1_addr, src1_lddr; // input feature maps

      MMALIB_bufParams3D_t dst_addr; // output feature maps, expand to 3D to support groups

      /* Compute data sizes */
      MMALIB_bufParams2D_t *pBias_addr;
      if (currPrm.staticBias == NULL)
      {
         // bias is merged with weights
         // kernel filter weights
         weights_addr.dim_x = kerInitArgs.Ni * kerInitArgs.Fc * kerInitArgs.Fr + kerInitArgs.numBiasVals;
         weights_addr.dim_y = kerInitArgs.No * kerInitArgs.numGroupsPerKernel;
         weights_addr.data_type = dataTypeA;
         weights_addr.stride_y = weights_addr.dim_x * MMALIB_sizeof(weights_addr.data_type);
         pBias_addr = NULL;
      }
      else
      {
         // kernel filter weights
         weights_addr.dim_x = kerInitArgs.Ni * kerInitArgs.Fc * kerInitArgs.Fr;
         weights_addr.dim_y = kerInitArgs.No * kerInitArgs.numGroupsPerKernel;
         weights_addr.data_type = dataTypeA;
         weights_addr.stride_y = weights_addr.dim_x * MMALIB_sizeof(weights_addr.data_type);

         bias_addr.dim_x = kerInitArgs.numBiasVals;
         bias_addr.dim_y = kerInitArgs.No * kerInitArgs.numGroupsPerKernel;
         bias_addr.data_type = dataTypeA;
         bias_addr.stride_y = bias_addr.dim_x * MMALIB_sizeof(bias_addr.data_type);

         pBias_addr = &bias_addr;
      }

      // calculate the buffer parameters for the reordered weights
      MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_reorderWeights_Args reorderWeightsArgs;
      reorderWeightsArgs.dataType = dataTypeA;
      reorderWeightsArgs.Ni = kerInitArgs.Ni;
      reorderWeightsArgs.No = kerInitArgs.No;
      reorderWeightsArgs.Fr = kerInitArgs.Fr;
      reorderWeightsArgs.Fc = kerInitArgs.Fc;
      reorderWeightsArgs.strideX = kerInitArgs.strideX;
      reorderWeightsArgs.strideY = kerInitArgs.strideY;
      reorderWeightsArgs.dilationX = kerInitArgs.dilationX;
      reorderWeightsArgs.featureWidth = kerInitArgs.featureWidth;
      reorderWeightsArgs.numBiasVals = kerInitArgs.numBiasVals;
      reorderWeightsArgs.numGroupsPerKernel = kerInitArgs.numGroupsPerKernel;

      // calculate the memory size for allocation
      inp0Size = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_reorderWeights_getMemorySize(&reorderWeightsArgs);
      // fill the bufParams for src0
      MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_reorderWeights_fillBufParams(&reorderWeightsArgs, &src0_addr);

      // currently assuming the feature map as 2D
      src1_addr.dim_x = kerInitArgs.Ni * currPrm.Lc; // input feature map
      int32_t LrWithGapRows = currPrm.Lr + gapRowsBetweenGroups;
      src1_addr.dim_y = LrWithGapRows * kerInitArgs.numGroupsPerKernel;
      src1_addr.data_type = dataTypeB;
      src1_addr.stride_y = kerInitArgs.blockFeaturePitch;
      // make sure the numGroupsPerKernel is even, but for loading source data we need the actual data size.
      src1_lddr = src1_addr;
      src1_lddr.dim_y = currPrm.Lr * kerInitArgs.numGroupsPerKernel;

      dst_addr.dim_x = Mc;
      dst_addr.dim_y = kerInitArgs.No * Mr;
      dst_addr.dim_z = kerInitArgs.numGroupsPerKernel;
      dst_addr.data_type = dataTypeC;
#if defined(MINIMUM_OUTPUT_SPACE)
      dst_addr.stride_y = Mc * MMALIB_sizeof(dataTypeC); // pack output rows for minimum storage space
#else
      dst_addr.stride_y = MMALIB_CALC_STRIDE(Mc * MMALIB_sizeof(dataTypeC), MMALIB_ALIGN_SHIFT_64BYTES); // align every row to 64 bytes;
#endif
      dst_addr.stride_z = kerInitArgs.No * dst_addr.stride_y * Mr;
      outSize = dst_addr.dim_z * dst_addr.stride_z;

      /* Allocate buffers for each test vector */
      // this works-around Valgrind warnings for predicated SA-based loads.  This should eventually be fixed in compiler/HE models.
      inp0Size = MMALIB_CALC_STRIDE(inp0Size + 64, MMALIB_ALIGN_SHIFT_64BYTES);
      void *src0 = TI_memalign(MMALIB_ALIGN_128BYTES, inp0Size); // weights

      inp1Size = src1_addr.dim_y * src1_addr.stride_y;
      inp1Size = MMALIB_ceilingDiv(inp1Size, 64) * 64;
      void *src1 = TI_memalign(MMALIB_ALIGN_128BYTES, inp1Size);
      void *pSrc1 = src1;

      uint8_t *shiftValues;
      if (currPrm.staticShiftValues != NULL)
      {
         shiftValues = (uint8_t *)TI_memalign(MMALIB_ALIGN_128BYTES, kerInitArgs.numGroupsPerKernel);
         int32_t i;
         for (i = 0; i < kerInitArgs.numGroupsPerKernel; i++)
         {
            shiftValues[i] = currPrm.staticShiftValues[i];
         }

         // this is a cheap hack until the driver file supports per group shift
         // this is to allow the dynamic range comparison logic to work
         currPrm.qShift = shiftValues[0];
      }
      else
      {
         shiftValues = NULL;
      }

      int32_t *biasBValues = NULL;
      if (currPrm.staticBiasBValues != NULL)
      {
         biasBValues = (int32_t *)TI_memalign(MMALIB_ALIGN_128BYTES, kerInitArgs.numGroupsPerKernel * sizeof(int32_t));
         int32_t i;
         for (i = 0; i < kerInitArgs.numGroupsPerKernel; i++)
         {
            biasBValues[i] = currPrm.staticBiasBValues[i];
         }
      }
      else
      {
         biasBValues = NULL;
      }

      void *dst;

      if (currPrm.outputDataLocation == MMALIB_TEST_OUTPUT_HEAP)
      {
         dst = TI_memalign(MMALIB_ALIGN_128BYTES, outSize);
      }
      else if (currPrm.outputDataLocation == MMALIB_TEST_OUTPUT_MSMC)
      {
         dst = (void *)msmcBuffer;
      }
      else
      {
         dst = NULL;
      }

      uint32_t i;
      // initialize all kernel coefficient memory even if it won't be used to address valgrind warnings
      for (i = 0; i < inp0Size; i++)
      {
         ((uint8_t *)src0)[i] = 5;
      }
      for (i = 0; i < inp1Size; i++)
      {
         ((uint8_t *)src1)[i] = 6;
      }

      void *dst_cn = NULL;
      if (currPrm.outputDataLocation == MMALIB_TEST_OUTPUT_HEAP)
      {
         dst_cn = TI_memalign(MMALIB_ALIGN_128BYTES, outSize);
      }
      else
      {
         dst_cn = (void *)ddrBuffer;
      }

      /* Only run the test if the buffer allocations fit in the heap */
      if (src0 && src1 && dst && dst_cn)
      {
         /* copy static test data into memory, or generate random test data */
         // copy/generate the filter coefficients
         reorderWeightsArgs.blockFeatureHeight = kerInitArgs.blockFeatureHeight;

         if (tpi % 2 == 0)
         {
            MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_reorderWeights_exec(REORDER_WEIGHTS_AND_BIAS, &reorderWeightsArgs, &weights_addr, currPrm.staticKernel, pBias_addr, currPrm.staticBias, &src0_addr, src0);
         }
         else
         {
            MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_reorderWeights_exec(REORDER_WEIGHTS, &reorderWeightsArgs, &weights_addr, currPrm.staticKernel, pBias_addr, currPrm.staticBias, &src0_addr, src0);
            MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_reorderWeights_exec(REORDER_BIAS, &reorderWeightsArgs, &weights_addr, currPrm.staticKernel, pBias_addr, currPrm.staticBias, &src0_addr, src0);
         }
         MMALIB_DEBUGPRINTFN(1, "currPrm.staticBias = %p\n", currPrm.staticBias);

         // copy/generate the input feature maps
         TI_fillBuffer(currPrm.testPattern,
                       (uint8_t)255,
                       pSrc1, currPrm.staticIn,
                       src1_lddr.dim_x, src1_lddr.dim_y, src1_lddr.stride_y,
                       MMALIB_sizeof(src1_lddr.data_type), testPatternString);

         /**********************************************************************************************************
          *
          * optimized C
          *
          *********************************************************************************************************/
         // initialize the kernel
         TI_profile_start(TI_PROFILE_KERNEL_INIT);
         MMALIB_asm(" MARK 0");
         kerInitArgs.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
         //               MMALIB_DEBUGPRINT_STACK_PTR;
         currTestStatus = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_init(handle,
                                                                           &src0_addr,
                                                                           &src1_addr,
                                                                           &dst_addr,
                                                                           &kerInitArgs);

         MMALIB_DEBUGPRINTFN(1, "currTestStatus = %d\n", currTestStatus);

         MMALIB_asm(" MARK 1");
         TI_profile_stop();
         if (currTestStatus == MMALIB_SUCCESS)
         {
            /* Test optimized kernel */
            MMALIB_DEBUGPRINTFN(1, "Starting optimized-C computation.  currTestStatus=%d.\n", currTestStatus);
            kerExecInArgs.blockFeatureWidth = src1_addr.dim_x;
            TI_profile_start(TI_PROFILE_KERNEL_OPT);
            MMALIB_asm(" MARK 2");
            //               MMALIB_DEBUGPRINT_STACK_PTR;
            currTestStatus = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_exec(handle, src0, pSrc1, shiftValues, biasBValues, dst,
                                                                              &kerExecInArgs,
                                                                              &kerExecOutArgs_ci);
            MMALIB_asm(" MARK 3");
            TI_profile_stop();

            if (currTestStatus == MMALIB_SUCCESS)
            {
               // call init again to get first-call cycles
               currTestStatus = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_init(handle,
                                                                                 &src0_addr,
                                                                                 &src1_addr,
                                                                                 &dst_addr,
                                                                                 &kerInitArgs);

               // run warm instruction cache test
               TI_profile_start(TI_PROFILE_KERNEL_OPT_WARM);
               MMALIB_asm(" MARK 4");
               currTestStatus = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_exec(handle,
                                                                                 src0,
                                                                                 pSrc1,
                                                                                 shiftValues,
                                                                                 biasBValues,
                                                                                 dst,
                                                                                 &kerExecInArgs,
                                                                                 &kerExecOutArgs_ci);
               MMALIB_asm(" MARK 5");
               TI_profile_stop();

               // get output to L1D
               uint32_t row, col;
               int16_t outSum = 0;
               // int16_t volatile volatileSum = 0;  // use volatile to keep compiler from removing this operation
               int8_t *pDst = (int8_t *)dst; // treat output as bytes to be data type agnostic
               for (row = 0; row < dst_addr.dim_y; row++)
               {
                  for (col = 0; col < dst_addr.stride_y; col++)
                  {
                     outSum += *pDst;
                     pDst++;
                  }
               }

               // dummy store of outSum to insure that the compiler does not remove it.
               volatileSum = outSum;
               // call init again to get first-call cycles
               if (currTestStatus == MMALIB_SUCCESS)
               {
                  currTestStatus = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_init(handle,
                                                                                    &src0_addr,
                                                                                    &src1_addr,
                                                                                    &dst_addr,
                                                                                    &kerInitArgs);

                  // run warm instruction cache test
                  TI_profile_start(TI_PROFILE_KERNEL_OPT_WARMWRB);
                  MMALIB_asm(" MARK 6");
                  currTestStatus = MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_exec(handle,
                                                                                    src0,
                                                                                    pSrc1,
                                                                                    shiftValues,
                                                                                    biasBValues,
                                                                                    dst,
                                                                                    &kerExecInArgs,
                                                                                    &kerExecOutArgs_ci);
                  MMALIB_asm(" MARK 7");
                  TI_profile_stop();
               }
            }
         }
         else
         {
            MMALIB_PRINTF("MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_init_checkParams failed!%s", "\n");
         }
      }
      free(handle);

   } // end idat test cases

   return fail;
}

int test_main(uint32_t *pProfile)
{
   if (TI_cache_init())
   {
      TI_memError("MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX");
      return 1;
   }
   else
   {
      return MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_d(pProfile, 0);
   }
}

/* Main call for individual test projects */

int mainConvSmall()
{
   int fail = 1;

   uint32_t profile[MMALIB_MAX_TEST_PER_KERNEL * 3];

   MMALIB_TEST_init();

   fail = test_main(&profile[0]);

#if !defined(NO_PRINTF)
   if (fail == 0)
   {
      printf("Test Pass!%s", "\n");
   }
   else
   {
      printf("Test Fail!%s", "\n");
   }

   int i;
   for (i = 0; i < test_cases; i++)
   {
      printf("Test %4d: Cold Cycles = %8d, Warm Cycles = %8d, Warm Cycles WRB = %8d\n", i, profile[3 * i], profile[3 * i + 1], profile[3 * i + 2]);
   }
#endif // NO_PRINTF

   return fail;
}

/* ======================================================================== */
/*  End of file:  MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_d.c           */
/* ======================================================================== */
