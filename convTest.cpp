#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include "tester.h"
#include <stdint.h>

void mmaConv()
{
    printf("%d", staticRefKernel_case13[0]);
    printf("%d", staticRefIn_case13[0]);
    printf("%d", staticRefOutput_case13[0]);

    int numInChannels = 1;
    int numOutChannels = 1;
    int numGroupsPerKernel = 1;

    MMALIB_bufParams2D_t kernelBuffer;
    int kDim = 3 * 3 * numInChannels;
    kernelBuffer.dim_x = kDim;
    kernelBuffer.dim_y = numOutChannels * numGroupsPerKernel; // numOfOutputChKerBuf * numGroupsPerKernel
    kernelBuffer.stride_y = 64;                               // pitchA
    kernelBuffer.data_type = MMALIB_INT8;

    int inSize = 66048;

    MMALIB_bufParams2D_t srcBuffer;
    srcBuffer.dim_x = inSize;                             // inChOffset
    srcBuffer.dim_y = numInChannels * numGroupsPerKernel; // numInChannels*numGroupsPerKernel
    srcBuffer.stride_y = inSize;                          // inChOffset
    srcBuffer.data_type = MMALIB_UINT8;

    int kernelWidth = 3;
    int kernelHeight = 3;
    int numBiasVals = kDim - kernelWidth * kernelHeight * numInChannels; // 今回はゼロ。
    int numBytes = 1;

    MMALIB_bufParams2D_t biasBuffer;
    biasBuffer.dim_x = numBiasVals;               // numBiasVals
    biasBuffer.dim_y = numOutChannels;            // numOutChannels
    biasBuffer.stride_y = numBiasVals * numBytes; // 今回はゼロ。
    biasBuffer.data_type = MMALIB_INT8;           // カーネルと同じデータ型。

    int outSize = 63990; // 63488;// validColsOut(256) * 2 + align

    MMALIB_bufParams3D_t dstBuffer;
    dstBuffer.dim_x = outSize;        // pitchC/numBytes
    dstBuffer.dim_y = numOutChannels; // numOutChannels
    dstBuffer.stride_y = outSize;
    dstBuffer.dim_z = numGroupsPerKernel;          // numGroupsPerKernel
    dstBuffer.stride_z = numOutChannels * outSize; // numOutChannels*pitchC
    dstBuffer.data_type = MMALIB_UINT8;

    int dilationWidth = 1;
    int dilationHeight = 1;
    int strideWidth = 1;
    int strideHeight = 1;
    int validColsIn = inSize;
    int subMChannels = 1;
    int inWidth = 1030;
    int pad = 1;
    int maxHeight = 256;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_InitArgs initArgs;
    initArgs.funcStyle = MMALIB_FUNCTION_NATC;
    initArgs.No = numOutChannels;
    initArgs.inChOffset = inSize;
    initArgs.validColsIn = validColsIn;
    initArgs.validColsPerRowIn = 0;
    initArgs.validRowsIn = 0;
    initArgs.inputPitchPerRow = 0;
    initArgs.outputPitchPerRow = 0;
    initArgs.inWidth = inWidth;
    initArgs.pad = pad;
    initArgs.maxHeight = maxHeight;
    initArgs.subMChannels = subMChannels;
    initArgs.numGroupsPerKernel = numGroupsPerKernel;
    initArgs.shift = 0;
    initArgs.Fr = kernelWidth;
    initArgs.Fc = kernelHeight;
    initArgs.strideX = strideWidth;
    initArgs.strideY = strideHeight;
    initArgs.dilationX = dilationWidth;
    initArgs.dilationY = dilationHeight;
    initArgs.bias = 0;
    initArgs.activationType = MMALIB_RELU;
    initArgs.mode = MMALIB_LINEAR;
    initArgs.weightReorderFlag = 0;
    initArgs.numBiasVals = numBiasVals;

    int32_t handleSize = MMALIB_CNN_convolve_row_ixX_ixX_oxX_getHandleSize(&initArgs);
    MMALIB_kernelHandle kernelHandle = malloc(handleSize);

    // Check that the parameters will generate a valid handle
    MMALIB_STATUS initCheck = MMALIB_CNN_convolve_row_ixX_ixX_oxX_init_checkParams(kernelHandle,
                                                                                   &kernelBuffer,
                                                                                   &srcBuffer,
                                                                                   &dstBuffer,
                                                                                   &initArgs);

    printf("Init check = %d.\n", initCheck);

    // Generate the handle
    MMALIB_STATUS initStatus = MMALIB_CNN_convolve_row_ixX_ixX_oxX_init(kernelHandle,
                                                                        &kernelBuffer,
                                                                        &srcBuffer,
                                                                        &dstBuffer,
                                                                        &initArgs);
    printf("Init status = %d.\n", initStatus);

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecInArgs kerExecInArgs;
    kerExecInArgs.subMChannels = subMChannels;
    kerExecInArgs.validColsIn = inSize;
    kerExecInArgs.validColsPerRowIn = 0; // inWidth;
    kerExecInArgs.validRowsIn = 0;       // 64;
    kerExecInArgs.pad = pad;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs;

    int8_t *kernel = (int8_t *)malloc(9);
    for (int i = 0; i < 9; i++)
    {
        kernel[i] = 1;
    }

    int8_t *src = (int8_t *)malloc(inSize); // paddingを考慮に入れてサイズを確保する
    for (int i = 0; i < inSize; i++)
    {
        src[i] = i % 10;
    }

    int8_t *dst = (int8_t *)malloc(outSize);
    for (int i = 0; i < outSize; i++)
    {
        dst[i] = 0;
    }

    MMALIB_STATUS execCheck = MMALIB_CNN_convolve_row_ixX_ixX_oxX_exec_checkParams(kernelHandle,
                                                                                   kernel,
                                                                                   src,
                                                                                   dst,
                                                                                   &kerExecInArgs);
    printf("Exec check = %d.\n", execCheck);

    MMALIB_STATUS execStatus = MMALIB_CNN_convolve_row_ixX_ixX_oxX_exec(kernelHandle,
                                                                        kernel,
                                                                        src,
                                                                        dst,
                                                                        &kerExecInArgs,
                                                                        &kerExecOutArgs);
    printf("Exec status = %d.\n", execStatus);

    printf("MatMulIntrinsics done...\n");
    free(kernelHandle);
    free(kernel);
    free(src);
    free(dst);
}

void mmaConvFullSpeed()
{
    // initialize status flags
    MMALIB_STATUS currTestStatus = MMALIB_STATUS_NOT_SET; /* Test current testcase status */

    long long clocks[10];

    int kernelSize = 3;
    int inWidth = 256;
    int height = 1;
    int pad = kernelSize / 2;
    int validColsIn = (inWidth + pad) * (height + pad * 2) + pad;
    int inChOffset = (validColsIn + 63) / 64 * 64;
    int maxHeight = height + pad * 2 + 1;

    int32_t inputBlockWidth = inWidth + pad;
    int32_t numInChannels = 1;       // prm[tpi].numInChannels;
    int32_t numOutChannels = 1;      // prm[tpi].numOutChannels;
    int32_t numOfOutputChKerBuf = 1; // prm[tpi].numOfOutputChKerBuf;
    int32_t subMChannels = 1;        // prm[tpi].subMChannels;
    int32_t kernelWidth = kernelSize;
    int32_t kernelHeight = kernelSize;
    int32_t dilationWidth = 1;
    int32_t dilationHeight = 1;
    int32_t strideWidth = 1;
    int32_t strideHeight = 1;
    int32_t kDim = 9;
    int32_t pitchA = 64;
    int32_t pitchC = 320;
    uint8_t dataTypeA = MMALIB_INT8;
    uint8_t dataTypeB = MMALIB_UINT8;
    uint8_t dataTypeC = MMALIB_UINT8;
    uint8_t activationType = MMALIB_RELU;
    int32_t shift = 0;
    int32_t bias = 10;
    uint8_t mode = MMALIB_LINEAR;
    int32_t validColsPerRowIn = 0;
    int32_t validRowsIn = 0;
    int32_t outputPitchPerRow = 0;
    int32_t inputPitchPerRow = 0;
    int32_t validColsInlast = 0;
    int32_t validRowsInlast = 0;
    int32_t validColsPerRowInlast = 0;
    int32_t numGroupsPerKernel = 1;
    int32_t MCounter = numOutChannels / subMChannels;
    MCounter = (numOutChannels % subMChannels == 0) ? MCounter - 1 : MCounter;

    int32_t numBytes = 1;
    int32_t MMA_SIZE = MMALIB_MMA_SIZE_8_BIT;

    MMALIB_kernelHandle handle;
    int32_t handleSize;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_InitArgs kerInitArgs;
    MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecInArgs kerExecInArgs;
    MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs;

    // Function returns the memory for the handle to be allocated
    handleSize = MMALIB_CNN_convolve_row_ixX_ixX_oxX_getHandleSize(&kerInitArgs);
    MMALIB_DEBUGPRINTFN(1, "\nAfter MMALIB_CNN Handle Size before %d\n", handleSize);

    // Predicate buffer size is returned for allocating appropriate memory by application
    int32_t defaultPredSize = MMALIB_CNN_seamPredicateRegistersSizeDefault();
    int32_t actualPredSize = MMALIB_CNN_seamPredicateRegistersSize(inWidth, pad, maxHeight, MMA_SIZE, numOutChannels, subMChannels);

    if (actualPredSize > defaultPredSize)
    {
        handleSize = handleSize + actualPredSize - defaultPredSize;
    }

    MMALIB_DEBUGPRINTFN(1, "\nAfter MMALIB_CNN Handle Size after %d\n", handleSize);
    handle = malloc(handleSize);

    int32_t strideShiftW = 0;
    if (strideWidth == 2)
    {
        strideShiftW = 1;
    }
    if (strideWidth == 4)
    {
        strideShiftW = 2;
    }

    int32_t strideShiftH = 0;
    if (strideHeight == 2)
    {
        strideShiftH = 1;
    }
    if (strideHeight == 4)
    {
        strideShiftH = 2;
    }

    // validColsOut calculated for non strided convolution
    int32_t validColsOut =
        ((validColsIn - inputBlockWidth * (kernelHeight * dilationHeight - 1)) >>
         strideShiftH) >>
        strideShiftW;
    int32_t validColsPerRow = 0;

    MMALIB_bufParams2D_t src0_addr; // paramsWgt
    MMALIB_bufParams2D_t src1_addr; // input
    MMALIB_bufParams2D_t src2_addr; // bias
    MMALIB_bufParams3D_t dst_addr;
    int32_t numBiasVals = kDim - kernelWidth * kernelHeight * numInChannels;

    // カーネルバッファのアロケート
    src0_addr.dim_x = kDim;
    src0_addr.dim_y = numOfOutputChKerBuf * numGroupsPerKernel;
    src0_addr.stride_y = pitchA;
    src0_addr.data_type = dataTypeA;

    // 入力バッファのアロケート
    src1_addr.dim_x = inChOffset;
    src1_addr.dim_y = numInChannels * numGroupsPerKernel;
    src1_addr.stride_y = inChOffset;
    src1_addr.data_type = dataTypeB;

    // バイアスバッファのアロケート
    src2_addr.dim_x = numBiasVals;
    src2_addr.dim_y = numOutChannels;
    src2_addr.stride_y = numBiasVals * numBytes;
    src2_addr.data_type = dataTypeA;

    // 出力バッファのアロケート
    dst_addr.dim_x = pitchC / numBytes;
    dst_addr.dim_y = numOutChannels;
    dst_addr.stride_y = pitchC;
    dst_addr.dim_z = numGroupsPerKernel;
    dst_addr.stride_z = numOutChannels * pitchC; // numOutChannels*pitchC = pitchG
    dst_addr.data_type = dataTypeC;

    // 初期化用の処理
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

    int8_t *src0 = (int8_t *)0x64810000;

    int8_t *src1 = (int8_t *)0x64820000;

    int8_t *dst = (int8_t *)0x64840000;

    int32_t biasSize = numOutChannels * numBiasVals * numBytes;
    int8_t *src2 = (int8_t *)0x64850000;

    // for debug
    {
        int iter = 0;

        iter = 0;
        for (; iter < validColsIn; iter++)
        {
            src1[iter] = iter % 10;
        }
    }

    /* Fill input arrays according to desired test pattern */
    if (kerInitArgs.weightReorderFlag == 1)
    {
        // for debug
        int8_t *src3 = (int8_t *)0x64860000;
        int iter = 0;
        for (; iter < kDim; iter++)
        {
            src3[iter] = iter;
        }

        MMALIB_CNN_convolve_row_reorderWeights(src3, src0, src2, &src0_addr, &src2_addr, &reorderWeights, &kerInitArgs);
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
    clocks[0] = __TSC;
    currTestStatus = MMALIB_CNN_convolve_row_ixX_ixX_oxX_init(
        handle, &src0_addr, &src1_addr, &dst_addr, &kerInitArgs);

    clocks[1] = __TSC;

    /* Test optimized kernel */
    MMALIB_DEBUGPRINTFN(1, "Entering:MMALIB_CNN_convolve_row_ixX_ixX_oxX with currTestStatus = %d\n", currTestStatus);

    int32_t iterN = 0;
    int32_t validOutputRows;

    if (currTestStatus == MMALIB_SUCCESS)
    {
        int8_t *src1_Iter = src1;

        kerExecInArgs.subMChannels = subMChannels;

        kerExecInArgs.validColsIn = validColsIn;
        kerExecInArgs.validColsPerRowIn = validColsPerRowIn;
        kerExecInArgs.validRowsIn = validRowsInlast;
        kerExecInArgs.pad = pad;

        long long startTsc = __TSC;
        clocks[2] = __TSC;
        currTestStatus = MMALIB_CNN_convolve_row_ixX_ixX_oxX_exec(
            handle, src0, src1, dst, &kerExecInArgs, &kerExecOutArgs);
        clocks[3] = __TSC;
        long long endTsc = __TSC;
        long long elapsed = endTsc - startTsc;
        validColsOut = kerExecOutArgs.validColsOut;
        validColsPerRow = kerExecOutArgs.validColsPerRowOut;

        MMALIB_DEBUGPRINTFN(1, "OptC: valid cols out %d itenN %d\n", kerExecOutArgs.validColsOut, iterN);
        iterN++;

        __SE0_CLOSE();
        __SE1_CLOSE();

        validOutputRows = kerExecOutArgs.validRowsOut;
    }

    int timeDiv = 1000;
    printf("Init time: %ld us\n", (clocks[1] - clocks[0]) / timeDiv);
    printf("Exec time: %ld us\n", (clocks[3] - clocks[2]) / timeDiv);
}