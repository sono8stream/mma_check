#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include "tester.h"
#include <stdint.h>

void mmaConv(){
    printf("%d",staticRefKernel_case13[0]);
    printf("%d",staticRefIn_case13[0]);
    printf("%d",staticRefOutput_case13[0]);

    int numInChannels = 1;
    int numOutChannels = 1;
    int numGroupsPerKernel = 1;

    MMALIB_bufParams2D_t kernelBuffer;
    int kDim = 3 * 3 * numInChannels;
    kernelBuffer.dim_x = kDim;
    kernelBuffer.dim_y = numOutChannels * numGroupsPerKernel;// numOfOutputChKerBuf * numGroupsPerKernel
    kernelBuffer.stride_y = 64;// pitchA
    kernelBuffer.data_type = MMALIB_INT8;

    int inSize = 66048;

    MMALIB_bufParams2D_t srcBuffer;
    srcBuffer.dim_x = inSize;// inChOffset
    srcBuffer.dim_y = numInChannels * numGroupsPerKernel;// numInChannels*numGroupsPerKernel
    srcBuffer.stride_y = inSize;// inChOffset
    srcBuffer.data_type = MMALIB_UINT8;

    int kernelWidth = 3;
    int kernelHeight = 3;
    int numBiasVals = kDim - kernelWidth*kernelHeight*numInChannels;// 今回はゼロ。
    int numBytes = 1;

    MMALIB_bufParams2D_t biasBuffer;
    biasBuffer.dim_x = numBiasVals;// numBiasVals
    biasBuffer.dim_y = numOutChannels;// numOutChannels
    biasBuffer.stride_y = numBiasVals * numBytes;// 今回はゼロ。
    biasBuffer.data_type = MMALIB_INT8;// カーネルと同じデータ型。

    int outSize = 63990;//63488;// validColsOut(256) * 2 + align

    MMALIB_bufParams3D_t dstBuffer;
    dstBuffer.dim_x = outSize;// pitchC/numBytes
    dstBuffer.dim_y = numOutChannels;// numOutChannels
    dstBuffer.stride_y = outSize;
    dstBuffer.dim_z = numGroupsPerKernel;// numGroupsPerKernel
    dstBuffer.stride_z = numOutChannels * outSize;// numOutChannels*pitchC
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
    kerExecInArgs.validColsIn       = inSize;
    kerExecInArgs.validColsPerRowIn = 0;//inWidth;
    kerExecInArgs.validRowsIn       = 0;//64;
    kerExecInArgs.pad               = pad;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs;

    int8_t* kernel = (int8_t*)malloc(9);
    for(int i=0;i<9;i++){
        kernel[i] = 1;
    }

    int8_t* src = (int8_t*)malloc(inSize);// paddingを考慮に入れてサイズを確保する
    for(int i=0;i<inSize;i++){
        src[i]=i%10;
    }

    int8_t* dst = (int8_t*)malloc(outSize);
    for(int i=0;i<outSize;i++){
        dst[i]=0;
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

void mmaConvFullSpeed(){
    printf("%d",staticRefKernel_case13[0]);
    printf("%d",staticRefIn_case13[0]);
    printf("%d",staticRefOutput_case13[0]);

    int numInChannels = 1;
    int numOutChannels = 1;
    int numGroupsPerKernel = 1;

    MMALIB_bufParams2D_t kernelBuffer;
    int kernelLength=5;
    int kDim = kernelLength * kernelLength * numInChannels;
    kernelBuffer.dim_x = kDim;
    kernelBuffer.dim_y = numOutChannels * numGroupsPerKernel;// numOfOutputChKerBuf * numGroupsPerKernel
    kernelBuffer.stride_y = 64;// pitchA
    kernelBuffer.data_type = MMALIB_INT8;

    int w=256;
    int h=10;
    int pad=2;
    int shellSize=kernelLength/2;

    //雑に256*10行に3x3Convolutionやってみる。
    int inSize = w*h+w*pad*2+h*pad+pad*pad*2+shellSize;

    MMALIB_bufParams2D_t srcBuffer;
    srcBuffer.dim_x = inSize;// inChOffset
    srcBuffer.dim_y = numInChannels * numGroupsPerKernel;// numInChannels*numGroupsPerKernel
    srcBuffer.stride_y = inSize;// inChOffset
    srcBuffer.data_type = MMALIB_UINT8;

    int numBiasVals = kDim - kernelLength*kernelLength*numInChannels;// 今回はゼロのはず。
    int numBytes = 1;

    MMALIB_bufParams2D_t biasBuffer;
    biasBuffer.dim_x = numBiasVals;// numBiasVals
    biasBuffer.dim_y = numOutChannels;// numOutChannels
    biasBuffer.stride_y = numBiasVals * numBytes;// 今回はゼロ。
    biasBuffer.data_type = MMALIB_INT8;// カーネルと同じデータ型。

    int outSize = (w+pad)*(h+pad*2-shellSize*2);

    MMALIB_bufParams3D_t dstBuffer;
    dstBuffer.dim_x = outSize;// pitchC/numBytes
    dstBuffer.dim_y = numOutChannels;// numOutChannels
    dstBuffer.stride_y = outSize;
    dstBuffer.dim_z = numGroupsPerKernel;// numGroupsPerKernel
    dstBuffer.stride_z = numOutChannels * outSize;// numOutChannels*pitchC
    dstBuffer.data_type = MMALIB_UINT8;

    int dilationWidth = 1;
    int dilationHeight = 1;
    int strideWidth = 1;
    int strideHeight = 1;
    int validColsIn = inSize;
    int subMChannels = 1;
    int inWidth = w;
    int maxHeight = 10;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_InitArgs initArgs;
    initArgs.funcStyle = MMALIB_FUNCTION_NATC;//MMALIB_FUNCTION_OPTIMIZED;
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
    initArgs.Fr = kernelLength;
    initArgs.Fc = kernelLength;
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
    kerExecInArgs.validColsIn       = inSize;
    kerExecInArgs.validColsPerRowIn = 0;//inWidth;
    kerExecInArgs.validRowsIn       = 0;//64;
    kerExecInArgs.pad               = pad;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs;

    // データをL2に置いてみる。
    int8_t* kernel = (int8_t*)0x64810000;
    for(int i=0;i<kernelLength*kernelLength;i++){
        kernel[i] = 1;
    }

    int8_t* src = (int8_t*)0x64820000;// paddingを考慮に入れてサイズを確保する
    for(int i=0;i<inSize;i++){
        src[i]=i%10;
    }

    int8_t* dst = (int8_t*)0x64840000;
    for(int i=0;i<outSize;i++){
        dst[i]=0;
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
