#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include "tester.h"
//#include <c7x_scalable.h>
//#include <stdlib.h>
//#include <iostream>

/**
 * hello.cpp
 */

const int N = 1000;
const int Width = 40;

//#define IMG_WIDTH  (16)
//#define IMG_HEIGHT (32)
//#define IMG_STRIDE (IMG_WIDTH)
//#define NUM_CH     (3)

void seTestTranspose(){

    /*
    int img_buffer[NUM_CH * IMG_STRIDE * IMG_HEIGHT];

    //Initialize image buffer
    for(int ch = 0; ch < NUM_CH; ch++) {

        //Update channel pointer
        int *pIn = (int *)&img_buffer[(ch * (IMG_STRIDE * IMG_HEIGHT))];

        //Reset value for every channel
        int val = 0;

        for(int row = 0; row < IMG_HEIGHT; row++) {
            for(int col = 0; col < IMG_WIDTH; col++) {
                pIn[(row * IMG_STRIDE) + col] = val++;
            }
        }
    }

    for(int ch = 0; ch < NUM_CH; ch++) {

        //Update channel pointer
        int *pIn = (int *)&img_buffer[(ch * (IMG_STRIDE * IMG_HEIGHT))];
        printf("Ch = %d \n", ch);
        for(int row = 0; row < IMG_HEIGHT; row++) {
            for(int col = 0; col < IMG_WIDTH; col++) {
                printf("%3d, ", pIn[(row * IMG_STRIDE) + col]);
            }
            printf("\n");
        }
        printf("\n");
    }
    */
}

void mmaTest(){
    const size_t rows = 3;
    const size_t cols = 3;
    const size_t size = rows * cols;

    int32_t matA[size];
    int32_t matB[size];
    int32_t matC[size];
    int32_t matOpt[size];

    size_t i;
    for (i = 0; i < size; ++i) {
        matA[i] = i;
        matB[i] = i;
    }

    MMALIB_bufParams2D_t aBuffer;
    aBuffer.data_type = MMALIB_INT32;
    aBuffer.dim_x = cols;
    aBuffer.dim_y = rows;
    aBuffer.stride_y = aBuffer.dim_x * MMALIB_sizeof(aBuffer.data_type);

    MMALIB_bufParams2D_t bBuffer;
    bBuffer.data_type = MMALIB_INT32;
    bBuffer.dim_x = cols;
    bBuffer.dim_y = rows;
    bBuffer.stride_y = bBuffer.dim_x * MMALIB_sizeof(bBuffer.data_type);

    MMALIB_bufParams2D_t resultBuffer;
    resultBuffer.data_type = MMALIB_INT32;
    resultBuffer.dim_x = cols;
    resultBuffer.dim_y = rows;
    resultBuffer.stride_y = resultBuffer.dim_x * MMALIB_sizeof(resultBuffer.data_type);

    MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_InitArgs initArgs;
    initArgs.funcStyle = MMALIB_FUNCTION_NATC;
    initArgs.bTranspose = MMALIB_LINALG_MATRIXMATRIXMULTIPLY_IXX_IXX_OXX_B_NON_TRANSPOSED;
    initArgs.shift = 0;

    int32_t handleSize = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_getHandleSize(&initArgs);
    MMALIB_kernelHandle kernelHandle = malloc(handleSize);

    // Check that the parameters will generate a valid handle
    MMALIB_STATUS initCheck = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_init_checkParams(kernelHandle,
        &aBuffer,
        &bBuffer,
        &resultBuffer,
        &initArgs);

    printf("Init check = %d.\n", initCheck);

    // Generate the handle
    MMALIB_STATUS initStatus = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_init(kernelHandle, &aBuffer, &bBuffer, &resultBuffer, &initArgs);
    printf("Init status = %d.\n", initStatus);

    // Check that the execute arguments are valid for execution
    MMALIB_STATUS execCheck = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_exec_checkParams(kernelHandle,
        &matA[0],
        &matB[0],
        &matC[0]);
    printf("Exec check = %d.\n", execCheck);

    // Execute the kernel
    MMALIB_STATUS execStatus = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_exec(kernelHandle,
        &matA[0],
        &matB[0],
        &matC[0]);

    printf("Exec status = %d.\n", execStatus);

    printf("The natural C result is: \n");
    size_t k;
    size_t idx = 0;
    for (i = 0; i < rows; i++) {
        for (k = 0; k < cols; k++) {
            printf("%4d ", matC[idx++]);
        }
        printf("\n");
    }

    initArgs.funcStyle = MMALIB_FUNCTION_OPTIMIZED;
    // Execute the kernel
    execStatus = MMALIB_LINALG_matrixMatrixMultiply_ixX_ixX_oxX_exec(kernelHandle,
        &matA[0],
        &matB[0],
        &matOpt[0]);
    printf("Exec status = %d.\n", execStatus);

    printf("The optimized C result is: \n");
    idx = 0;
    for (i = 0; i < rows; i++) {
        for (k = 0; k < cols; k++) {
            printf("%4d ", matOpt[idx++]);
        }
        printf("\n");
    }

    printf("MatMulIntrinsics done...\n");
    free(kernelHandle);
}

void mmaTest2(){
    printf("%d",staticRefKernel_case13[0]);
    printf("%d",staticRefIn_case13[0]);
    printf("%d",staticRefOutput_case13[0]);

    int numInChannels = 256;
    int numGroupsPerKernel = 1;

    MMALIB_bufParams2D_t kernelBuffer;
    int kDim = 2304;
    kernelBuffer.dim_x = kDim;
    kernelBuffer.dim_y = 64 * numGroupsPerKernel;// numOfOutputChKerBuf * numGroupsPerKernel
    kernelBuffer.stride_y = 2624;// pitchA
    kernelBuffer.data_type = MMALIB_INT8;

    MMALIB_bufParams2D_t srcBuffer;
    srcBuffer.dim_x = 1088;// inChOffset
    srcBuffer.dim_y = numInChannels * numGroupsPerKernel;// numInChannels*numGroupsPerKernel
    srcBuffer.stride_y = 1088;// inChOffset
    srcBuffer.data_type = MMALIB_INT8;

    int kernelWidth = 3;
    int kernelHeight = 3;
    int numBiasVals = kDim - kernelWidth*kernelHeight*numInChannels;// 今回はゼロ。
    int numBytes = 1;
    int numOutChannels = 64;

    MMALIB_bufParams2D_t biasBuffer;
    biasBuffer.dim_x = numBiasVals;// numBiasVals
    biasBuffer.dim_y = numOutChannels;// numOutChannels
    biasBuffer.stride_y = numBiasVals * numBytes;// 今回はゼロ。
    biasBuffer.data_type = MMALIB_INT8;// カーネルと同じデータ型。

    int pitchC = 576;

    MMALIB_bufParams3D_t dstBuffer;
    dstBuffer.dim_x = pitchC / 1;// pitchC/numBytes
    dstBuffer.dim_y = numOutChannels;// numOutChannels
    dstBuffer.stride_y = pitchC;
    dstBuffer.dim_z = numGroupsPerKernel;// numGroupsPerKernel
    dstBuffer.stride_z = numOutChannels * pitchC;// numOutChannels*pitchC
    dstBuffer.data_type = MMALIB_UINT8;

    int dilationWidth = 1;
    int dilationHeight = 1;
    int strideWidth = 1;
    int strideHeight = 1;
    int validColsIn = 772;
    int subMChannels = 64;
    int inWidth = 256;
    int pad = 1;
    int maxHeight = 256;
    int inChOffset = 1088;

    MMALIB_CNN_convolve_row_ixX_ixX_oxX_InitArgs initArgs;
    initArgs.funcStyle = MMALIB_FUNCTION_NATC;
    initArgs.No = numOutChannels;
    initArgs.inChOffset = inChOffset;
    initArgs.validColsIn = validColsIn;
    initArgs.validColsPerRowIn = 0;
    initArgs.validRowsIn = 0;
    initArgs.inputPitchPerRow = 0;
    initArgs.outputPitchPerRow = 0;
    initArgs.inWidth = inWidth;
    initArgs.pad = 1;
    initArgs.maxHeight = maxHeight;
    initArgs.subMChannels = subMChannels;
    initArgs.numGroupsPerKernel = numGroupsPerKernel;
    initArgs.shift = 14;
    initArgs.Fr = kernelWidth;
    initArgs.Fc = kernelHeight;
    initArgs.strideX = strideWidth;
    initArgs.strideY = strideHeight;
    initArgs.dilationX = dilationWidth;
    initArgs.dilationY = dilationHeight;
    initArgs.bias = 20;
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

    printf("MatMulIntrinsics done...\n");
    free(kernelHandle);
}

int main(void)
{
    printf("Hello World!\n");

    //mmaTest();
    mmaTest2();

    //seTestTranspose();

    return 0;
}
