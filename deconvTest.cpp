#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include "tester2.h"
#include <stdint.h>

typedef struct {
   uint8_t testPattern; /* 0: constant, 1: sequential, 2: random, 3: static
                           array, 4: file, etc */
   void *  staticIn8Bit;
   void *  staticKernel8Bit;
   void *  staticOutMMALIB8Bit;
   void *  staticOut8Bit;
   void *  staticIn16Bit;
   void *  staticKernel16Bit;
   void *  staticOutMMALIB16Bit;
   void *  staticOut16Bit;
   int32_t kernelWidth;
   int32_t kernelHeight;
   int32_t strideWidth;
   int32_t strideHeight;
   int32_t dilationHeight;
   int32_t dilationWidth;
   int32_t inWidth;
   int32_t pad;
   int32_t maxHeight;
   int32_t inChOffset;
   int32_t validColsIn;
   int32_t kDim;
   int32_t pitchA;
   int32_t numOfOutputChKerBuf;
   int32_t pitchC;
   int32_t subMChannels;
   int32_t numInChannels;
   int32_t numOutChannels;
   int32_t numGroups;
   uint8_t qShift;
   int32_t biasB;
   int8_t  activationType;
   uint8_t dataTypeA;
   uint8_t dataTypeB;
   uint8_t dataTypeC;
   uint8_t mode;
   int32_t circularOffset;
   int32_t totalN;
   int32_t subN;
   int32_t validColsPerRowIn;
   int32_t validRowsIn;
   int32_t outputPitchPerRow;
   int32_t inputPitchPerRow;
   int32_t outputDataLocation; // 0 -> L2SRAM, 1-> MSMC
   bool    preProcessFlag;
   int32_t testID;
} deconvolve_row_ixX_ixX_oxX_testParams_t;

void mmaDeconv(){

    deconvolve_row_ixX_ixX_oxX_testParams_t param={
       3, // testPattern
       staticRefInputXCaseCase3_i8s_i8s_o8s,
       staticRefInputHCaseCase3_i8s_i8s_o8s,
       staticRefOutputYMMALIBCaseCase3_i8s_i8s_o8s,
       staticRefOutputYCaseCase3_i8s_i8s_o8s,
       NULL,
       NULL,
       NULL,
       NULL,
       4,                 // kernelWidth
       4,                 // kernelHeight
       2,                 // strideWidth
       2,                 // strideHeight
       1,                 // dilationHeight
       1,                 // dilationWidth
       68,                // inWidth
       1,                 // pad
       8,                 // maxHeight
       448,               // inChoffset
       414,               // validColsIn
       16,                 // kDim
       320,               // pitchA
       1,                 // numOfOutputChKerBuf
       1768,              // pitchC
       1,                 // subMChannels
       1,                 // numInChannels
       1,                 // numOutChannels
       1,                 // numGroups
       0,                 // qShift
       0,                // biasB
       MMALIB_SATURATION, // activationType
       MMALIB_INT8,       // dataTypeA
       MMALIB_INT8,       // dataTypeB
       MMALIB_INT8,       // dataTypeC
       0,                 // mode
       0,                 // circularOffset
       1,                 // totalN
       1,                 // subN
       0,                 // validColsPerRowIn
       0,                 // ValidRowsIn
       0,                 // outputPitchPerRow
       0,                 // inputPitchPerRow
       MMALIB_TEST_OUTPUT_MSMC,
       0, // preProcessFlag
       3,
    };

    int numOfOutputChKerBuf = param.numOfOutputChKerBuf * 4;// 出力サイズは4倍のサイズになるはず。

    MMALIB_bufParams2D_t kernelBuffer;
    kernelBuffer.dim_x = param.kDim;
    kernelBuffer.dim_y =  numOfOutputChKerBuf * param.numGroups;
    kernelBuffer.stride_y = param.pitchA;
    kernelBuffer.data_type = param.dataTypeA;

    MMALIB_bufParams2D_t srcBuffer;
    srcBuffer.dim_x = param.inChOffset;// inChOffset
    srcBuffer.dim_y = param.numInChannels * param.numGroups;// numInChannels*numGroupsPerKernel
    srcBuffer.stride_y = param.inChOffset;// inChOffset
    srcBuffer.data_type = param.dataTypeB;

    MMALIB_bufParams2D_t dstBuffer;
    dstBuffer.dim_x = param.pitchC;
    dstBuffer.dim_y = param.numOutChannels * param.numGroups;// numOutChannels
    dstBuffer.stride_y = param.pitchC;
    dstBuffer.data_type = param.dataTypeC;

    MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_InitArgs kerInitArgs;
    kerInitArgs.Fc                = param.kernelWidth;
    kerInitArgs.Fr                = param.kernelHeight;
    kerInitArgs.dilationX         = param.dilationWidth;
    kerInitArgs.dilationY         = param.dilationHeight;
    kerInitArgs.strideX           = param.strideWidth;
    kerInitArgs.strideY           = param.strideHeight;
    kerInitArgs.validColsIn       = param.validColsIn;
    kerInitArgs.subMChannels      = param.subMChannels;
    kerInitArgs.inWidth           = param.inWidth;
    kerInitArgs.pad               = param.pad;
    kerInitArgs.maxHeight         = param.maxHeight;
    kerInitArgs.inChOffset        = param.inChOffset;
    kerInitArgs.shift             = param.qShift;
    kerInitArgs.No                = param.numOutChannels;
    kerInitArgs.numGroups         = param.numGroups;
    kerInitArgs.bias              = param.biasB;
    kerInitArgs.activationType    = param.activationType;
    kerInitArgs.mode              = param.mode;
    kerInitArgs.validColsPerRowIn = param.validColsPerRowIn;
    kerInitArgs.validRowsIn       = param.validRowsIn;
    kerInitArgs.outputPitchPerRow = param.outputPitchPerRow;
    kerInitArgs.inputPitchPerRow  = param.inputPitchPerRow;
    kerInitArgs.funcStyle         = MMALIB_FUNCTION_NATC;

    int32_t handleSize = MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_getHandleSize(&kerInitArgs);
    MMALIB_kernelHandle kernelHandle = malloc(handleSize);

    // Check that the parameters will generate a valid handle
    MMALIB_STATUS initCheck = MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_init_checkParams(kernelHandle,
        &kernelBuffer,
        &srcBuffer,
        &dstBuffer,
        &kerInitArgs);

    printf("Init check = %d.\n", initCheck);

    // Generate the handle
    MMALIB_STATUS initStatus = MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_init(kernelHandle,
                                                                        &kernelBuffer,
                                                                        &srcBuffer,
                                                                        &dstBuffer,
                                                                        &kerInitArgs);
    printf("Init status = %d.\n", initStatus);

    // 計算用のカーネルを得る
    const int kernelBaseSize = 64;
    int8_t* kernelBase = (int8_t*)malloc(kernelBaseSize);// paddingを考慮に入れてサイズを確保する
    for(int i=0;i<kernelBaseSize;i++){
        if(i/16==0){
            kernelBase[i] = 1;
        }
        else{
            kernelBase[i] = 0;
        }
    }

    int kernelSize = (numOfOutputChKerBuf * param.pitchA) * param.numGroups;
    int8_t* kernel = (int8_t*)malloc(kernelSize);
    MMALIB_CNN_deconvolve_row_4x4Stride2PreProcessParameters (
            param.kDim, param.numInChannels, param.pitchA,
            param.numOutChannels, // before pre-processing
            param.numGroups, MMALIB_MMA_SIZE_8_BIT, kernelBase,
            kernel);

    MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_ExecInArgs kerExecInArgs;
    kerExecInArgs.subMChannels = param.subMChannels;
    kerExecInArgs.validColsIn       = param.validColsIn;

    MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_ExecOutArgs kerExecOutArgs;

    const int inSize = 448;
    int8_t* src = (int8_t*)malloc(inSize);// paddingを考慮に入れてサイズを確保する
    for(int i=0;i<inSize;i++){
        if(i==0){
            src[i]=0;
        }else{
            src[i]=2;
        }
    }

    int8_t* dst = (int8_t*)malloc(1376);
    for(int i=0;i<1376;i++){
        dst[i]=0;
    }

    MMALIB_STATUS execCheck = MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_exec_checkParams(kernelHandle,
                                                                                   kernel,
                                                                                   src,
                                                                                   dst,
                                                                            &kerExecInArgs);
    printf("Exec check = %d.\n", execCheck);

    MMALIB_STATUS execStatus = MMALIB_CNN_deconvolve_row_ixX_ixX_oxX_exec(kernelHandle,
                                                                          kernel,
                                                                          src,
                                                                        dst,
                                                                        &kerExecInArgs,
                                                                        &kerExecOutArgs);
    printf("Exec status = %d.\n", execStatus);

    printf("MatMulIntrinsics done...\n");
    free(kernelHandle);
}
