#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include <stdint.h>

void mmaLinalg(){
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
