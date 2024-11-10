#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include <stdint.h>
#include "convTest.h"
// #include "deconvTest.h"
// #include "linalgTest.h"

extern "C"
{
#include "tester/test/MMALIB_test.h"
#include "tester/cnn/convSmallTest/MMALIB_CNN_convolve_col_smallNo_ixX_ixX_oxX_d.h"
#include "tester/cnn/convTest/MMALIB_CNN_convolve_row_ixX_ixX_oxX_d.h"
}

/**
 * hello.cpp
 */

void updatePC()
{
    //  プログラムをまとめて64bit空間に退避する。
}

void resetHwaError()
{
    __HWAOPEN(configRegisterStruct_i8u_i8s_o8u, offsetRegStruct_diagonal_8bit, __MMA_OPEN_FSM_RESET);
    __HWACLOSE(0);
}

int main(void)
{
    printf("Hello World!\n");

    char *hoge = (char *)0xA0300000;
    *hoge = 1;
    *(hoge + 1) = 2;

    resetHwaError();

    mainConvSmall();
    // mainConv();

    // MMALIB_TEST_init();

    // mmaLinalg();
    // mmaConvFullSpeed();
    // mmaDeconv();

    // seTestTranspose();

    return 0;
}
