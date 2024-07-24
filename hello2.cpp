#include <stdio.h>
#include <c7x.h>
#include <c7x_scalable.h>
#include <c6x_migration.h>
#include <stdlib.h>
#include <iostream>
#include <vcop/vcop.h>

/**
 * hello2.cpp
 */

namespace{
const int N = 1000;
const int Width = 40;

// SIMDサンプル
#define INPUT_SIZE (96)
#define COEFF_SIZE (4)
#define OUTPUT_SIZE (INPUT_SIZE - COEFF_SIZE + 1)

// SEサンプル
#define SIMD_WIDTH (32) //32 -16b elements in a 512b vector
#define SE0_PARAM_OFFSET (0)
#define SE1_PARAM_OFFSET (SE0_PARAM_OFFSET + SE_PARAM_SIZE)

// SEの転置アクセスサンプル
/*
#define IMG_WIDTH  (16)
#define IMG_HEIGHT (32)
#define IMG_STRIDE (IMG_WIDTH)
#define NUM_CH     (3)
*/

using namespace std;

/*
void binNaive(char src[N], char dst[N], char lowBound, char upBound, char inVal,
              char outVal)
{
    for (int i = 0; i < N; i++)
    {
        if (src[i] < lowBound || src[i] > upBound)
        {
            dst[i] = outVal;
        }
        else
        {
            dst[i] = inVal;
        }
    }
}

void binC6Opt(char src[N], char dst[N], const char lowBound, const char upBound,
              const char inVal, const char outVal)
{
    unsigned long long lowBounds = (lowBound << 24) + (lowBound << 16)
            + (lowBound << 8) + lowBound;
    lowBounds = (lowBounds << 32) + lowBounds;
    unsigned long long upBounds = (upBound << 24) + (upBound << 16)
            + (upBound << 8) + upBound;
    upBounds = (upBounds << 32) + upBounds;

    unsigned long long inVals = (inVal << 24) + (inVal << 16) + (inVal << 8)
            + inVal;
    inVals = (inVals << 32) + inVals;
    unsigned long long outVals = (outVal << 24) + (outVal << 16) + (outVal << 8)
            + outVal;
    outVals = (outVals << 32) + outVals;

    unsigned long long startTsc[N], currentTsc[N];

    int i = 0;
    while (i < N)
    {
        //startTsc[i]=__TSC;

        unsigned long long val = _mem8_const(&src[i]);
        unsigned long long loOk = _dcmpgtu4(val, lowBounds);
        unsigned long long hiOk = _dcmpgtu4(upBounds, val);
        unsigned long long inOrNot = loOk & hiOk;
        unsigned long long valid = _dxpnd4(inOrNot);
        unsigned long long invalid = ~valid;

        unsigned long long ins = (valid & inVals);
        unsigned long long outs = (invalid & outVals);
        _mem8(&dst[i]) = (ins | outs);

        //currentTsc[i]=__TSC;

        i += 8;

        /*
         printf("val:%lx\n",val);
         printf("val:%lx\n",(val>>32));
         printf("loOk:%lx\n",loOk);
         printf("loOk:%lx\n",(loOk>>32));
         printf("hiOk:%lx\n",hiOk);
         printf("hiOk:%lx\n",(hiOk>>32));
         printf("inOrNot:%lx\n",inOrNot);
         printf("inOrNot:%lx\n",(inOrNot>>32));
         printf("valid:%lx\n",valid);
         printf("valid:%lx\n",(valid>>32));
         printf("inalid:%lx\n",invalid);
         printf("inalid:%lx\n",(invalid>>32));
         printf("ins:%lx\n",ins);
         printf("ins:%lx\n",(ins>>32));
         printf("outs:%lx\n",outs);
         printf("outs:%lx\n",(outs>>32));
         printf("res:%lx\n",(ins|outs));
         printf("res:%lx\n",((ins|outs)>>32));
         */
    //}

    /*for(i=0;i<N;i+=8){
     printf("%lld cycle\n", currentTsc[i]-startTsc[i]);
     }*/
//}

void binC7Opt(char src[N], char dst[N], const char lowBound, const char upBound,
              const char inVal, const char outVal)
{
    uchar64 lowBounds;
    uchar64 upBounds;
    char64 inVals;
    char64 outVals;

    for (int i = 0; i < 64; i++)
    {
        lowBounds.s[i] = lowBound;
        upBounds.s[i] = upBound;
        inVals.s[i] = inVal;
        outVals.s[i] = outVal;
    }

    uchar64 *srcPtr = reinterpret_cast<uchar64*>(src);
    char64 *dstPtr = reinterpret_cast<char64*>(dst);

    for (int i = 0; i < N; i += 64)
    {

        uchar64 val = *srcPtr;

        bool64 loOk = __cmp_gt_bool(val, lowBounds);
        bool64 hiOk = __cmp_gt_bool(upBounds, val);

        bool64 inOrNot = __and(loOk, hiOk);
        char64 valid = __expand(inOrNot);
        char64 invalid = ~valid;

        char64 ins = valid & inVals;
        char64 outs = invalid & outVals;
        *dstPtr = ins | outs;

        srcPtr++;
        dstPtr++;

        /*
         for(int j=0;j<64;j++){
         printf("loOk %d:%d\n",j,loOk.s[j]);
         printf("hiOk %d:%d\n",j,hiOk.s[j]);
         printf("valid %d:%d\n",j,valid.s[j]);
         }
         */

        /*
         printf("val:%llx\n",val);
         printf("loOk:%llx\n",loOk);
         printf("hiOk:%llx\n",hiOk);
         printf("inOrNot:%llx\n",inOrNot);
         printf("valid:%llx\n",valid);
         printf("inalid:%llx\n",invalid);
         printf("ins:%llx\n",ins);
         printf("outs:%llx\n",outs);
         printf("res:%llx\n",(ins|outs));
         */

    }
}

void sobelXNaive(char src[N], char dst[N], int width)
{
    int height = N / width;
    for (int i = 0; i < N; i++)
    {
        int x = i % width;
        int y = i / width;

        if (x == 0 || x == width - 1)
        {
            dst[i] = 0;
            continue;
        }

        if (y == 0 || y == height - 1)
        {
            dst[i] = 0;
            continue;
        }

        char line1 = src[i - width + 1] - src[i - width - 1];
        char line2 = 2 * (src[i + 1] - src[i - 1]);
        char line3 = src[i + width + 1] - src[i + width - 1];

        uchar val = line1 + line2 + line3;
        val += 128;
        dst[i] = val;
    }
}

void testSobelX()
{
    char src[N];
    char dst[N];
    char dst_expect[N];
    for (int i = 0; i < N; i++)
    {
        src[i] = i;
    }

    char src2[N];
    char dst2[N];
    char dst2_expect[N];
    for (int i = 0; i < N; i++)
    {
        src2[i] = N - i;
    }

    // 正解値を計算しておく
    sobelXNaive(src, dst_expect, Width);
    sobelXNaive(src2, dst2_expect, Width);

    unsigned long long startTsc, currentTsc;
    startTsc = __TSC;

    // ターゲット関数
    sobelXNaive(src, dst, Width);

    currentTsc = __TSC;
    printf("%lld cycle\n", currentTsc - startTsc);

    // 期待値チェック
    for (int i = 0; i < N; i++)
    {
        if (dst[i] != dst_expect[i])
        {
            printf("Invalid at %d:%d, res-> %d, expect-> %d\n", i, src[i],
                   dst[i], dst_expect[i]);
        }
        printf("%d:%d,%d\n", i, src[i], dst[i]);
    }

    startTsc = __TSC;

    // ターゲット関数
    sobelXNaive(src2, dst2, Width);

    currentTsc = __TSC;
    printf("%lld cycle\n", currentTsc - startTsc);

    for (int i = 0; i < N; i++)
    {
        if (dst2[i] != dst2_expect[i])
        {
            printf("Invalid at %d:%d, res-> %d, expect-> %d\n", i, src2[i],
                   dst2[i], dst2_expect[i]);
        }
        printf("%d:%d,%d\n", i, src2[i], dst2[i]);
    }

}

void testMemSpeed(int loopSize)
{
    // 適当に初期化。10MB。
    // L1Dのサイズは32KB。TDA4VMのデフォルト値。
    // 参考→https://www.ti.com/lit/an/spradc1/spradc1.pdf?ts=1714959868424&ref_url=https%253A%252F%252Fwww.google.com%252F
    // 大きいと処理が長すぎるので1MBで扱う
    int size = 1024 * 1024 * 5;
    int blockSize = 128;
    char *buf = (char*) malloc(size);
    char *top = buf;
    for (int i = 0; i < size; i++)
    {
        *(buf + i) = i;
    }

    char *cpyData = (char*) malloc(blockSize);
    for (int i = 0; i < blockSize; i++)
    {
        *(cpyData + i) = 2;
    }

    int loopCount = size / blockSize;

    unsigned long long startTsc, currentTsc;
    float clockRate = 1000000.0f;
    startTsc = __TSC;

    // コピーを走らせる
    for (int i = 0; i < loopCount; i++)
    {
        memcpy(top, cpyData, blockSize);
        top = (buf + (i * blockSize % loopSize));
    }

    currentTsc = __TSC;
    float elapsedMilliseconds = (currentTsc - startTsc) / clockRate;
    printf("Mem copy : %lld cycles\n", currentTsc - startTsc);
    printf("Mem copy : %f ms\n", elapsedMilliseconds);

    free(buf);
    free(cpyData);
}

void seTest(void){
    //Declare input array
    int16_t inArray[INPUT_SIZE];

    //Declare coeff array
    int16_t coeffArray[COEFF_SIZE];

    //Initialize input array
    for(int32_t ctr = 0; ctr < INPUT_SIZE; ctr++) {
        inArray[ctr] = ctr;
    }

    //Initialize coeff array
    for(int32_t ctr = 0; ctr < COEFF_SIZE; ctr++) {
        coeffArray[ctr] = ctr + 1;
    }

    //Display input array
    std::cout << "input array = ";
    for(int32_t ctr = 0; ctr < INPUT_SIZE; ctr++) {
        std::cout << inArray[ctr] << ", ";
    }
    std::cout << "\n\n";

    //Display coeff array
    std::cout << "coeff array = ";
    for(int32_t ctr = 0; ctr < COEFF_SIZE; ctr++) {
        std::cout << coeffArray[ctr] << ", ";
    }

    //Declare output array
    int16_t outArray[INPUT_SIZE];

    //Initialize output array
    for(int32_t ctr = 0; ctr < OUTPUT_SIZE; ctr++) {
        outArray[ctr] = ctr;
    }

    // SE試す
    //Setup Streaming Engine 0 to fetch input data
    __SE_TEMPLATE_v1 seParam0 = __gen_SE_TEMPLATE_v1();
    seParam0.DIMFMT   = __SE_DIMFMT_3D;
    seParam0.ELETYPE  = __SE_ELETYPE_16BIT;
    seParam0.VECLEN   = __SE_VECLEN_32ELEMS;
    seParam0.ICNT0 = SIMD_WIDTH;
    seParam0.ICNT1 = COEFF_SIZE;
    seParam0.DIM1 = 1;          //Stride by 1 element
    seParam0.ICNT2 = INPUT_SIZE/SIMD_WIDTH;
    seParam0.DIM2 = SIMD_WIDTH; //Stride by 32 elements

    //Setup Streaming Engine 1 to fetch and duplicate coefficients
    __SE_TEMPLATE_v1 seParam1 = __gen_SE_TEMPLATE_v1();
    seParam1.DIMFMT   = __SE_DIMFMT_3D;
    seParam1.ELETYPE   = __SE_ELETYPE_16BIT;
    seParam1.ELEDUP    = __SE_ELEDUP_32X;
    seParam1.VECLEN    = __SE_VECLEN_32ELEMS;
    seParam1.ICNT0 = COEFF_SIZE;
    seParam1.ICNT1 = INPUT_SIZE/SIMD_WIDTH;
    seParam1.DIM1  = 0;
    seParam1.ICNT2 = 1;
    seParam1.DIM2 = 0;

    int16_t *pIn    = &inArray[0];    //pointer to input array
    int16_t *pCoeff = &coeffArray[0]; //pointer to coeff array
    int16_t *pOut   = &outArray[0];   //pointer to output array
    short32* pOut2=(short32 *)&outArray[0];

    int32_t itr = 0;

    //Read template and open Streaming Engine 0
    __SE0_OPEN(pIn, seParam0);

    //Read template and open Streaming Engine 1
    __SE1_OPEN(pCoeff, seParam1);

    //Loop iterates for OUTPUT_SIZE/SIMD_WIDTH
    for(int32_t j = 0; j < OUTPUT_SIZE; j+=SIMD_WIDTH) {

        //Initialize output vector
        short32 vOut = (short32)(0);

        //Loop iterates for number of coefficients
        for(int32_t i = 0; i < COEFF_SIZE; i++) {

            std::cout << " Iteration: " << itr++;
            std::cout << " Coeff: "<< i << "\n";

            //Read a vector of 32-16b elements from input Array
            short32 vIn = __SE0ADV(short32);

            //Load and duplicate each coefficient 32 times
            short32 vCoeff = __SE1ADV(short32);

            //Multiply input with coefficient and accumulate the result
            vOut += (vIn * vCoeff);
        }

        //Store the result
        *pOut2 = vOut;

        for(int i=0;i<32;i++){
            std::cout << pOut[j + i] << " ";
        }
        std::cout << "\n\n";
        pOut2+=SIMD_WIDTH;
    }

    //Close Streaming Engine
    __SE1_CLOSE();
    __SE0_CLOSE();
}

/*
void seTestTranspose(){

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
}
*/

int main(void)
{
    printf("Hello World!\n");

    //testSobelX();

    //testMemSpeed(256 * 1024);

    //testMemSpeed(128 * 1024);

    //testMemSpeed(64 * 1024);

    //testMemSpeed(32 * 1024);

    //testMemSpeed(16 * 1024);

    //testMemSpeed(8 * 1024);

    //testMemSpeed(4 * 1024);

    //testMemSpeed(128);

    //testMemSpeed(256);

    //testMemSpeed(512);

    //testMemSpeed(1 * 1024);

    //testMemSpeed(2 * 1024);

    seTest();

    return 0;
}
}
