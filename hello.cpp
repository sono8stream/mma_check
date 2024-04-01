#include <stdio.h>
#include <c7x.h>
#include <c7x_scalable.h>
#include <c6x_migration.h>

/**
 * hello.cpp
 */

const int N=1000;

void binNaive(char src[N], char

              dst[N], char lowBound, char upBound, char inVal, char outVal){
    for(int i=0;i<N;i++){
        if(src[i]<lowBound||src[i]>upBound){
            dst[i]=outVal;
        }
        else{
            dst[i]=inVal;
        }
    }
}

void binC6Opt(char src[N], char dst[N], const char lowBound, const char upBound, const char inVal, const char outVal){
    unsigned long long lowBounds=(lowBound<<24)+(lowBound<<16)+(lowBound<<8)+lowBound;
    lowBounds=(lowBounds<<32)+lowBounds;
    unsigned long long upBounds=(upBound<<24)+(upBound<<16)+(upBound<<8)+upBound;
    upBounds=(upBounds<<32)+upBounds;

    unsigned long long inVals=(inVal<<24)+(inVal<<16)+(inVal<<8)+inVal;
    inVals=(inVals<<32)+inVals;
    unsigned long long outVals=(outVal<<24)+(outVal<<16)+(outVal<<8)+outVal;
    outVals=(outVals<<32)+outVals;

    unsigned long long startTsc[N],currentTsc[N];

    int i=0;
    while(i<N){
        //startTsc[i]=__TSC;

        unsigned long long val=_mem8_const(&src[i]);
        unsigned long long loOk=_dcmpgtu4(val,lowBounds);
        unsigned long long hiOk=_dcmpgtu4(upBounds,val);
        unsigned long long inOrNot=loOk & hiOk;
        unsigned long long valid=_dxpnd4(inOrNot);
        unsigned long long invalid=~valid;

        unsigned long long ins=(valid & inVals);
        unsigned long long outs=(invalid & outVals);
        _mem8(&dst[i]) = (ins | outs);

        //currentTsc[i]=__TSC;

        i+=8;

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
    }

    /*for(i=0;i<N;i+=8){
        printf("%lld cycle\n", currentTsc[i]-startTsc[i]);
    }*/
}


void binC7Opt(char src[N], char dst[N], const char lowBound, const char upBound, const char inVal, const char outVal){
    uchar8 lowBounds=(lowBound<<24)+(lowBound<<16)+(lowBound<<8)+lowBound;
    lowBounds=(lowBounds<<32)+lowBounds;
    uchar8 upBounds=(upBound<<24)+(upBound<<16)+(upBound<<8)+upBound;
    upBounds=(upBounds<<32)+upBounds;

    char8 inVals=(inVal<<24)+(inVal<<16)+(inVal<<8)+inVal;
    inVals=(inVals<<32)+inVals;
    char8 outVals=(outVal<<24)+(outVal<<16)+(outVal<<8)+outVal;
    outVals=(outVals<<32)+outVals;

    uchar64 lowBounds2;
    uchar64 upBounds2;
    char64 inVals2;
    char64 outVals2;

    for(int i=0;i<64;i++){
        lowBounds2.s[i]=lowBound;
        upBounds2.s[i]=lowBound;
        inVals2.s[i]=inVal;
        outVals2.s[i]=outVal;
    }

    unsigned long long startTsc[N],currentTsc[N];

    uchar8* srcPtr= reinterpret_cast<uchar8*>(src);
    char8* dstPtr= reinterpret_cast<char8*>(dst);

    for(int i=0;i<N;i+=8){
        //startTsc[i]=__TSC;

        uchar8 val=*srcPtr;

        bool8 loOk=__cmp_gt_bool(val, lowBounds);
        bool8 hiOk=__cmp_gt_bool(upBounds, val);

        bool8 inOrNot=__and(loOk, hiOk);
        char8 valid=__expand(inOrNot);
        char8 invalid=~valid;

        char8 ins=valid & inVals;
        char8 outs=invalid & outVals;
        *dstPtr= ins | outs;

        //currentTsc[i]=__TSC;

        srcPtr++;
        dstPtr++;

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

    for(int i=0;i<N;i+=8){
        printf("%lld cycle\n", currentTsc[i]-startTsc[i]);
    }
}


void sobelXNaive(){

}

int main(void)
{
    printf("Hello World!\n");

    //dma();

    char src[N];
    char dst[N];
    for(int i=0;i<N;i++){
        src[i]=N-i;
    }

    char src2[N];
    char dst2[N];
    for(int i=0;i<N;i++){
        src2[i]=i;
    }

    unsigned long long startTsc,currentTsc;
    startTsc=__TSC;

    binC7Opt(src2, dst2, 30, 75, 50, 20);

    currentTsc=__TSC;
    printf("%lld cycle\n", currentTsc-startTsc);


    for(int i=0;i<N;i++){
        //printf("%d:%d,%d\n",i,src[i],dst[i]);
    }

    for(int i=0;i<N;i++){
        printf("%d:%d,%d\n",i,src2[i],dst2[i]);
    }

    return 0;
}
