#include <stdio.h>
#include <c7x.h>
#include <mmalib.h>
#include <stdint.h>
#include "convTest.h"
#include "deconvTest.h"
#include "linalgTest.h"

/**
 * hello.cpp
 */

void updatePC(){
    //  �v���O�������܂Ƃ߂�64bit��Ԃɑޔ�����B

}

int main(void)
{
    printf("Hello World!\n");

    char* hoge=(char*)0xA0300000;
    *hoge=1;
    *(hoge+1)=2;

    //mmaLinalg();
    mmaConvFullSpeed();
    //mmaDeconv();

    //seTestTranspose();

    return 0;
}
