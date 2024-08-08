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

int main(void)
{
    printf("Hello World!\n");

    //mmaLinalg();
    mmaConv();
    //mmaDeconv();

    //seTestTranspose();

    return 0;
}
