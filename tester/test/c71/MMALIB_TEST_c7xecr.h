#ifndef MMALIB_TEST_C7XECR_H_
#define MMALIB_TEST_C7XECR_H_

/**
 *     C7X ECR Register Access Functions
 *  \par
 *   ================================================================================
 *
 *   @n   (C)  Copyright 2017 Texas Instruments Incorporated
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *      Redistributions of source code must retain the above copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *      Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the
 *      distribution.
 *
 *      Neither the name of Texas Instruments Incorporated nor the names of
 *      its contributors may be used to endorse or promote products derived
 *      from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void MMALIB_TEST_c7xSetTCR0 (uint64_t param);

void MMALIB_TEST_c7xSetTBR0 (uint64_t param);

void MMALIB_TEST_c7xSetMAR (uint64_t param);

void MMALIB_TEST_c7xSetSCR (uint64_t param);

// added during J7ES bring-up to perform TLB invalidate ALL after the MMU setup
// code
void MMALIB_TEST_c7xMmuTlbInvAll (void);

void MMALIB_TEST_c7xSetL2CFG (uint64_t param);

uint64_t MMALIB_TEST_c7xGetL2CFG (void);

void MMALIB_TEST_c7xSetL1DCFG (uint64_t param);

uint64_t MMALIB_TEST_c7xGetL1DCFG (void);

uint64_t MMALIB_c7xGetL1DWBINV (void);
uint64_t MMALIB_c7xGetL1DINV (void);
void     MMALIB_c7xSetL1DWBINV (uint64_t param);
void     MMALIB_c7xSetL1DINV (uint64_t param);
uint64_t MMALIB_c7xL1PCacheInvalidateAll ();

#ifdef __cplusplus
}
#endif

#endif /* end of MMALIB_TEST_C7XECR_H_ definition */
