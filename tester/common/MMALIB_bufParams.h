/******************************************************************************
*                                                                             *
* module name       :MMALIB                                                   *
*                                                                             *
* module descripton :Matrix Multiply Accelerator Library module for C7x+MMA   *
*                                                                             *
* Copyright (C) 2017-2018 Texas Instruments Incorporated - http://www.ti.com/ *
* ALL RIGHTS RESERVED                                                         *
*                                                                             *
******************************************************************************/

/**
  ******************************************************************************
  *
  *  @brief    File to hold buffer parameter related info for MMALIB
  *
  *  @version  0.1 - Jan 2018 : Initial Version
  *            
 *******************************************************************************
*/

#ifndef MMALIB_BUFPARAMS_H_
#define MMALIB_BUFPARAMS_H_

#ifndef __OPENCL_VERSION__
#include <stdint.h>
#endif


/*!
 * @addtogroup MMALIB_COMMON
 * @details
 */

/**@{*/

static inline int32_t MMALIB_sizeof(uint32_t type);

/*! \brief The enumeration of all data types. */
typedef enum _MMALIB_data_type_e
{
   /* Standard fixed point data types */
   MMALIB_INT8    = 0,  /*!<  Signed 8-bit integer */
   MMALIB_INT16   = 1,  /*!<  Signed 16-bit integer */
   MMALIB_INT32   = 2,  /*!<  Signed 32-bit integer */
   MMALIB_INT64   = 3,  /*!<  Signed 64-bit integer */
   MMALIB_INT128  = 4,  /*!<  Signed 128-bit integer */

   MMALIB_UINT8   = 5,  /*!<  Unsigned 8-bit integer */
   MMALIB_UINT16  = 6,  /*!<  Unsigned 16-bit integer */
   MMALIB_UINT24  = 7,  /*!<  Unsigned 24-bit integer */
   MMALIB_UINT32  = 8,  /*!<  Unsigned 32-bit integer */
   MMALIB_UINT64  = 9,  /*!<  Unsigned 64-bit integer */
   MMALIB_UINT128 = 10, /*!<  Unsigned 128-bit integer */

   /* Standard floating point data types */
   MMALIB_FLOAT16 = 11,  /*!<  16-bit floating point number */
   MMALIB_FLOAT32 = 12,  /*!<  32-bit floating point number */
   MMALIB_FLOAT64 = 13   /*!<  32-bit floating point number */

} MMALIB_data_type_e;


/* ---------------------------------------------------------------- */
/*  MISRAC Rule 4.9(DEFINE.FUNC) Deviation: The advisory is not     */
/*  being addressed                                                 */
/* ---------------------------------------------------------------- */  
// MACRO for determining signedness of MMALIB data types; fragile in that it depends on ordering of MMALIB_data_type_e
#define MMALIB_ISSIGNED(type) ((type) < MMALIB_UINT8 || (type) > MMALIB_UINT128)
#define MMALIB_SIGNEDNESS(type) (MMALIB_ISSIGNED((type)) ? "signed" : "unsigned")

/*! \brief The enumeration of multidimensional buffer types. */
typedef enum _MMALIB_dimension_e {
   MMALIB_2D= 0,   /*!<  Two dimensional buffer */
   MMALIB_3D= 1,   /*!<  Three dimensional buffer */
   MMALIB_4D= 2,   /*!<  Four dimensional buffer */
   MMALIB_5D= 3,   /*!<  Five dimensional buffer */
   MMALIB_6D= 4    /*!<  Six dimensional buffer */
} MMALIB_dimension_e;

/*! \brief A structure for a 1 dimensional buffer descriptor. */
typedef struct
{
   uint32_t data_type;    /*!< \brief Values are of type MMALIB_data_type_e. */

   uint32_t dim_x;        /*!< \brief Width of buffer in X dimension in elements. */

} MMALIB_bufParams1D_t;

/*! \brief A structure for a 2 dimensional buffer descriptor. */
typedef struct
{
   uint32_t data_type;     /*!< \brief Values are of type MMALIB_data_type_e. */

   uint32_t dim_x;        /*!< \brief Width of buffer in X dimension in elements. */
   uint32_t dim_y;        /*!< \brief Height of buffer in Y dimension in elements. */

   int32_t  stride_y;     /*!< \brief Stride in Y dimension in bytes. */

} MMALIB_bufParams2D_t;

/*! \brief A structure for a 3 dimensional buffer descriptor. */
typedef struct
{
   uint32_t data_type;     /*!< \brief Values are of type MMALIB_data_type_e. */
 
   uint32_t dim_x;        /*!< \brief Width of buffer in X dimension in elements. */

   uint32_t dim_y;        /*!< \brief Height of buffer in Y dimension in elements. */
   int32_t  stride_y;     /*!< \brief Stride in Y dimension in bytes. */
 
   uint32_t dim_z;        /*!< \brief Depth of patch in Z dimension in elements. */
   int32_t  stride_z;     /*!< \brief Stride in Z dimension in bytes. */
 
} MMALIB_bufParams3D_t;

/*! \brief A structure for a N dimensional buffer descriptor. */
typedef struct
{
    uint32_t data_type;     /*!< \brief Values are of type MMALIB_data_type_e. */

    uint32_t dim_0;

    uint32_t dim_1;
    int32_t  stride_1; /*!< \brief  Stride in bytes */

    uint32_t dim_2;  /*!< \brief  Uninitialized when dimension = {MMALIB_2D} */
    int32_t  stride_2; /*!< \brief  Stride in bytes; Uninitialized when dimension = {MMALIB_2D} */

    uint32_t dim_3;  /*!< \brief  Uninitialized when dimension = {MMALIB_2D, MMALIB_3D} */
    int32_t  stride_3; /*!< \brief  Stride in bytes; Uninitialized when dimension = {MMALIB_2D, MMALIB_3D} */

    uint32_t dim_4;  /*!< \brief  Uninitialized when dimension = {MMALIB_2D, MMALIB_3D, MMALIB_4D} */
    int32_t  stride_4; /*!< \brief  Stride in bytes; Uninitialized when dimension = {MMALIB_2D, MMALIB_3D, MMALIB_4D} */

    uint32_t dim_5;  /*!< \brief  Uninitialized when dimension = {MMALIB_2D, MMALIB_3D, MMALIB_4D, MMALIB_5D} */
    int32_t  stride_5; /*!< \brief  Stride in bytes; Uninitialized when dimension = {MMALIB_2D, MMALIB_3D, MMALIB_4D, MMALIB_5D} */

    uint32_t dimension;    /*!< \brief Values are of type MMALIB_dimension_e. */

} MMALIB_bufParamsND_t;


/*! \brief Inline function returns number of bytes per element given a type of \ref MMALIB_data_type_e. */
static inline int32_t MMALIB_sizeof(uint32_t type)
{

   int32_t size = 1;

   if((type == (uint32_t)MMALIB_INT8)  ||
      (type == (uint32_t)MMALIB_UINT8))
   {
      size = 1;
   }
   else if((type == (uint32_t)MMALIB_INT16)  ||
	   (type == (uint32_t)MMALIB_UINT16) ||
	   (type == (uint32_t)MMALIB_FLOAT16))
   {
      size = 2;
   }
   else if(type == (uint32_t)MMALIB_UINT24)
   {
      size = 3;
   }
   else if((type == (uint32_t)MMALIB_INT32)  ||
	   (type == (uint32_t)MMALIB_UINT32) ||
	   (type == (uint32_t)MMALIB_FLOAT32))
   {
      size = 4;
   }
   else if((type == (uint32_t)MMALIB_INT64)  ||
	   (type == (uint32_t)MMALIB_UINT64) ||
	   (type == (uint32_t)MMALIB_FLOAT64))
   {
      size = 8;
   }
   else if((type == (uint32_t)MMALIB_INT128)  ||
	   (type == (uint32_t)MMALIB_UINT128))
   {
      size = 16;
   }
   else {
      size = 1;
   }

   return size;
}

/**@}*/
#endif /* MMALIB_BUFPARAMS_H_ */
