#ifndef NNIE_WORK_H
#define NNIE_WORK_H

#include <stdint.h>
#include "sample_comm_nnie.h"

#if __cplusplus
extern "C"
{
#endif

    HI_S32 Chess_Cnn_Load_Model(const char *modelname);
    HI_S32 Chess_Cnn_Work_Func(IVE_IMAGE_S *srcImage, HI_S32 *Run_Result);

#ifdef __cplusplus
}
#endif
#endif