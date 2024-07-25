#ifndef FRAME_CONVERT_H
#define FRAME_CONVERT_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include "sample_comm_nnie.h"

#if __cplusplus
extern "C"
{
#endif

#define IMG_FULL_CHN 3
#define IVE_ALIGN 16

    void Ive_Img_Destroy(IVE_IMAGE_S *img);
    HI_S32 Mat_File_Convert_To_IVE_YUV(IVE_IMAGE_S *Srcimg, const char *name);

#ifdef __cplusplus
}
#endif
#endif