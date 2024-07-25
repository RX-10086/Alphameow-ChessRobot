#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include "frame_convert.h"

#include "unistd.h"
#include "sample_comm_nnie.h"
#include "sample_comm_ive.h"

using namespace std;
using namespace cv;

// IVE_IMAGE_S格式转换成VIDEO_FRAME_INFO_S格式, 仅复制数据指针, 不复制数据
static HI_S32 Orig_Img_To_Frame(const IVE_IMAGE_S *img, VIDEO_FRAME_INFO_S *frm)
{
    static const int chnNum = 2;
    IVE_IMAGE_TYPE_E enType = img->enType;
    if (memset_s(frm, sizeof(*frm), 0, sizeof(*frm)) != EOK)
    {
        HI_ASSERT(0);
    }

    frm->stVFrame.u32Width = img->u32Width;
    frm->stVFrame.u32Height = img->u32Height;

    if (enType == IVE_IMAGE_TYPE_YUV420SP)
    {
        frm->stVFrame.enPixelFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    }
    else if (enType == IVE_IMAGE_TYPE_YUV422SP)
    {
        frm->stVFrame.enPixelFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_422;
    }
    else
    {
        HI_ASSERT(0);
        return -1;
    }

    for (int i = 0; i < chnNum; i++)
    {
        frm->stVFrame.u64PhyAddr[i] = img->au64PhyAddr[i];
        frm->stVFrame.u64VirAddr[i] = img->au64VirAddr[i];
        frm->stVFrame.u32Stride[i] = img->au32Stride[i];
    }
    return 0;
}

// 销毁IVE image
void Ive_Img_Destroy(IVE_IMAGE_S *img)
{
    for (int i = 0; i < IMG_FULL_CHN; i++)
    {
        if (img->au64PhyAddr[0] && img->au64VirAddr[0])
        {
            HI_MPI_SYS_MmzFree(img->au64PhyAddr[i], (void *)((uintptr_t)img->au64VirAddr[i]));
            img->au64PhyAddr[i] = 0;
            img->au64VirAddr[i] = 0;
        }
    }
    if (memset_s(img, sizeof(*img), 0, sizeof(*img)) != EOK)
    {
        HI_ASSERT(0);
    }
}

// OpenCV中的Mat变量转为IVE图像
static HI_S32 Mat_Write_Data_To_Ive(Mat &Sourcefile, IVE_IMAGE_S *Dstimage, int wid, int hig)
{
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U8 *srcFILE = NULL;
    HI_U8 *data = Sourcefile.data;
    srcFILE = (HI_U8 *)(HI_UL)Dstimage->au64VirAddr[0]; // 虚拟内存地址获取
    for (int i = 0; i < hig; i++)
    {
        memcpy_s(srcFILE, wid * 3 * sizeof(HI_U8), data, wid * 3 * sizeof(HI_U8));
        srcFILE += Dstimage->au32Stride[0] * 3;
        data += Dstimage->au32Stride[0] * 3;
    }
    return s32Ret;
}

// 读取一张jpg格式图像，先转为Mat变量，再转为YUV格式的IVE图像
HI_S32 Mat_File_Convert_To_IVE_YUV(IVE_IMAGE_S *Srcimg, const char *name)
{
    int s32Ret = HI_SUCCESS;

    IVE_IMAGE_S Tempimg;
    Mat SrcMat = imread(name);
    Size SrcSize = SrcMat.size();
    int WIDTH = SrcSize.width; // 图像的宽、高
    int HEIGHT = SrcSize.height;

    SAMPLE_COMM_IVE_CheckIveMpiInit();

    s32Ret = SAMPLE_COMM_IVE_CreateImage(&Tempimg, IVE_IMAGE_TYPE_U8C3_PACKAGE, WIDTH, HEIGHT); // 创建图像缓存
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("IVE Creat Error, err code = %#x", s32Ret);
        return s32Ret;
    }

    s32Ret = Mat_Write_Data_To_Ive(SrcMat, &Tempimg, WIDTH, HEIGHT); // 写入图像数据
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Mat Write Data to IVE image Error, err code = %#x", s32Ret);
        return s32Ret;
    }

    s32Ret = SAMPLE_COMM_IVE_CreateImage(Srcimg, IVE_IMAGE_TYPE_YUV420SP, WIDTH, HEIGHT);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("IVE Creat Error, err code = %#x", s32Ret);
        return s32Ret;
    }

    IVE_HANDLE iveHnd;
    HI_BOOL finish;
    IVE_CSC_CTRL_S stCSCCtrl = {IVE_CSC_MODE_PIC_BT709_RGB2YUV}; // IVE_CSC_MODE_PIC_BT709_RGB2YUV

    s32Ret = HI_MPI_IVE_CSC(&iveHnd, &Tempimg, Srcimg, &stCSCCtrl, HI_TRUE); // 色彩空间转换，RGB2YUV
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("IVE-CSC Error, err code = %#x", s32Ret);
        return s32Ret;
    }
    s32Ret = HI_MPI_IVE_Query(iveHnd, &finish, HI_TRUE); // 查询已创建任务完成情况，与上面的色彩空间转换联用
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("IVE-Query Error, err code = %#x", s32Ret);
        return s32Ret;
    }
    Ive_Img_Destroy(&Tempimg);

    s32Ret = SAMPLE_COMM_IVE_IveMpiExit();
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("IVE Exit Error, err code = %#x", s32Ret);
        return s32Ret;
    }
    return s32Ret;
}
