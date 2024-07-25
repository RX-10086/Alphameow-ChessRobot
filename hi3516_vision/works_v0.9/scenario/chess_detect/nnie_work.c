#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/prctl.h>

#include "sample_comm_nnie.h"
#include "sample_comm_svp.h"
#include "resnet_infer_process.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "base_interface.h"
#include "osd_img.h"
#include "nnie_work.h"
#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif /* End of #ifdef __cplusplus */

    SAMPLE_SVP_NNIE_MODEL_S CnnModel = {0};
    SAMPLE_SVP_NNIE_PARAM_S CnnNnieParam = {0};
    SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S CnnSoftwareParam = {0};

    HI_S32 Chess_Cnn_Load_Model(const char *modelname)
    {
        HI_S32 s32Ret = HI_SUCCESS;
        s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(modelname, &CnnModel);
        return s32Ret;
    }

    HI_S32 Chess_Cnn_Work_Func(IVE_IMAGE_S *srcImage, HI_S32 *Run_Result)
    {
        HI_S32 s32Ret = HI_SUCCESS;
        SAMPLE_PRT("--------NNIE Run!--------\n");
        HI_CHAR *pcSrcFile = NULL;
        SAMPLE_SVP_NNIE_CFG_S stNnieCfg = {0};
        SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
        SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};
        stNnieCfg.pszPic = pcSrcFile;
        stNnieCfg.u32MaxInputNum = 1; // max input image num in each batch
        stNnieCfg.u32MaxRoiNum = 0;
        stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
        CnnSoftwareParam.u32TopN = 5;

        SAMPLE_COMM_SVP_CheckSysInit();

        CnnNnieParam.pstModel = &CnnModel.stModel;
        s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg, &CnnNnieParam, &CnnSoftwareParam);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(CnnNnieParam.astForwardCtrl[0].stTskBuf));
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        stInputDataIdx.u32SegIdx = 0;
        stInputDataIdx.u32NodeIdx = 0;
        s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &CnnNnieParam, &stInputDataIdx, srcImage);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        stProcSegIdx.u32SegIdx = 0;
        s32Ret = SAMPLE_SVP_NNIE_Forward(&CnnNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        s32Ret = SAMPLE_SVP_NNIE_Cnn_GetTopN(&CnnNnieParam, &CnnSoftwareParam);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        s32Ret = SAMPLE_SVP_NNIE_Cnn_PrintResult(&(CnnSoftwareParam.stGetTopN), CnnSoftwareParam.u32TopN, Run_Result);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(CnnNnieParam.astForwardCtrl[0].stTskBuf));
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Error, %#x\n", s32Ret);
            return s32Ret;
        }
        // memset_s(&CnnNnieParam, sizeof(SAMPLE_SVP_NNIE_PARAM_S), 0, sizeof(SAMPLE_SVP_NNIE_PARAM_S));
        // memset_s(&CnnSoftwareParam, sizeof(SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S), 0, sizeof(SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S));
        SAMPLE_COMM_SVP_CheckSysExit();
        return s32Ret;
    }

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */