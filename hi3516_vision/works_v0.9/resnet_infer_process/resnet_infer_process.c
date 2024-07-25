#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>
#include <sys/prctl.h>
#include <math.h>
#include <assert.h>

#include "hi_common.h"
#include "hi_comm_sys.h"
#include "hi_comm_svp.h"
#include "sample_comm_svp.h"
#include "hi_comm_ive.h"
#include "sample_svp_nnie_software.h"
#include "sample_media_ai.h"
#include "resnet_infer_process.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif /* End of #ifdef __cplusplus */

    /* Cnn 参数储存 */
    static SAMPLE_SVP_NNIE_MODEL_S s_stCnnModel = {0};
    static SAMPLE_SVP_NNIE_PARAM_S s_stCnnNnieParam = {0};
    static SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S s_stCnnSoftwareParam = {0};

    /******************************************************************************
     * function : NNIE 推理
     ******************************************************************************/
    HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                   SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx,
                                   SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant)
    {
        HI_S32 s32Ret = HI_SUCCESS;
        HI_U32 i = 0, j = 0;
        HI_BOOL bFinish = HI_FALSE;
        SVP_NNIE_HANDLE hSvpNnieHandle = 0;
        HI_U32 u32TotalStepNum = 0;

        SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
                                   SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
                                   pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

        for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
        {
            if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
            {
                for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
                {
                    u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) + j);
                }
                SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                           SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                                           u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
            }
            else
            {
                SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                           SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                                           pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                                               pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                                               pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                                               pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
            }
        }

        /*set input blob according to node name*/
        if (pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
        {
            for (i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
            {
                for (j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
                {
                    if (0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                                     pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                                     SVP_NNIE_NODE_NAME_LEN))
                    {
                        pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                            pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                        break;
                    }
                }
                SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                                          HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,can't find %d-th seg's %d-th src blob!\n",
                                          pstProcSegIdx->u32SegIdx, i);
            }
        }

        /*NNIE_Forward*/
        s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
                                         pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
                                         pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
                                         &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                  "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

        if (bInstant)
        {
            /*Wait NNIE finish*/
            while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
                                                                                    hSvpNnieHandle, &bFinish, HI_TRUE)))
            {
                usleep(100);
                SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
                                 "HI_MPI_SVP_NNIE_Query Query timeout!\n");
            }
        }
        u32TotalStepNum = 0;

        for (i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
        {
            if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
            {
                for (j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
                {
                    u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) + j);
                }
                SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                           SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                                           u32TotalStepNum * pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
            }
            else
            {
                SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                           SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
                                           pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num *
                                               pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn *
                                               pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height *
                                               pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
            }
        }

        return s32Ret;
    }
    /******************************************************************************
     * function : Cnn 软件参数 去初始化
     ******************************************************************************/
    static HI_S32 SAMPLE_SVP_NNIE_Cnn_SoftwareDeinit(SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara)
    {
        HI_S32 s32Ret = HI_SUCCESS;
        SAMPLE_SVP_CHECK_EXPR_RET(NULL == pstCnnSoftWarePara, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                  "Error, pstCnnSoftWarePara can't be NULL!\n");
        if (0 != pstCnnSoftWarePara->stGetTopN.u64PhyAddr && 0 != pstCnnSoftWarePara->stGetTopN.u64VirAddr)
        {
            SAMPLE_SVP_MMZ_FREE(pstCnnSoftWarePara->stGetTopN.u64PhyAddr,
                                pstCnnSoftWarePara->stGetTopN.u64VirAddr);
            pstCnnSoftWarePara->stGetTopN.u64PhyAddr = 0;
            pstCnnSoftWarePara->stGetTopN.u64VirAddr = 0;
        }
        return s32Ret;
    }
    /******************************************************************************
     * function : Cnn 去初始化
     ******************************************************************************/
    static HI_S32 SAMPLE_SVP_NNIE_Cnn_Deinit(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                             SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstSoftWareParam, SAMPLE_SVP_NNIE_MODEL_S *pstNnieModel)
    {

        HI_S32 s32Ret = HI_SUCCESS;
        /* 硬件参数 去初始化 */
        if (pstNnieParam != NULL)
        {
            s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit(pstNnieParam);
            SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                        "Error,SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
        }
        /* 软件参数 去初始化 */
        if (pstSoftWareParam != NULL)
        {
            s32Ret = SAMPLE_SVP_NNIE_Cnn_SoftwareDeinit(pstSoftWareParam);
            SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                        "Error,SAMPLE_SVP_NNIE_Cnn_SoftwareDeinit failed!\n");
        }
        /* 模型卸载 */
        if (pstNnieModel != NULL)
        {
            s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel(pstNnieModel);
            SAMPLE_SVP_CHECK_EXPR_TRACE(HI_SUCCESS != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                        "Error,SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
        }
        return s32Ret;
    }

    /******************************************************************************
     * function : 填充原始数据
     ******************************************************************************/
    HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg,
                                       SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                       SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx,
                                       IVE_IMAGE_S *img)
    {
        HI_U32 i = 0, j = 0, n = 0;
        HI_U32 u32Height = 0, u32Width = 0, u32Chn = 0, u32Stride = 0, u32Dim = 0;
        HI_U32 u32VarSize = 0;
        HI_S32 s32Ret = HI_SUCCESS;
        HI_U8 *pu8PicAddr = NULL;
        HI_U32 *pu32StepAddr = NULL;
        HI_U32 u32SegIdx = pstInputDataIdx->u32SegIdx;
        HI_U32 u32NodeIdx = pstInputDataIdx->u32NodeIdx;
        HI_U32 u32TotalStepNum = 0;

        /* 获取单位数据大小 */
        if (SVP_BLOB_TYPE_U8 <= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType &&
            SVP_BLOB_TYPE_YVU422SP >= pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            u32VarSize = sizeof(HI_U8);
        }
        else
        {
            u32VarSize = sizeof(HI_U32);
        }

        /* 填充源数据 */
        if (SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
        {
            HI_ASSERT(0);
        }
        else
        {
            u32Height = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Height;
            u32Width = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Width;
            u32Chn = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].unShape.stWhc.u32Chn;
            u32Stride = pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Stride;
            pu8PicAddr = SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U8, pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr);
            if (SVP_BLOB_TYPE_YVU420SP == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
            {
                for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
                {
                    // Y
                    const uint8_t *srcData = (const uint8_t *)(uintptr_t)img->au64VirAddr[0];
                    HI_ASSERT(srcData);
                    for (j = 0; j < u32Height; j++)
                    {
                        if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK)
                        {
                            HI_ASSERT(0);
                        }
                        pu8PicAddr += u32Stride;
                        srcData += img->au32Stride[0];
                    }
                    // UV
                    srcData = (const uint8_t *)(uintptr_t)img->au64VirAddr[1];
                    HI_ASSERT(srcData);
                    for (j = 0; j < u32Height / 2; j++)
                    { // 2: 1/2Height
                        if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK)
                        {
                            HI_ASSERT(0);
                        }
                        pu8PicAddr += u32Stride;
                        srcData += img->au32Stride[1];
                    }
                }
            }
            else if (SVP_BLOB_TYPE_YVU422SP == pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].enType)
            {
                for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
                {
                    // Y
                    const uint8_t *srcData = (const uint8_t *)(uintptr_t)img->au64VirAddr[0];
                    HI_ASSERT(srcData);
                    for (j = 0; j < u32Height; j++)
                    {
                        if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK)
                        {
                            HI_ASSERT(0);
                        }
                        pu8PicAddr += u32Stride;
                        srcData += img->au32Stride[0];
                    }
                    // UV
                    srcData = (const uint8_t *)(uintptr_t)img->au64VirAddr[1];
                    HI_ASSERT(srcData);
                    for (j = 0; j < u32Height; j++)
                    {
                        if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK)
                        {
                            HI_ASSERT(0);
                        }
                        pu8PicAddr += u32Stride;
                        srcData += img->au32Stride[1];
                    }
                }
            }
            else
            {
                for (n = 0; n < pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num; n++)
                {
                    for (i = 0; i < u32Chn; i++)
                    {
                        for (j = 0; j < u32Height; j++)
                        {
                            const uint8_t *srcData = (const uint8_t *)(uintptr_t)img->au64VirAddr[i];
                            HI_ASSERT(srcData);
                            for (j = 0; j < u32Height; j++)
                            {
                                if (memcpy_s(pu8PicAddr, u32Width * u32VarSize, srcData, u32Width * u32VarSize) != EOK)
                                {
                                    HI_ASSERT(0);
                                }
                                pu8PicAddr += u32Stride;
                                srcData += img->au32Stride[i];
                            }
                        }
                    }
                }
            }
            SAMPLE_COMM_SVP_FlushCache(pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64PhyAddr,
                                       SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u64VirAddr),
                                       pstNnieParam->astSegData[u32SegIdx].astSrc[u32NodeIdx].u32Num * u32Chn * u32Height * u32Stride);
        }

        return HI_SUCCESS;
    FAIL:

        return HI_FAILURE;
    }
    /******************************************************************************
     * function : Cnn 软件参数 初始化
     ******************************************************************************/
    static HI_S32 SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg,
                                                       SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara)
    {
        HI_U32 u32GetTopNMemSize = 0;
        HI_U32 u32GetTopNAssistBufSize = 0;
        HI_U32 u32GetTopNPerFrameSize = 0;
        HI_U32 u32TotalSize = 0;
        HI_U32 u32ClassNum = pstCnnPara->pstModel->astSeg[0].astDstNode[0].unShape.stWhc.u32Width;
        HI_U64 u64PhyAddr = 0;
        HI_U8 *pu8VirAddr = NULL;
        HI_S32 s32Ret = HI_SUCCESS;

        /*get mem size*/
        u32GetTopNPerFrameSize = pstCnnSoftWarePara->u32TopN * sizeof(SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S);
        u32GetTopNMemSize = SAMPLE_SVP_NNIE_ALIGN16(u32GetTopNPerFrameSize) * pstNnieCfg->u32MaxInputNum;
        u32GetTopNAssistBufSize = u32ClassNum * sizeof(SAMPLE_SVP_NNIE_CNN_GETTOPN_UNIT_S);
        u32TotalSize = u32GetTopNMemSize + u32GetTopNAssistBufSize;

        /*malloc mem*/
        s32Ret = SAMPLE_COMM_SVP_MallocMem("SAMPLE_CNN_INIT", NULL, (HI_U64 *)&u64PhyAddr,
                                           (void **)&pu8VirAddr, u32TotalSize);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                  "Error,Malloc memory failed!\n");
        memset(pu8VirAddr, 0, u32TotalSize);

        /* init GetTopn */
        pstCnnSoftWarePara->stGetTopN.u32Num = pstNnieCfg->u32MaxInputNum;
        pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Chn = 1;
        pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Height = 1;
        pstCnnSoftWarePara->stGetTopN.unShape.stWhc.u32Width = u32GetTopNPerFrameSize / sizeof(HI_U32);
        pstCnnSoftWarePara->stGetTopN.u32Stride = SAMPLE_SVP_NNIE_ALIGN16(u32GetTopNPerFrameSize);
        pstCnnSoftWarePara->stGetTopN.u64PhyAddr = u64PhyAddr;
        pstCnnSoftWarePara->stGetTopN.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

        /* init AssistBuf */
        pstCnnSoftWarePara->stAssistBuf.u32Size = u32GetTopNAssistBufSize;
        pstCnnSoftWarePara->stAssistBuf.u64PhyAddr = u64PhyAddr + u32GetTopNMemSize;
        pstCnnSoftWarePara->stAssistBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr + u32GetTopNMemSize;

        return s32Ret;
    }

    /******************************************************************************
     * function : Cnn 参数初始化
     ******************************************************************************/
    HI_S32 SAMPLE_SVP_NNIE_Cnn_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg,
                                         SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara)
    {
        HI_S32 s32Ret = HI_SUCCESS;
        /* 硬件参数初始化 */
        s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit(pstNnieCfg, pstCnnPara);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error(%#x),SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n", s32Ret);

        /* 软件参数初始化 */
        if (pstCnnSoftWarePara != NULL)
        {
            s32Ret = SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit(pstNnieCfg, pstCnnPara, pstCnnSoftWarePara);
            SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, INIT_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                       "Error(%#x),SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit failed!\n", s32Ret);
        }

        return s32Ret;
    INIT_FAIL_0:
        s32Ret = SAMPLE_SVP_NNIE_Cnn_Deinit(pstCnnPara, pstCnnSoftWarePara, NULL);
        SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                  "Error(%#x),SAMPLE_SVP_NNIE_Cnn_Deinit failed!\n", s32Ret);
        return HI_FAILURE;
    }

    /******************************************************************************
     * function : Cnn 数据处理
     ******************************************************************************/
    HI_S32 SAMPLE_SVP_NNIE_Cnn_PrintResult(SVP_BLOB_S *pstGetTopN, HI_U32 u32TopN, int *final_num)
    {
        HI_U32 i = 0, j = 0;
        HI_U32 *pu32Tmp = NULL;
        HI_U32 u32Stride = pstGetTopN->u32Stride;
        SAMPLE_SVP_CHECK_EXPR_RET(NULL == pstGetTopN, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                  "Error,pstGetTopN can't be NULL!\n");
        for (j = 0; j < pstGetTopN->u32Num; j++)
        {
            SAMPLE_SVP_TRACE_INFO("==== The %dth image info====\n", j);
            pu32Tmp = (HI_U32 *)((HI_UL)pstGetTopN->u64VirAddr + j * u32Stride);
            for (i = 0; i < u32TopN * 2; i += 2)
            {
                SAMPLE_SVP_TRACE_INFO("num %d -- score:%d\n", pu32Tmp[i], pu32Tmp[i + 1] * 100 / 4096);
            }
        }
        *final_num = pu32Tmp[0];
        return HI_SUCCESS;
    }
    /******************************************************************************
     * function : Cnn 运行全过程
     ******************************************************************************/
    void SAMPLE_SVP_NNIE_Cnn(IVE_IMAGE_S *IMG, const char *pcmodelname, int *finalresult)
    {
        HI_CHAR *pcSrcFile = NULL;          // eg: ./0_28x28.y
        HI_CHAR *pcModelName = pcmodelname; // eg: ./inst_mnist_cycle.wk
        HI_U32 u32PicNum = 1;
        HI_S32 s32Ret = HI_SUCCESS;
        SAMPLE_SVP_NNIE_CFG_S stNnieCfg = {0};
        SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
        SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

        /* 设定配置参数 */
        stNnieCfg.pszPic = pcSrcFile;
        stNnieCfg.u32MaxInputNum = u32PicNum; // max input image num in each batch
        stNnieCfg.u32MaxRoiNum = 0;
        stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; // set NNIE core
        s_stCnnSoftwareParam.u32TopN = 5;

        /* SVP 模块初始化 */
        SAMPLE_COMM_SVP_CheckSysInit();

        /* 加载 CNN 模型*/
        SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
        s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName, &s_stCnnModel);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

        /* CNN 参数初始化 */
        /* Cnn 软件参数在 SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit 中被初始化，
        如果用户更改了网络结构，请确保 SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit 函数的输入数据是正确的 */

        SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
        s_stCnnNnieParam.pstModel = &s_stCnnModel.stModel;
        s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg, &s_stCnnNnieParam, &s_stCnnSoftwareParam);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");

        SAMPLE_PRT("model={ type=%x, frmNum=%u, chnNum=%u, w=%u, h=%u, stride=%u }\n",
                   s_stCnnNnieParam.astSegData[0].astSrc[0].enType,
                   s_stCnnNnieParam.astSegData[0].astSrc[0].u32Num,
                   s_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Chn,
                   s_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Width,
                   s_stCnnNnieParam.astSegData[0].astSrc[0].unShape.stWhc.u32Height,
                   s_stCnnNnieParam.astSegData[0].astSrc[0].u32Stride);

        /* 记录任务Buffer */
        s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");

        /* 填充源数据 */
        SAMPLE_SVP_TRACE_INFO("Cnn start!\n");
        stInputDataIdx.u32SegIdx = 0;
        stInputDataIdx.u32NodeIdx = 0;
        s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg, &s_stCnnNnieParam, &stInputDataIdx, IMG);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

        /* NNIE 推理过程(process the 0-th segment) */
        stProcSegIdx.u32SegIdx = 0;
        s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stCnnNnieParam, &stInputDataIdx, &stProcSegIdx, HI_TRUE);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

        /* 结果处理 */
        /* 如果用户更改了网络结构，请确保 SAMPLE_SVP_NNIE_Cnn_GetTopN 函数的输入数据是正确的 */
        s32Ret = SAMPLE_SVP_NNIE_Cnn_GetTopN(&s_stCnnNnieParam, &s_stCnnSoftwareParam);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,SAMPLE_SVP_NNIE_CnnGetTopN failed!\n");

        /* 输出结果 */
        SAMPLE_SVP_TRACE_INFO("Cnn result:\n");
        s32Ret = SAMPLE_SVP_NNIE_Cnn_PrintResult(&(s_stCnnSoftwareParam.stGetTopN),
                                                 s_stCnnSoftwareParam.u32TopN, finalresult);
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,SAMPLE_SVP_NNIE_Cnn_PrintResult failed!\n");
    CNN_FAIL_1:
        /* 移除 TskBuf */
        s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
        SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
                                   "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");
    CNN_FAIL_0:
        SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stCnnNnieParam, &s_stCnnSoftwareParam, &s_stCnnModel);
        SAMPLE_COMM_SVP_CheckSysExit();
    }

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */
