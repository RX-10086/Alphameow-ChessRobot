#ifndef AI_INFER_PROCESS_H
#define AI_INFER_PROCESS_H

#include <stdint.h>
#include "sample_comm_nnie.h"

#if __cplusplus
extern "C"
{
#endif

    // 矩形框坐标定义
    typedef struct RectBox
    {
        int xmin;
        int xmax;
        int ymin;
        int ymax;
    } RectBox;

    HI_S32 SAMPLE_SVP_NNIE_Cnn_ParamInit(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg,
                                         SAMPLE_SVP_NNIE_PARAM_S *pstCnnPara, SAMPLE_SVP_NNIE_CNN_SOFTWARE_PARAM_S *pstCnnSoftWarePara);

    HI_S32 SAMPLE_SVP_NNIE_FillSrcData(SAMPLE_SVP_NNIE_CFG_S *pstNnieCfg,
                                       SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                       SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx,
                                       IVE_IMAGE_S *img);

    HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_PARAM_S *pstNnieParam,
                                   SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx,
                                   SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant);

    HI_S32 SAMPLE_SVP_NNIE_Cnn_PrintResult(SVP_BLOB_S *pstGetTopN, HI_U32 u32TopN, int *final_num);

    void SAMPLE_SVP_NNIE_Cnn(IVE_IMAGE_S *IMG, const char *pcmodelname, int *finalresult);

#ifdef __cplusplus
}
#endif
#endif
