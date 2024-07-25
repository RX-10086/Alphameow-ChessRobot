#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <ctime>
#include <unistd.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>

#include "hi_mipi_tx.h"
#include "sdk.h"
#include "sample_comm.h"
#include "hisignalling.h"
#include "resnet_infer_process.h"
#include "chess_detect.h"
#include "vgs_img.h"
#include "base_interface.h"
#include "posix_help.h"
#include "sample_media_ai.h"
#include "sample_media_opencv.h"
#include "acodec.h"
#include "audio_aac_adp.h"
#include "audio_dl_adp.h"

using namespace std;

static AicMediaInfo g_aicChessMediaInfo = {0};
HI_S32 NNIE_READY_FLAG = HI_FALSE;

static int mipi_fd = -1;
char message_send[40] = {0};

chessboard_detect opencv;

HI_S32 message_judgment(char *str, const char *base_str)
{
    HI_S32 s32Ret = HI_SUCCESS;
    for (int i = 0; i < 6; i++)
    {
        if (str[i] != base_str[i])
            return HI_FAILURE;
    }
    return s32Ret;
}

HI_VOID ChessDetectProcess(VIDEO_FRAME_INFO_S frm, HI_S32 voLayer, HI_S32 voChn, HI_S32 Uart_Fd_Num, HI_S32 Mipi_Fd_Num, struct timeval *S_time)
{
    int ret;
    VIDEO_FRAME_INFO_S calFrm;
    ret = MppFrmResize(&frm, &calFrm, 1920, 1080); // 1920: FRM_WIDTH, 1080: FRM_HEIGHT
    ret = opencv.ChessBoardDetect(&calFrm, Uart_Fd_Num, S_time);
    SAMPLE_CHECK_EXPR_GOTO(ret == 127, NNIE_WORK, "Not Error, We are Ready for NNIE Work!, ret=%#x\n", ret);
    SAMPLE_CHECK_EXPR_GOTO(ret < 0, CHESS_RELEASE, "ChessBoardDetect err, ret=%#x\n", ret);
    ret = HI_MPI_VO_SendFrame(voLayer, voChn, &frm, 0); // frm
    SAMPLE_CHECK_EXPR_GOTO(ret != HI_SUCCESS, CHESS_RELEASE,
                           "HI_MPI_VO_SendFrame err, ret=%#x\n", ret);
    MppFrmDestroy(&calFrm);

CHESS_RELEASE:
    ret = HI_MPI_VPSS_ReleaseChnFrame(g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0, &frm);
    if (ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                   ret, g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0);
    }
    return;
NNIE_WORK:
    ret = HI_MPI_VO_SendFrame(voLayer, voChn, &frm, 0); // frm
    SAMPLE_CHECK_EXPR_GOTO(ret != HI_SUCCESS, CHESS_RELEASE,
                           "HI_MPI_VO_SendFrame err, ret=%#x\n", ret);
    MppFrmDestroy(&calFrm);
    ret = HI_MPI_VPSS_ReleaseChnFrame(g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0, &frm);
    if (ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                   ret, g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0);
    }
    NNIE_READY_FLAG = HI_TRUE;
}

HI_VOID ChessBoardCalibrate(VIDEO_FRAME_INFO_S frm, HI_S32 voLayer, HI_S32 voChn)
{
    int ret;
    VIDEO_FRAME_INFO_S calFrm;
    ret = MppFrmResize(&frm, &calFrm, 1920, 1080); // 1920: FRM_WIDTH, 1080: FRM_HEIGHT
    ret = opencv.ChessBoardManualCalibrate(&calFrm);
    SAMPLE_CHECK_EXPR_GOTO(ret < 0, CALIB_RELEASE, "ChessBoardCalib err, ret=%#x\n", ret);
    ret = HI_MPI_VO_SendFrame(voLayer, voChn, &calFrm, 0); // frm
    SAMPLE_CHECK_EXPR_GOTO(ret != HI_SUCCESS, CALIB_RELEASE,
                           "HI_MPI_VO_SendFrame err, ret=%#x\n", ret);
    MppFrmDestroy(&calFrm);

CALIB_RELEASE:
    ret = HI_MPI_VPSS_ReleaseChnFrame(g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0, &frm);
    if (ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                   ret, g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0);
    }
    return;
}

HI_S32 NNIE_WORK_FUNC(int FIRST_RUN_FLAG)
{
    HI_S32 s32Ret;
    s32Ret = opencv.ChessBoardNnieProcess(FIRST_RUN_FLAG);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("NNIE Work Process Error, err code = %#x\n", s32Ret);
        return s32Ret;
    }
    return s32Ret;
}

HI_S32 MPI_VPSS_GetChannelFrame(VIDEO_FRAME_INFO_S *frame, HI_S32 MilliSec)
{
    HI_S32 ret;
    ret = HI_MPI_VPSS_GetChnFrame(g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0, frame, MilliSec);
    return ret;
}

HI_S32 MPI_ReleaseChannelFrame(VIDEO_FRAME_INFO_S *frame)
{
    HI_S32 ret;
    SAMPLE_PRT("HI_MPI_VPSS_GetChnFrame FAIL, err=%#x, grp=%d, chn=%d\n",
               ret, g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0);
    ret = HI_MPI_VPSS_ReleaseChnFrame(g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0, frame);
    if (ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Error(%#x),HI_MPI_VPSS_ReleaseChnFrame failed,Grp(%d) chn(%d)!\n",
                   ret, g_aicChessMediaInfo.vpssGrp, g_aicChessMediaInfo.vpssChn0);
    }
    return ret;
}

/*
 * 将sensor采集到数据显示到液晶屏上
 * 视频输入->视频处理子系统->视频输出->显示屏
 * VI->VPSS->VO->MIPI
 */
HI_S32 VIO_VPSS_MIPI_Init(SAMPLE_VI_CONFIG_S *Viconfig, SAMPLE_IVE_SWITCH_S *SwitchS, PIC_SIZE_E *SizeE,
                          SIZE_S *SizeS, SAMPLE_VO_CONFIG_S *VoConfigs)
{
    HI_S32 s32Ret = HI_SUCCESS;

    SIZE_S astSize[VPSS_CHN_NUM];
    PIC_SIZE_E aenSize[VPSS_CHN_NUM];
    VI_CHN_ATTR_S stViChnAttr;
    SAMPLE_RC_E enRcMode = SAMPLE_RC_CBR;
    PAYLOAD_TYPE_E enStreamType = PT_H264;
    HI_BOOL bRcnRefShareBuf = HI_FALSE;
    VENC_GOP_ATTR_S stGopAttr;
    VI_DEV ViDev0 = 0;
    VI_PIPE ViPipe0 = 0;
    VI_CHN ViChn = 0;
    HI_S32 s32ViCnt = 1;
    HI_S32 s32WorkSnsId = 0;
    VPSS_GRP VpssGrp = 0;
    VENC_CHN VeH264Chn = 0;
    WDR_MODE_E enWDRMode = WDR_MODE_NONE;
    DYNAMIC_RANGE_E enDynamicRange = DYNAMIC_RANGE_SDR8;
    PIXEL_FORMAT_E enPixFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    VIDEO_FORMAT_E enVideoFormat = VIDEO_FORMAT_LINEAR;
    COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
    VI_VPSS_MODE_E enMastPipeMode = VI_ONLINE_VPSS_OFFLINE;

    /* 检查是否为空指针 */
    CHECK_NULL_PTR(Viconfig);
    CHECK_NULL_PTR(SwitchS);
    CHECK_NULL_PTR(SizeE);
    (HI_VOID) memset_s(Viconfig, sizeof(SAMPLE_VI_CONFIG_S), 0, sizeof(SAMPLE_VI_CONFIG_S));

    /* 得到 Sensor 型号*/
    SAMPLE_COMM_VI_GetSensorInfo(Viconfig);
    Viconfig->s32WorkingViNum = s32ViCnt;

    Viconfig->as32WorkingViId[0] = 0;
    Viconfig->astViInfo[0].stSnsInfo.MipiDev = SAMPLE_COMM_VI_GetComboDevBySensor(Viconfig->astViInfo[0].stSnsInfo.enSnsType, 0);
    Viconfig->astViInfo[0].stSnsInfo.s32BusId = 0;

    Viconfig->astViInfo[0].stDevInfo.ViDev = ViDev0;
    Viconfig->astViInfo[0].stDevInfo.enWDRMode = enWDRMode;

    Viconfig->astViInfo[0].stPipeInfo.enMastPipeMode = enMastPipeMode;
    Viconfig->astViInfo[0].stPipeInfo.aPipe[0] = ViPipe0;
    Viconfig->astViInfo[0].stPipeInfo.aPipe[1] = -1; /* pipe index: 1 */
    Viconfig->astViInfo[0].stPipeInfo.aPipe[2] = -1; /* pipe index: 2 */
    Viconfig->astViInfo[0].stPipeInfo.aPipe[3] = -1; /* pipe index: 3 */

    Viconfig->astViInfo[0].stChnInfo.ViChn = ViChn;
    Viconfig->astViInfo[0].stChnInfo.enPixFormat = enPixFormat;
    Viconfig->astViInfo[0].stChnInfo.enDynamicRange = enDynamicRange;
    Viconfig->astViInfo[0].stChnInfo.enVideoFormat = enVideoFormat;
    Viconfig->astViInfo[0].stChnInfo.enCompressMode = enCompressMode;

    /* 通过 Sensor 型号得到图像大小*/
    s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(Viconfig->astViInfo[s32WorkSnsId].stSnsInfo.enSnsType, &aenSize[0]);
    aenSize[1] = *SizeE;

    /* step  1: 初始化视频缓存池 */
    HI_U64 u64BlkSize;
    VB_CONFIG_S stVbConf;
    CHECK_NULL_PTR(aenSize);
    CHECK_NULL_PTR(astSize);
    (HI_VOID) memset_s(&stVbConf, sizeof(VB_CONFIG_S), 0, sizeof(VB_CONFIG_S));
    stVbConf.u32MaxPoolCnt = 128; /* max pool count: 128 */

    for (int i = 0; i < VPSS_CHN_NUM; i++)
    {
        s32Ret = SAMPLE_COMM_SYS_GetPicSize(aenSize[i], &astSize[i]);
        u64BlkSize = COMMON_GetPicBufferSize(astSize[i].u32Width, astSize[i].u32Height, SAMPLE_PIXEL_FORMAT,
                                             DATA_BITWIDTH_8, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
        /* comm video buffer */
        stVbConf.astCommPool[i].u64BlkSize = u64BlkSize;
        stVbConf.astCommPool[i].u32BlkCnt = 16; /* vb block size: 16 */
    }
    /* 初始化视频缓存池和 MPI */
    s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);

    /* step 2: 启动 VI */
    s32Ret = SAMPLE_COMM_VI_SetParam(Viconfig);
    s32Ret = SAMPLE_COMM_VI_StartVi(Viconfig);

    /* step 3: 启动 VPSS */
    // s32Ret = SAMPLE_COMM_IVE_StartVpss(astSize, VPSS_CHN_NUM);
    VPSS_CHN_ATTR_S astVpssChnAttr[VPSS_MAX_CHN_NUM];
    VPSS_GRP_ATTR_S stVpssGrpAttr = {0};
    HI_BOOL abChnEnable[VPSS_MAX_CHN_NUM] = {HI_FALSE, HI_FALSE, HI_FALSE, HI_FALSE};

    CHECK_NULL_PTR(astSize);
    stVpssGrpAttr.u32MaxW = IMG_2M_WIDTH;
    stVpssGrpAttr.u32MaxH = IMG_2M_HEIGHT;
    stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssGrpAttr.enDynamicRange = DYNAMIC_RANGE_SDR8;
    stVpssGrpAttr.enPixelFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    stVpssGrpAttr.bNrEn = HI_FALSE;

    for (int i = 0; i < VPSS_CHN_NUM; i++)
    {
        abChnEnable[i] = HI_TRUE;
    }

    for (int i = 0; i < VPSS_MAX_CHN_NUM; i++)
    {
        astVpssChnAttr[i].u32Width = astSize[i].u32Width;
        astVpssChnAttr[i].u32Height = astSize[i].u32Height;
        astVpssChnAttr[i].enChnMode = VPSS_CHN_MODE_USER;
        astVpssChnAttr[i].enCompressMode = COMPRESS_MODE_NONE;
        astVpssChnAttr[i].enDynamicRange = DYNAMIC_RANGE_SDR8;
        astVpssChnAttr[i].enVideoFormat = VIDEO_FORMAT_LINEAR;
        astVpssChnAttr[i].enPixelFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
        astVpssChnAttr[i].stFrameRate.s32SrcFrameRate = -1;
        astVpssChnAttr[i].stFrameRate.s32DstFrameRate = -1;
        astVpssChnAttr[i].u32Depth = 1;
        astVpssChnAttr[i].bMirror = HI_FALSE;
        astVpssChnAttr[i].bFlip = HI_FALSE;
        astVpssChnAttr[i].stAspectRatio.enMode = ASPECT_RATIO_NONE;
    }

    s32Ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr, &astVpssChnAttr[0]);

    /* step 4: VPSS 与 VI 连接 */
    s32Ret = SAMPLE_COMM_VI_Bind_VPSS(ViPipe0, ViChn, VpssGrp);

    // Set vi frame
    s32Ret = HI_MPI_VI_GetChnAttr(ViPipe0, ViChn, &stViChnAttr);
    s32Ret = HI_MPI_VI_SetChnAttr(ViPipe0, ViChn, &stViChnAttr);

    s32Ret = SAMPLE_COMM_SYS_GetPicSize(*SizeE, SizeS);

    s32Ret = SAMPLE_VO_CONFIG_MIPI(&mipi_fd);

    VI_LDC_ATTR_S LDC_cfg;
    LDC_cfg.bEnable = HI_TRUE;
    LDC_cfg.stAttr.bAspect = HI_TRUE;
    LDC_cfg.stAttr.s32XRatio = 100; /* x ratio: 100 */
    LDC_cfg.stAttr.s32YRatio = 100; /* y ratio: 100 */
    LDC_cfg.stAttr.s32XYRatio = 100;
    LDC_cfg.stAttr.s32CenterXOffset = 0;
    LDC_cfg.stAttr.s32CenterYOffset = 0;
    LDC_cfg.stAttr.s32DistortionRatio = 95;
    s32Ret = HI_MPI_VI_SetChnLDCAttr(ViPipe0, ViChn, &LDC_cfg);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_PRT("Set LDC Config failed!\n");
        return s32Ret;
    }

    /* step 5: 启动 Vo */
    VO_PUB_ATTR_S VoPubAttr;
    if (SwitchS->bVo == HI_TRUE)
    {
        // s32Ret = SAMPLE_COMM_IVE_StartVo();
        SAMPLE_COMM_VO_GetDefConfig(VoConfigs);
        VoConfigs->enDstDynamicRange = DYNAMIC_RANGE_SDR8;
        VoConfigs->enVoIntfType = VO_INTF_MIPI;
        VoConfigs->enIntfSync = VO_OUTPUT_USER;
        VoConfigs->enPicSize = *SizeE;
        /* 启动 Mipi 屏幕*/
        SampleCommVoStartMipi(VoConfigs, &VoPubAttr);
    }
    return s32Ret;
}

/*
 * 将上面的VI->VPSS->VO->MIPI过程去初始化
 */
HI_S32 VIO_VPSS_MIPI_UnInit(SAMPLE_VI_CONFIG_S *Viconfig, SAMPLE_IVE_SWITCH_S *SwitchS, SAMPLE_VO_CONFIG_S *VoConfig)
{
    HI_S32 s32Ret = HI_SUCCESS;
    if (SwitchS->bVo == HI_TRUE)
    {
        SAMPLE_VO_DISABLE_MIPITx(mipi_fd);
        SampleCloseMipiTxFd(mipi_fd, VoConfig);
        system("echo 0 > /sys/class/gpio/gpio55/value");
    }

    SAMPLE_COMM_VI_UnBind_VPSS(Viconfig->astViInfo[0].stPipeInfo.aPipe[0],
                               Viconfig->astViInfo[0].stChnInfo.ViChn, 0);
    // SAMPLE_COMM_IVE_StopVpss(VPSS_CHN_NUM);
    VPSS_GRP VpssGrp = 0;
    HI_BOOL abChnEnable[VPSS_MAX_CHN_NUM] = {HI_FALSE, HI_FALSE, HI_FALSE, HI_FALSE};

    for (int i = 0; i < VPSS_CHN_NUM; i++)
        abChnEnable[i] = HI_TRUE;

    (HI_VOID) SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);
    SAMPLE_COMM_VI_StopVi(Viconfig);

    SAMPLE_COMM_SYS_Exit();

    (HI_VOID) memset_s(Viconfig, sizeof(SAMPLE_VI_CONFIG_S), 0, sizeof(SAMPLE_VI_CONFIG_S));
    return s32Ret;
}

// 串口初始化函数
HI_S32 Serial_Port_Key_Init(void)
{
    // 串口初始化
    int uart_fdnum = UartOpenInit();

    // 按键初始化
    InitGpio1();
    InitGpio2();
    return uart_fdnum;
}

HI_S32 NNIE_READY_OUTPUT(void)
{
    return NNIE_READY_FLAG;
}

HI_S32 NNIE_CHANGE(void)
{
    NNIE_READY_FLAG = -NNIE_READY_FLAG;
    return HI_SUCCESS;
}