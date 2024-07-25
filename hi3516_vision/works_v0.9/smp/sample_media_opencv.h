/*
 *
 */

#ifndef SAMPLE_MEDIA_OPENCV_H
#define SAMPLE_MEDIA_OPENCV_H

#include "sample_comm.h"
#include "list.h"

#ifdef __cplusplus
#if __cplusplus
extern "C"
{
#endif
#endif /* End of #ifdef __cplusplus */

    HI_VOID ChessDetectProcess(VIDEO_FRAME_INFO_S frm, HI_S32 voLayer, HI_S32 voChn, HI_S32 Uart_Fd_Num, HI_S32 Mipi_Fd_Num, struct timeval *S_time);
    HI_VOID ChessBoardCalibrate(VIDEO_FRAME_INFO_S frm, HI_S32 voLayer, HI_S32 voChn);

    HI_S32 MPI_VPSS_GetChannelFrame(VIDEO_FRAME_INFO_S *frame, HI_S32 MilliSec);
    HI_S32 MPI_ReleaseChannelFrame(VIDEO_FRAME_INFO_S *frame);
    HI_S32 NNIE_WORK_FUNC(int FIRST_RUN_FLAG);

    HI_S32 VIO_VPSS_MIPI_Init(SAMPLE_VI_CONFIG_S *Viconfig, SAMPLE_IVE_SWITCH_S *SwitchS, PIC_SIZE_E *SizeE,
                              SIZE_S *SizeS, SAMPLE_VO_CONFIG_S *VoConfigs);
    HI_S32 VIO_VPSS_MIPI_UnInit(SAMPLE_VI_CONFIG_S *Viconfig, SAMPLE_IVE_SWITCH_S *SwitchS,
                                SAMPLE_VO_CONFIG_S *VoConfig);
    HI_S32 Serial_Port_Key_Init(void);

    HI_S32 NNIE_READY_OUTPUT(void);
    HI_S32 NNIE_CHANGE(void);
    HI_VOID PauseDoUnloadChessModel(HI_VOID);

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif /* End of #ifdef __cplusplus */

#endif /* End of #ifndef __SAMPLE_MEDIA_OPENCV_H__ */