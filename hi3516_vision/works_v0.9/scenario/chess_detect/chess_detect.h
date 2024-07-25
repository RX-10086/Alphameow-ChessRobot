#ifndef CHESS_DETECT_H
#define CHESS_DETECT_H

#include <iostream>
#include "sample_comm_nnie.h"

#if __cplusplus
extern "C"
{
#endif

    HI_S32 CnnLoadChessModel(const char *model_filename);
    void ChessBoardInfoDisplay(int flag_a, int flag_b);
    void ChessBoardInfoLoad(char *loadstr);
    int ReceivedMsgTransToHi3861(char *srcstr, char *dststr, char *chess_eat);

    typedef struct tagIPC_IMAGE
    {
        HI_U64 u64PhyAddr;
        HI_U64 u64VirAddr;
        HI_U32 u32Width;
        HI_U32 u32Height;
    } IPC_IMAGE;

    class chessboard_detect
    {
    public:
        HI_S32 ChessModelDetectLoad(uintptr_t *model);
        HI_S32 ChessModelDetectUnload(uintptr_t model);
        HI_S32 ChessBoardManualCalibrate(VIDEO_FRAME_INFO_S *dstFrm);
        HI_S32 ChessBoardDetect(VIDEO_FRAME_INFO_S *srcFrm, int uart_fdnum, struct timeval *Time);
        HI_S32 ChessBoardNnieProcess(int flag);
    };

#ifdef __cplusplus
}
#endif
#endif