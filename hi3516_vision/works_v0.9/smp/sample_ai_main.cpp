/**
 * @author Weijia Yan, Mr_Tsura@126.com
 * @brief 象棋机器人, AI视觉部分
 * @copyright Copyright (c) 2024 大傻猫启动
 */

// 该cpp文件为主函数文件
#include <iostream>
#include <pthread.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <csignal>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include "unistd.h"
#include "sdk.h"
#include "sample_comm.h"
#include "hi_comm_svp.h"
#include "sample_comm_ive.h"
#include "sample_media_ai.h"
#include "sample_media_opencv.h"
#include "chess_detect.h"
#include "audio_play.h"
#include "hisignalling.h"
#include "acodec.h"
#include "audio_aac_adp.h"
#include "audio_dl_adp.h"

using namespace std;

static HI_BOOL Main_Stop_Signal = HI_FALSE;
static pthread_t Main_ThreadT = 0;
static HI_S32 Serial_fdnum = -1;
static HI_S32 mipi_fd_num = 0;

struct timeval StartTime, EndTime;

HI_U8 status_ch = '0';
int socket_fd_cloud, socket_fd_hi3861;
struct sockaddr_in servaddr_cloud, servaddr_hi3861;
static int FIRST_TIME_SCAN = HI_TRUE;

char FENSendData[] = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1";
char CloudGetData[60];
char Hi3861SendData[20] = "%7, 6, 8, 4"; // 发送给Hi3861的指令
char EatenChess[] = "%2, 5, 0, 0";       // 吃子的指令
char Hi3861GetData[20];                  // 从Hi3861得到的Ack信号数据

// UDP网络连接初始化
static int UDP_Init(int port, const char *ipstring, struct sockaddr_in *servaddr)
{
    int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd < 0)
        perror("socket error");
    bzero(servaddr, sizeof(struct sockaddr_in));
    servaddr->sin_family = AF_INET;
    servaddr->sin_port = htons(port);
    if (inet_pton(AF_INET, ipstring, &(servaddr->sin_addr)) <= 0) // 云端服务器IP
        perror("inet_pton error");
    usleep(100);
    return socket_fd;
}

// 创建线程运行棋盘检测推理计算
static HI_VOID *Chess_Detect_Task(HI_VOID *arg)
{
    int ret;
    VIDEO_FRAME_INFO_S frm;
    HI_S32 s32MilliSec = 2000;
    HI_S32 voLayer = 0;
    HI_S32 voChn = 0;

    while (HI_FALSE == Main_Stop_Signal)
    {
        ret = MPI_VPSS_GetChannelFrame(&frm, s32MilliSec);
        if (ret != HI_SUCCESS)
        {
            ret = MPI_ReleaseChannelFrame(&frm);
            continue;
        }
        ChessDetectProcess(frm, voLayer, voChn, Serial_fdnum, mipi_fd_num, &StartTime);
        if (NNIE_READY_OUTPUT() == HI_TRUE)
        {
            Main_Stop_Signal = HI_TRUE;
            return HI_NULL;
        }
    }
    return HI_NULL;
}
static HI_VOID *Chess_Calib_Task(HI_VOID *arg)
{
    int ret;
    VIDEO_FRAME_INFO_S frm;
    HI_S32 s32MilliSec = 2000;
    HI_S32 voLayer = 0;
    HI_S32 voChn = 0;

    while (HI_FALSE == Main_Stop_Signal)
    {
        ret = MPI_VPSS_GetChannelFrame(&frm, s32MilliSec);
        if (ret != HI_SUCCESS)
        {
            ret = MPI_ReleaseChannelFrame(&frm);
            continue;
        }
        ChessBoardCalibrate(frm, voLayer, voChn);
    }
    return HI_NULL;
}
// 程序功能
static void SAMPLE_SVP_Usage(const char *pchPrgName)
{
    printf("Usage : %s <index> \n", pchPrgName);
    printf("index:\n");
    printf("\t 0) Chess Detect Runt Mode.\n");
    printf("\t 1) Chess Detect Run Mode, Only Support Normal Opening!\n");
    printf("\t 2) Chess Board Manual Calibrate Mode.\n");
}
// 主函数
int main(int argc, char *argv[])
{
    HI_S32 s32Ret = HI_SUCCESS;
    socket_fd_cloud = UDP_Init(9666, "114.116.210.63", &servaddr_cloud); // 连接服务器
    socket_fd_hi3861 = UDP_Init(888, "192.168.164.4", &servaddr_hi3861); // 连接Hi3861
    if (argc < 2 || argc > 2)
    { /* only support 2 parameters */
        SAMPLE_SVP_Usage(argv[0]);
        close(socket_fd_cloud);
        close(socket_fd_hi3861);
        s32Ret = HI_FAILURE;
        return s32Ret;
    }
    sdk_init();
    SIZE_S stSize;
    PIC_SIZE_E enSize = PIC_1080P;
    SAMPLE_IVE_SWITCH_S ChessSwitch;
    ChessSwitch.bVenc = HI_FALSE;
    ChessSwitch.bVo = HI_TRUE;
    SAMPLE_VI_CONFIG_S ChessViConfig;
    SAMPLE_VO_CONFIG_S ChessVoConfig;

    switch (*argv[1])
    {
    case '0':
    case '1':
        s32Ret = CnnLoadChessModel("./test_inst.wk");
    START_POINT:
        SAMPLE_PRT("Chess Robot Vision-Part Start!\n");
        Serial_fdnum = Serial_Port_Key_Init();
        if (FIRST_TIME_SCAN == HI_TRUE)
        {
            SAMPLE_PRT("Chess Board First Time Detecting...\n");
        }
        s32Ret = VIO_VPSS_MIPI_Init(&ChessViConfig, &ChessSwitch, &enSize, &stSize, &ChessVoConfig);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("VIO & MIPI Init Failed. s32Ret=%#x\n", s32Ret);
            return s32Ret;
        }
        // 创建工作线程运行ai
        s32Ret = pthread_create(&Main_ThreadT, NULL, Chess_Detect_Task, NULL);
        // 等待一个线程结束，线程间同步的操作
        pthread_join(Main_ThreadT, nullptr);
        Main_ThreadT = 0;
        s32Ret = VIO_VPSS_MIPI_UnInit(&ChessViConfig, &ChessSwitch, &ChessVoConfig);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("VIO & MIPI UnInit Failed. s32Ret=%#x\n", s32Ret);
            return s32Ret;
        }
        if (NNIE_READY_OUTPUT() == HI_TRUE)
        {
            NNIE_WORK_FUNC(FIRST_TIME_SCAN);
            gettimeofday(&EndTime, NULL);
            double timeuse = (EndTime.tv_sec - StartTime.tv_sec) + (EndTime.tv_usec - StartTime.tv_usec) / 1000000.0;
            SAMPLE_PRT("Vision Run Time = %lf\n", timeuse);
            int ch = (*argv[1]) - '0';
            long int waitnum = 0;
            ChessBoardInfoDisplay(ch, FIRST_TIME_SCAN);
            ChessBoardInfoLoad(FENSendData);
            Main_Stop_Signal = HI_FALSE;
            NNIE_CHANGE();
            if (FIRST_TIME_SCAN == HI_TRUE)
            {
                SAMPLE_PRT("Who Play First, Player or AI(P/A)?\n");
                status_ch = getchar();
                if (status_ch == 'P' || status_ch == 'p')
                {
                    SAMPLE_PRT("Player Play!\n");
                    FIRST_TIME_SCAN = HI_FALSE;
                    goto START_POINT;
                }
                else if (status_ch == 'A' || status_ch == 'a')
                {
                    SAMPLE_PRT("AI Play!\n");
                    // AudioPlay("./aac_file/13.aac");
                SEND_A:
                    waitnum = 0;
                    sendto(socket_fd_cloud, FENSendData, strlen(FENSendData), 0, (struct sockaddr *)&servaddr_cloud, sizeof(servaddr_cloud));
                    while (!recvfrom(socket_fd_cloud, CloudGetData, sizeof(CloudGetData), 0, NULL, NULL)) // 等待云端指令
                    {
                        // waitnum++;
                        // if (waitnum >= 500000)
                        //     goto SEND_A;
                    }
                    int eat = ReceivedMsgTransToHi3861(CloudGetData, Hi3861SendData, EatenChess);
                    if (eat == HI_TRUE)
                    {
                    SEND_B:
                        waitnum = 0;
                        sendto(socket_fd_hi3861, EatenChess, strlen(EatenChess), 0, (struct sockaddr *)&servaddr_hi3861, sizeof(servaddr_hi3861));
                        while (!recvfrom(socket_fd_hi3861, Hi3861GetData, sizeof(Hi3861GetData), 0, NULL, NULL)) // 等待Hi3861发送Ack
                        {
                            // waitnum++;
                            // if (waitnum >= 500000)
                            //     goto SEND_B;
                        }
                        SAMPLE_PRT("Ack Received!\n");
                    }
                SEND_C:
                    waitnum = 0;
                    memset(Hi3861GetData, 0, 20 * sizeof(char));
                    sendto(socket_fd_hi3861, Hi3861SendData, strlen(Hi3861SendData), 0, (struct sockaddr *)&servaddr_hi3861, sizeof(servaddr_hi3861));
                    while (!recvfrom(socket_fd_hi3861, Hi3861GetData, sizeof(Hi3861GetData), 0, NULL, NULL))
                    {
                        // waitnum++;
                        // if (waitnum >= 500000)
                        //     goto SEND_C;
                    }
                    SAMPLE_PRT("Ack Received!\n");
                    memset(CloudGetData, 0, 45 * sizeof(char));
                    memset(Hi3861SendData, 0, 20 * sizeof(char));
                }
                else
                {
                    SAMPLE_PRT("Wrong Command! Please Reboot Vision Part.\n");
                    return s32Ret;
                }
                FIRST_TIME_SCAN = HI_FALSE;
            }
            else
            {
                SAMPLE_PRT("AI Play!\n");
            SEND_D:
                waitnum = 0;
                sendto(socket_fd_cloud, FENSendData, strlen(FENSendData), 0, (struct sockaddr *)&servaddr_cloud, sizeof(servaddr_cloud));
                while (!recvfrom(socket_fd_cloud, CloudGetData, sizeof(CloudGetData), 0, NULL, NULL))
                {
                    // waitnum++;
                    // if (waitnum >= 500000)
                    //     goto SEND_D;
                }
                /*
                 * “吃子”被分为两步：
                 * 第一步：将被吃的棋子移动到棋盘外
                 * 第二步：将吃的棋子移动到原来被吃棋子的位置
                 */
                int eat = ReceivedMsgTransToHi3861(CloudGetData, Hi3861SendData, EatenChess);
                if (eat == HI_TRUE)
                {
                SEND_E:
                    waitnum = 0;
                    sendto(socket_fd_hi3861, EatenChess, strlen(EatenChess), 0, (struct sockaddr *)&servaddr_hi3861, sizeof(servaddr_hi3861));
                    while (!recvfrom(socket_fd_hi3861, Hi3861GetData, sizeof(Hi3861GetData), 0, NULL, NULL))
                    {
                        // waitnum++;
                        // if (waitnum >= 500000)
                        //     goto SEND_E;
                    }
                    SAMPLE_PRT("Ack Received!\n");
                }
                memset(Hi3861GetData, 0, 20 * sizeof(char));
            SEND_F:
                waitnum = 0;
                sendto(socket_fd_hi3861, Hi3861SendData, strlen(Hi3861SendData), 0, (struct sockaddr *)&servaddr_hi3861, sizeof(servaddr_hi3861));
                while (!recvfrom(socket_fd_hi3861, Hi3861GetData, sizeof(Hi3861GetData), 0, NULL, NULL))
                {
                    // waitnum++;
                    // if (waitnum >= 500000)
                    //     goto SEND_F;
                }
                SAMPLE_PRT("Ack Received!\n");
                memset(CloudGetData, 0, 45 * sizeof(char));
                memset(Hi3861SendData, 0, 20 * sizeof(char));
            }
        }
        goto START_POINT;

        sdk_exit();
        SAMPLE_PRT("\nsdk exit success\n");
        break;
    case '2':
        SAMPLE_PRT("Vision Calibrate Mode!\n");
        s32Ret = VIO_VPSS_MIPI_Init(&ChessViConfig, &ChessSwitch, &enSize, &stSize, &ChessVoConfig);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("VIO & MIPI Init Failed. s32Ret=%#x\n", s32Ret);
            return s32Ret;
        }
        // 创建工作线程运行校准
        s32Ret = pthread_create(&Main_ThreadT, NULL, Chess_Calib_Task, NULL);
        PAUSE();
        Main_Stop_Signal = HI_TRUE;
        pthread_join(Main_ThreadT, nullptr);
        Main_ThreadT = 0;
        s32Ret = VIO_VPSS_MIPI_UnInit(&ChessViConfig, &ChessSwitch, &ChessVoConfig);
        if (s32Ret != HI_SUCCESS)
        {
            SAMPLE_PRT("VIO & MIPI UnInit Failed. s32Ret=%#x\n", s32Ret);
            return s32Ret;
        }
        sdk_exit();
        break;
    }
    return s32Ret;
}
