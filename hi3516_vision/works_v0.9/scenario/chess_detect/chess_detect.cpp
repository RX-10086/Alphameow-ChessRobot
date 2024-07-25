// 基于OpenCV与NNIE实现的棋盘检测功能

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <vector>
#include <fstream>
#include <cmath>
#include "chess_detect.h"

#include "unistd.h"
#include "sample_comm_nnie.h"
#include "sample_comm_ive.h"
#include "sample_media_ai.h"
#include "vgs_img.h"
#include "ive_img.h"
#include "misc_util.h"
#include "frame_convert.h"
#include "hisignalling.h"
#include "resnet_infer_process.h"
#include "nnie_work.h"

using namespace std;
using namespace cv;

static IVE_SRC_IMAGE_S pstSrc;
static IVE_DST_IMAGE_S pstDst;
static IVE_CSC_CTRL_S stCscCtrl;

vector<Vec3f> circles; // 存储检测到的圆数据

RectBox TESTBOARDboxs[10] = {0};
RectBox TESTBOARDboxs_B[10] = {0};
RectBox TESTBOARDboxs_C[10] = {0};

// OpenCV 镜头畸变矫正参数
Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1593.618127665734, 0, 890.8677125039185, // 焦距，cx
                    0, 1515.303144473761, 621.013552467895,                            // 焦距，cy
                    0, 0, 1);
Mat distCoeffs = (cv::Mat_<double>(1, 5) << -0.4028516188319258, 0.229977103760291,
                  -0.003064013772764661, -0.001674729803788447, -0.1479444165471021);

const char chess_name[10][10] = {"bing", "jiang", "ju", "ma", "pao", "shi", "shuai", "xiang", "xiang_b", "zu"};
const char chess_color[2][6] = {"Red", "Black"};
const char chess_abbre[10][2] = {"P", "k", "R", "N", "C", "A", "K", "B", "b", "p"};
const char chess_abbre_B[10][2] = {"p", "k", "r", "n", "c", "a", "k", "b", "b", "p"};

typedef struct CHESS_DIS
{
    char name[10] = "None";
    char color_name[6] = "None";
};
typedef struct CHESS_DIS_ABBRE
{
    int actual_number = -127;
    int color_number = -127;
};
int chessnum = 0;
struct CHESS_DIS display[9][10];
struct CHESS_DIS_ABBRE display_abbre[9][10];
struct CHESS_DIS_ABBRE display_new[9][10];

// 设定识别红色棋子的HSV阈值范围
// 注意：这些值需要根据实际棋子颜色进行调整
int lowerH = 0;   // Hue的最小值
int lowerS = 43;  // Saturation的最小值
int lowerV = 46;  // Value的最小值
int upperH = 10;  // Hue的最大值
int upperS = 255; // Saturation的最大值
int upperV = 255; // Value的最大值

int lowerH_K = 0;   // Hue的最小值
int lowerS_K = 0;   // Saturation的最小值
int lowerV_K = 0;   // Value的最小值
int upperH_K = 180; // Hue的最大值
int upperS_K = 255; // Saturation的最大值
int upperV_K = 46;  // Value的最大值

#define COLOR_NONE (0)
#define COLOR_RED (1)
#define COLOR_BLACK (2)

#define MODEL_FILE_CHESS "./test_inst.wk" // 模型的路径

typedef struct CHESS
{
    int num;        // 棋子的标号，无实际意义
    float center_x; // 棋子圆心在图片上的绝对坐标
    float center_y;
    float act_x; // 棋子圆心在图片上的相对坐标
    float act_y;
    int cols; // 棋子在棋盘的几行几列
    int rows;
    float radius_s = 0;    // 识别棋子的圆形轮廓半径大小
    int color = -127;      // 棋子的颜色，1为红色，2为黑色
    int actual_num = -127; // 表示棋子的实际种类
};

HI_S32 num = 0;
int yellowdetectnum = 0;
char str[60] = {0};
char str_name[60] = {0};
char str_chess[60] = {0};
struct CHESS CHESS_CROPED[36];

int datasetnum = 0;
int first_time_chessboard_cal = HI_TRUE;

// 计算图片亮度
static float lightMean(Mat &img)
{
    Scalar scalar = mean(img);
    float imgChannel1 = scalar.val[0];
    float imgChannel2 = scalar.val[1];
    float imgChannel3 = scalar.val[2];
    float imgLight = (imgChannel1 + imgChannel2 + imgChannel3) / 3;
    return imgLight;
}

// 调节图片亮度与对比度
static Mat Light_Contrast_Change(Mat &Unprepared_Src, float alpha, float beta) // 设置参数
{
    int ret = HI_SUCCESS;
    // 获取图像的高和宽
    int height = Unprepared_Src.rows;
    int width = Unprepared_Src.cols;
    Mat Pre_Dst = Mat::zeros(Unprepared_Src.size(), Unprepared_Src.type());

    // 遍历图像像素点
    // 分别对每一个通道的像素值操作
    for (int y = 0; y < Pre_Dst.rows; y++)
    {
        for (int x = 0; x < Pre_Dst.cols; x++)
        {
            for (int c = 0; c < 3; c++)
            {
                Pre_Dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha * (Unprepared_Src.at<Vec3b>(y, x)[c]) + beta);
            }
        }
    }
    return Pre_Dst;
}

HI_S32 CnnLoadChessModel(const char *model_filename)
{
    int ret = HI_SUCCESS;
    ret = Chess_Cnn_Load_Model(model_filename);
    return ret;
}

HI_S32 CnnChessClassifyCal(IVE_IMAGE_S *srcImage, int *result) // int x_min, int x_max, int y_min, int y_max
{
    SAMPLE_PRT("begin Chess Classify Cal\n");
    int ret = HI_SUCCESS;
    ret = Chess_Cnn_Work_Func(srcImage, result);
    return ret;
}

static HI_VOID IveImageParamCfg(IVE_SRC_IMAGE_S *pstSrc, IVE_DST_IMAGE_S *pstDst,
                                VIDEO_FRAME_INFO_S *srcFrame)
{
    pstSrc->enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc->au64VirAddr[0] = srcFrame->stVFrame.u64VirAddr[0];
    pstSrc->au64VirAddr[1] = srcFrame->stVFrame.u64VirAddr[1];
    pstSrc->au64VirAddr[2] = srcFrame->stVFrame.u64VirAddr[2]; // 2: Image data virtual address

    pstSrc->au64PhyAddr[0] = srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc->au64PhyAddr[1] = srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc->au64PhyAddr[2] = srcFrame->stVFrame.u64PhyAddr[2]; // 2: Image data physical address

    pstSrc->au32Stride[0] = srcFrame->stVFrame.u32Stride[0];
    pstSrc->au32Stride[1] = srcFrame->stVFrame.u32Stride[1];
    pstSrc->au32Stride[2] = srcFrame->stVFrame.u32Stride[2]; // 2: Image data span

    pstSrc->u32Width = srcFrame->stVFrame.u32Width;
    pstSrc->u32Height = srcFrame->stVFrame.u32Height;

    pstDst->enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDst->u32Width = pstSrc->u32Width;
    pstDst->u32Height = pstSrc->u32Height;
    pstDst->au32Stride[0] = pstSrc->au32Stride[0];
    pstDst->au32Stride[1] = 0;
    pstDst->au32Stride[2] = 0; // 2: Image data span
}

// 将YUV视频帧转为RGB格式的IVE图片，并传入中间变量
static HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB; // IVE_CSC_MODE_VIDEO_BT601_YUV2RGB
    IveImageParamCfg(&pstSrc, &pstDst, srcFrame);

    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstDst.au64PhyAddr[0], (void **)&pstDst.au64VirAddr[0],
                                        "User", HI_NULL, pstDst.u32Height * pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        SAMPLE_PRT("HI_MPI_SYS_MmzFree err\n");
        return s32Ret;
    }

    s32Ret = HI_MPI_SYS_MmzFlushCache(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0],
                                      pstDst.u32Height * pstDst.au32Stride[0] * 3); // 3: multiple
    if (HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }
    // 3: multiple
    memset_s((void *)pstDst.au64VirAddr[0], pstDst.u32Height * pstDst.au32Stride[0] * 3,
             0, pstDst.u32Height * pstDst.au32Stride[0] * 3); // 3: multiple
    HI_BOOL bInstant = HI_TRUE;

    s32Ret = HI_MPI_IVE_CSC(&hIveHandle, &pstSrc, &pstDst, &stCscCtrl, bInstant);
    if (HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        return s32Ret;
    }

    if (HI_TRUE == bInstant)
    {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
        {
            usleep(100); // 100: usleep time
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
    }
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];
    dstImage->u32Width = pstDst.u32Width;
    dstImage->u32Height = pstDst.u32Height;

    return HI_SUCCESS;
}
// 视频帧转为Mat
static HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame, Mat &dstMat)
{
    HI_U32 w = srcFrame->stVFrame.u32Width;
    HI_U32 h = srcFrame->stVFrame.u32Height;
    int bufLen = w * h * 3;
    HI_U8 *srcRGB = NULL;
    IPC_IMAGE dstImage;
    if (yuvFrame2rgb(srcFrame, &dstImage) != HI_SUCCESS)
    {
        SAMPLE_PRT("yuvFrame2rgb err\n");
        return HI_FAILURE;
    }
    srcRGB = (HI_U8 *)dstImage.u64VirAddr;
    dstMat.create(h, w, CV_8UC3);
    memcpy_s(dstMat.data, bufLen * sizeof(HI_U8), srcRGB, bufLen * sizeof(HI_U8));
    HI_MPI_SYS_MmzFree(dstImage.u64PhyAddr, (void *)&(dstImage.u64VirAddr));
    return HI_SUCCESS;
}

HI_S32 chessboard_detect::ChessModelDetectLoad(uintptr_t *model)
{
    HI_S32 ret = 1;
    *model = 1;
    SAMPLE_PRT("ChessModelDetectLoad success\n");
    return ret;
}

HI_S32 chessboard_detect::ChessModelDetectUnload(uintptr_t model)
{
    model = 0;
    return HI_SUCCESS;
}
// 手动校准，屏幕会显示矩形框供参考
HI_S32 chessboard_detect::ChessBoardManualCalibrate(VIDEO_FRAME_INFO_S *dstFrm)
{
    int ret = 0;
    for (int x = 0; x < 10; x++)
    {
        TESTBOARDboxs[x].xmin = 516 + x;
        TESTBOARDboxs[x].xmax = 1396 + x;
        TESTBOARDboxs[x].ymin = 54 + x;
        TESTBOARDboxs[x].ymax = 1016 + x;
    }
    // 棋盘校正框显示
    ret = MppFrmDrawRects(dstFrm, TESTBOARDboxs, 10, RGB888_GREEN, 2); // 2: DRAW_RETC_THICK
    if (ret != HI_SUCCESS)
        SAMPLE_PRT("Error(%#x),MppFrmDrawRects failed!\n", ret);
    for (int x = 0; x < 3; x++)
    {
        TESTBOARDboxs_B[x].xmin = 958 + x;
        TESTBOARDboxs_B[x].xmax = 960 + x;
        TESTBOARDboxs_B[x].ymin = 58 + x;
        TESTBOARDboxs_B[x].ymax = 1020 + x;
    }
    ret = MppFrmDrawRects(dstFrm, TESTBOARDboxs_B, 3, RGB888_GREEN, 2); // 2: DRAW_RETC_THICK
    if (ret != HI_SUCCESS)
        SAMPLE_PRT("Error(%#x),MppFrmDrawRects failed!\n", ret);
    for (int x = 0; x < 3; x++)
    {
        TESTBOARDboxs_C[x].xmin = 518 + x;
        TESTBOARDboxs_C[x].xmax = 1398 + x;
        TESTBOARDboxs_C[x].ymin = 538 + x;
        TESTBOARDboxs_C[x].ymax = 540 + x;
    }
    ret = MppFrmDrawRects(dstFrm, TESTBOARDboxs_C, 3, RGB888_GREEN, 2); // 2: DRAW_RETC_THICK
    Mat srcImage;
    frame2Mat(dstFrm, srcImage);
    if (srcImage.size == 0)
    {
        SAMPLE_PRT("image is null\n");
        return HI_FAILURE;
    }
    imwrite("./CALIB/Result.jpg", srcImage);
    if (ret != HI_SUCCESS)
        SAMPLE_PRT("Error(%#x),MppFrmDrawRects failed!\n", ret);
    return ret;
}

void find_max_min_chess_loc()
{
}

// 利用棋盘横纵像素坐标值最大和最小的两颗棋子的横纵坐标解算其他棋子的行列
static HI_S32 location_cal(struct CHESS chess[], int max_x, int min_x, int max_y, int min_y, vector<Vec3f> CIR)
{
    float x1 = max_x;
    float y1 = max_y;
    float x2 = min_x;
    float y2 = min_y;
    float delta_x = x1 - x2, delta_y = y1 - y2;
    float unit_x = delta_x / 8;
    float river_y = delta_y - 8 * unit_x;
    for (int i = 0; i < CIR.size(); i++)
    {
        int cols = (int)(chess[i].center_x - x2) / (int)unit_x;
        if (((int)(chess[i].center_x - x2) % (int)unit_x) <= 0.5 * (int)unit_x)
            chess[i].cols = cols;
        else if (((int)(chess[i].center_x - x2) % (int)unit_x) >= 0.5 * (int)unit_x)
            chess[i].cols = cols + 1;
    }
    for (int i = 0; i < CIR.size(); i++)
    {
        int rows = (int)(chess[i].center_y - y2) / (int)unit_x;
        if (rows <= 4)
        {
            if (((int)(chess[i].center_y - y2) % (int)unit_x) <= 0.5 * (int)unit_x)
                chess[i].rows = rows;
            else if (((int)(chess[i].center_y - y2) % (int)unit_x) >= 0.5 * (int)unit_x)
                chess[i].rows = rows + 1;
        }
        else
        {
            rows = 5 + (((int)(chess[i].center_y - y2) - 4 * (int)unit_x - (int)river_y) / (int)unit_x);
            if (((int)(chess[i].center_y - y2) - 4 * (int)unit_x - (int)river_y) % (int)unit_x <= 0.5 * (int)unit_x)
                chess[i].rows = rows;
            else if (((int)(chess[i].center_y - y2) - 4 * (int)unit_x - (int)river_y) % (int)unit_x >= 0.5 * (int)unit_x)
                chess[i].rows = rows + 1;
        }
    }
    for (int i = 0; i < CIR.size(); i++)
        printf("Num: %d, cols: %d, rows: %d\n", i, chess[i].cols, chess[i].rows);

    return HI_SUCCESS;
}

int max_n = 0, min_n = 0;
float max_num = 0, min_num = 999999;
float x_max, y_max, x_min, y_min;

// 棋盘检测推理
HI_S32 chessboard_detect::ChessBoardDetect(VIDEO_FRAME_INFO_S *srcFrm, int uart_fdnum, struct timeval *Time)
{
    int ret = 0;
    int num_key = GpioRead(1);
    Mat srcImage, midImage;
    frame2Mat(srcFrm, srcImage);
    if (srcImage.size == 0)
    {
        SAMPLE_PRT("image is null\n");
        return HI_FAILURE;
    }
    cvtColor(srcImage, midImage, COLOR_BGR2GRAY);
    GaussianBlur(midImage, midImage, Size(9, 9), 2, 2);
    HoughCircles(midImage, circles, HOUGH_GRADIENT, 1, midImage.rows / 16, 64, 32, 30, 45);
    SAMPLE_PRT("Circus Number: %d \n", circles.size());
    if (num_key == 0) // 当按下按钮时
    {
        gettimeofday(Time, NULL);
        sprintf(str, "./picture/img%d.jpg", num);
        num++;
        Mat midimage;
        Mat img_copy;
        srcImage.copyTo(img_copy);
        imwrite("./CalibResult.jpg", srcImage);
        cvtColor(srcImage, midimage, COLOR_BGR2GRAY);                                           // 转化边缘检测后的图为灰度图
        GaussianBlur(midimage, midimage, Size(9, 9), 2, 2);                                     // 高斯变换
        HoughCircles(midimage, circles, HOUGH_GRADIENT, 1, midimage.rows / 16, 64, 32, 30, 45); // 进行霍夫圆变换

        float x1, y1, r;
        for (size_t i = 0; i < circles.size(); i++)
        {
            sprintf(str_chess, "./chess_picture/img_chess%d_cut.png", i + 1);
            x1 = circles[i][0], y1 = circles[i][1], r = circles[i][2];
            // 拷贝图像
            // 将图像中的棋子部分切割出来，单独存储
            Mat img_copy_cut = img_copy(Rect((int)(x1 - r), (int)(y1 - r), (int)(2 * r), (int)(2 * r)));
            Mat Dst;
            if (lightMean(img_copy_cut) < 110)
                Dst = Light_Contrast_Change(img_copy_cut, 1.2, 35);
            else if (lightMean(img_copy_cut) < 125)
                Dst = Light_Contrast_Change(img_copy_cut, 1.2, 25);
            else if (lightMean(img_copy_cut) < 140)
                Dst = Light_Contrast_Change(img_copy_cut, 1.1, 15);
            else
                img_copy_cut.copyTo(Dst);
            Mat hsvImage;
            Size DstSize = Dst.size();
            int h = DstSize.height, w = DstSize.width;
            long pixel = h * w;
            cvtColor(img_copy_cut, hsvImage, COLOR_BGR2HSV); // 将图像从BGR转换为HSV色彩空间

            Mat mask, mask2; // 根据阈值创建掩膜
            inRange(hsvImage, Scalar(lowerH, lowerS, lowerV), Scalar(upperH, upperS, upperV), mask);
            inRange(hsvImage, Scalar(lowerH_K, lowerS_K, lowerV_K), Scalar(upperH_K, upperS_K, upperV_K), mask2);
            long red_pixels_count_a = countNonZero(mask); // 统计红色像素个数
            long red_pixels_count_b = countNonZero(mask2);
            printf("%ld, %ld\n", red_pixels_count_a, red_pixels_count_b);
            if (red_pixels_count_a > red_pixels_count_b)
                CHESS_CROPED[i].color = COLOR_RED; // 红色
            else
                CHESS_CROPED[i].color = COLOR_BLACK; // 黑色
            imwrite(str_chess, Dst);
        }
        // 依次在图中绘制出圆
        for (size_t i = 0; i < circles.size(); i++)
        {
            // 参数定义
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // 绘制圆心
            circle(srcImage, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            // 绘制圆轮廓
            if (CHESS_CROPED[i].color == COLOR_RED)
                circle(srcImage, center, radius, Scalar(155, 50, 255), 3, 8, 0);
            if (CHESS_CROPED[i].color == COLOR_BLACK)
                circle(srcImage, center, radius, Scalar(255, 255, 255), 3, 8, 0);
            x1 = circles[i][0], y1 = circles[i][1], r = circles[i][2];
            CHESS_CROPED[i].num = i + 1;
            CHESS_CROPED[i].center_x = x1;
            CHESS_CROPED[i].center_y = y1;
            CHESS_CROPED[i].radius_s = r;
        }
        if (first_time_chessboard_cal == HI_TRUE)
        {
            for (size_t i = 0; i < circles.size(); i++)
            {
                if (circles[i][0] + circles[i][1] > max_num) // 大
                {
                    max_num = circles[i][0] + circles[i][1];
                    max_n = i;
                }
                if (circles[i][0] + circles[i][1] < min_num) // 小
                {
                    min_num = circles[i][0] + circles[i][1];
                    min_n = i;
                }
            }
            x_max = CHESS_CROPED[max_n].center_x;
            x_min = CHESS_CROPED[min_n].center_x;
            y_max = CHESS_CROPED[max_n].center_y;
            y_min = CHESS_CROPED[min_n].center_y;
        }
        // x_max, x_min, y_max, y_min
        location_cal(CHESS_CROPED, 1353, 579, 961, 121, circles); // 560 99 1342 973
        // 输出效果图
        imwrite("./Finalresult.jpg", srcImage);
        // 输出检测圆的个数
        SAMPLE_PRT("Circus Number: %d\n", circles.size());
        int move_chessnum = -127, move_chesscolor = -127;
        if (first_time_chessboard_cal == HI_FALSE && chessnum == circles.size())
        {
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    display_new[j][i].color_number = -127;
                    display_new[j][i].actual_number = -127;
                }
            }
            for (int i = 0; i < circles.size(); i++)
            {
                int x = CHESS_CROPED[i].cols;
                int y = CHESS_CROPED[i].rows;
                display_new[x][y].color_number = CHESS_CROPED[i].color;
                display_new[x][y].actual_number = 100;
            }
            move_chessnum = -127, move_chesscolor = -127;
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    if (display_abbre[j][i].actual_number != -127 && display_new[j][i].actual_number == -127)
                    {
                        move_chessnum = display_abbre[j][i].actual_number;
                        move_chesscolor = display_abbre[j][i].color_number;
                        display_abbre[j][i].actual_number = -127;
                        display_abbre[j][i].color_number = -127;
                    }
                }
            }
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    if (display_abbre[j][i].actual_number == -127 && display_new[j][i].actual_number == 100)
                    {
                        display_abbre[j][i].actual_number = move_chessnum;
                        display_abbre[j][i].color_number = move_chesscolor;
                        SAMPLE_PRT("CHESS: %d, %d, %s, %s\n", i, j,
                                   chess_name[display_abbre[j][i].actual_number], chess_color[display_abbre[j][i].color_number - 1]);
                    }
                }
            }
        }
        else if (first_time_chessboard_cal == HI_FALSE && chessnum != circles.size())
        {
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    display_new[j][i].color_number = -127;
                    display_new[j][i].actual_number = -127;
                }
            }
            for (int i = 0; i < circles.size(); i++)
            {
                int x = CHESS_CROPED[i].cols;
                int y = CHESS_CROPED[i].rows;
                display_new[x][y].color_number = CHESS_CROPED[i].color;
                display_new[x][y].actual_number = 100;
            }
            move_chessnum = -127, move_chesscolor = -127;
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    if (display_abbre[j][i].actual_number != -127 && display_new[j][i].actual_number == -127)
                    {
                        move_chessnum = display_abbre[j][i].actual_number;
                        move_chesscolor = display_abbre[j][i].color_number;
                        display_abbre[j][i].actual_number = -127;
                        display_abbre[j][i].color_number = -127;
                        SAMPLE_PRT("CHESS: %d, %d, %s, %s\n", i, j,
                                   chess_name[move_chessnum], chess_color[move_chesscolor - 1]);
                    }
                }
            }
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < 9; j++)
                {
                    if (display_abbre[j][i].actual_number != -127 &&
                        (display_new[j][i].color_number != display_abbre[j][i].color_number))
                    {
                        display_abbre[j][i].actual_number = move_chessnum;
                        display_abbre[j][i].color_number = move_chesscolor;
                        SAMPLE_PRT("CHESS: %d, %d, %s, %s\n", i, j,
                                   chess_name[display_abbre[j][i].actual_number], chess_color[display_abbre[j][i].color_number - 1]);
                    }
                }
            }
        }
        first_time_chessboard_cal = HI_FALSE;
        chessnum = circles.size();
        return 127;
    }
    return ret;
}

// 图片缩放、二值化与模型推理
HI_S32 chessboard_detect::ChessBoardNnieProcess(int flag)
{
    int ret = 0;
    if (flag == HI_FALSE)
    {
        SAMPLE_PRT("Not First Time Run, No Need to Run NNIE!\n");
        return ret;
    }
    for (int i = 0; i < circles.size(); i++)
    {
        IVE_IMAGE_S Srcimage;
        sprintf(str_name, "./chess_picture/img_chess%d_cut.png", i + 1);
        Mat Res = imread(str_name), midRes;
        Size dst_size(128, 128);
        resize(Res, midRes, dst_size);
        Mat gray, finalimage;
        cvtColor(midRes, gray, COLOR_BGR2GRAY);
        threshold(gray, gray, 127, 255, THRESH_BINARY); // 127？
        bitwise_not(gray, finalimage);
        imwrite("./final.png", finalimage);
        ret = Mat_File_Convert_To_IVE_YUV(&Srcimage, "./final.png");
        if (ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Mat to frame error, err code = %#x", ret);
            return ret;
        }
        ret = CnnChessClassifyCal(&Srcimage, &(CHESS_CROPED[i].actual_num));
        if (ret != HI_SUCCESS)
        {
            SAMPLE_PRT("Cnn Chess Cal error, err code = %#x", ret);
            return ret;
        }
        Ive_Img_Destroy(&Srcimage);
    }
    for (int j = 0; j < circles.size(); j++)
        SAMPLE_PRT("Num: %d, Result: %s\n", j + 1, chess_name[CHESS_CROPED[j].actual_num]);
    return ret;
}

char FENstring[] = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 0 1";
static int have_written_flag = HI_FALSE;
const struct CHESS_DIS_ABBRE chess_normal[9][10] =
    {
        {{2, 2}, {-127, -127}, {-127, -127}, {9, 2}, {-127, -127}, {-127, -127}, {0, 1}, {-127, -127}, {-127, -127}, {2, 1}},
        {{3, 2}, {-127, -127}, {4, 2}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {4, 1}, {-127, -127}, {3, 1}},
        {{8, 2}, {-127, -127}, {-127, -127}, {9, 2}, {-127, -127}, {-127, -127}, {0, 1}, {-127, -127}, {-127, -127}, {7, 1}},
        {{5, 2}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {5, 1}},
        {{1, 2}, {-127, -127}, {-127, -127}, {9, 2}, {-127, -127}, {-127, -127}, {0, 1}, {-127, -127}, {-127, -127}, {6, 1}},
        {{5, 2}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {5, 1}},
        {{8, 2}, {-127, -127}, {-127, -127}, {9, 2}, {-127, -127}, {-127, -127}, {0, 1}, {-127, -127}, {-127, -127}, {7, 1}},
        {{3, 2}, {-127, -127}, {4, 2}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {4, 1}, {-127, -127}, {3, 1}},
        {{2, 2}, {-127, -127}, {-127, -127}, {9, 2}, {-127, -127}, {-127, -127}, {0, 1}, {-127, -127}, {-127, -127}, {2, 1}}};
// {
//     {{2, 2}, {3, 2}, {8, 2}, {5, 2}, {1, 2}, {5, 2}, {8, 2}, {3, 2}, {2, 2}},
//     {{-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}},
//     {{-127, -127}, {4, 2}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {4, 2}, {-127, -127}},
//     {{9, 2}, {-127, -127}, {9, 2}, {-127, -127}, {9, 2}, {-127, -127}, {9, 2}, {-127, -127}, {9, 2}},
//     {{-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}},
//     {{-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}},
//     {{0, 1}, {-127, -127}, {0, 1}, {-127, -127}, {0, 1}, {-127, -127}, {0, 1}, {-127, -127}, {0, 1}},
//     {{-127, -127}, {4, 1}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {4, 1}, {-127, -127}},
//     {{-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}, {-127, -127}},
//     {{2, 1}, {3, 1}, {7, 1}, {5, 1}, {6, 2}, {5, 1}, {7, 1}, {3, 1}, {2, 1}}
// };

// 显示棋盘信息，并转换为FEN格式棋谱
void ChessBoardInfoDisplay(int flag_a, int flag_b)
{
    if (flag_a == HI_TRUE && have_written_flag == HI_FALSE)
    {
        have_written_flag = HI_TRUE;
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                display_abbre[j][i].actual_number = chess_normal[j][i].actual_number;
                display_abbre[j][i].color_number = chess_normal[j][i].color_number;
                if (display_abbre[j][i].color_number != -127 && display_abbre[j][i].actual_number != -127)
                {
                    strcpy(display[j][i].color_name, chess_color[display_abbre[j][i].color_number - 1]);
                    strcpy(display[j][i].name, chess_name[display_abbre[j][i].actual_number]);
                }
                else if (display_abbre[j][i].color_number == -127 && display_abbre[j][i].actual_number == -127)
                {
                    strcpy(display[j][i].color_name, "None");
                    strcpy(display[j][i].name, "None");
                }
            }
        }
    }
    else if (flag_a == HI_FALSE && flag_b == HI_TRUE)
    {
        for (int i = 0; i < circles.size(); i++)
        {
            int x = CHESS_CROPED[i].cols;
            int y = CHESS_CROPED[i].rows;
            strcpy(display[x][y].color_name, chess_color[CHESS_CROPED[i].color - 1]);
            strcpy(display[x][y].name, chess_name[CHESS_CROPED[i].actual_num]);
            display_abbre[x][y].actual_number = CHESS_CROPED[i].actual_num;
            display_abbre[x][y].color_number = CHESS_CROPED[i].color;
        }
    }
    else
    {
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                if (display_abbre[j][i].color_number != -127 && display_abbre[j][i].actual_number != -127)
                {
                    strcpy(display[j][i].color_name, chess_color[display_abbre[j][i].color_number - 1]);
                    strcpy(display[j][i].name, chess_name[display_abbre[j][i].actual_number]);
                }
                else if (display_abbre[j][i].color_number == -127 && display_abbre[j][i].actual_number == -127)
                {
                    strcpy(display[j][i].color_name, "None");
                    strcpy(display[j][i].name, "None");
                }
            }
        }
    }
    for (int i = 0; i < 10; i++)
    {
        printf("------------------------------------------------------------------------------------\n");
        for (int j = 0; j < 9; j++)
        {
            printf("(%-7s, %-5s)", display[j][i].name, display[j][i].color_name);
        }
        printf("\n");
    }
    printf("\n");
    int w = 0;
    char *strpointer = FENstring;
    for (int i = 0; i < 10; i++)
    {
        int empty_num = 0;
        for (int j = 0; j < 9; j++)
        {
            if (display_abbre[j][i].actual_number == -127)
                empty_num++;
            else if (display_abbre[j][i].color_number == COLOR_RED)
            {
                if (empty_num != 0)
                {
                    strpointer[w] = (char)(empty_num + 48);
                    w++;
                }
                strpointer[w] = chess_abbre[display_abbre[j][i].actual_number][0];
                w++;
                empty_num = 0;
            }
            else
            {
                if (empty_num != 0)
                {
                    strpointer[w] = (char)(empty_num + 48);
                    w++;
                }
                strpointer[w] = chess_abbre_B[display_abbre[j][i].actual_number][0];
                w++;
                empty_num = 0;
            }
        }
        if (empty_num != 0)
        {
            strpointer[w] = (char)(empty_num + 48);
            w++;
        }
        if (i <= 8)
        {
            strpointer[w] = '/';
            w++;
        }
    }
    char *sw = " b - - 0 1";
    for (int p = 0; p < 11; p++)
    {
        strpointer[w] = sw[p];
        w++;
    }
    printf("\n");
}

void ChessBoardInfoLoad(char *loadstr)
{
    strcpy(loadstr, FENstring);
    SAMPLE_PRT("%s\n", loadstr);
}

// 吃子指令转换
void chess_eat_sign_convert(char *chess_msg)
{
    int ch1 = (9 - (chess_msg[1] - 'a'));
    int ch2 = (10 - (chess_msg[4] - '0'));
    sprintf(chess_msg, "%%%d ,%d, %d, %d", ch2, ch1, 1, 0);
    display_abbre[9 - ch1][ch2 - 1].color_number = -127;
    display_abbre[9 - ch1][ch2 - 1].actual_number = -127;
    chessnum = chessnum - 1;
}

// 正常移动棋子指令转化
void chess_sign_convert(char *chess_msg)
{
    int ch1 = (9 - (chess_msg[1] - 'a'));
    int ch2 = (10 - (chess_msg[4] - '0'));

    int ch3 = (9 - (chess_msg[7] - 'a'));
    int ch4 = (10 - (chess_msg[10] - '0'));
    sprintf(chess_msg, "%%%d ,%d, %d, %d", ch2, ch1, ch4, ch3);

    int temp_colornum = display_abbre[9 - ch1][ch2 - 1].color_number;
    int temp_actualnum = display_abbre[9 - ch1][ch2 - 1].actual_number;
    display_abbre[9 - ch1][ch2 - 1].color_number = -127;
    display_abbre[9 - ch1][ch2 - 1].actual_number = -127;
    display_abbre[9 - ch3][ch4 - 1].color_number = temp_colornum;
    display_abbre[9 - ch3][ch4 - 1].actual_number = temp_actualnum;
}

// 将服务器的指令转化为发送给Hi3861的指令格式
int ReceivedMsgTransToHi3861(char *srcstr, char *dststr, char *chess_eat)
{
    SAMPLE_PRT("%s\n", srcstr);
    int chess_eaten = HI_FALSE;
    if (srcstr[0] != '0')
    {
        chess_eaten = HI_TRUE;
        for (int j = 0; srcstr[j] != ','; j++)
        {
            chess_eat[j * 3 + 1] = srcstr[j];
        }
        chess_eat_sign_convert(chess_eat);
        SAMPLE_PRT("%s\n", chess_eat);
    }
    int i = 0;
    for (i = 0; srcstr[i] != ','; i++)
    {
        printf("%c", srcstr[i]);
    }
    i++;
    for (; srcstr[i] != ','; i++)
    {
        printf("%c", srcstr[i]);
    }
    printf("\n");
    i++;
    for (int w = 0; w < 4; i++, w++)
    {
        dststr[w * 3 + 1] = srcstr[i];
    }
    chess_sign_convert(dststr);
    SAMPLE_PRT("%s\n", dststr);
    return chess_eaten;
}
