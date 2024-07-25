#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <signal.h>
#include <limits.h>
#include "sample_comm.h"
#include "acodec.h"
#include "audio_aac_adp.h"
#include "audio_dl_adp.h"
#include "hi_resampler_api.h"

#if defined(HI_VQE_USE_STATIC_MODULE_REGISTER)
#include "hi_vqe_register_api.h"
#endif

#define SAMPLE_DBG(s32Ret)                                                         \
    do                                                                             \
    {                                                                              \
        printf("s32Ret = %#x, fuc:%s, line:%d\n", s32Ret, __FUNCTION__, __LINE__); \
    } while (0)

static PAYLOAD_TYPE_E gs_enPayloadType = PT_AAC;
static HI_BOOL gs_bAioReSample = HI_FALSE;
static AUDIO_SAMPLE_RATE_E g_in_sample_rate = AUDIO_SAMPLE_RATE_BUTT;
static AUDIO_SAMPLE_RATE_E g_out_sample_rate = AUDIO_SAMPLE_RATE_BUTT;
static HI_BOOL g_sample_audio_exit = HI_FALSE;

/* function : Add dynamic load path */
#ifndef HI_VQE_USE_STATIC_MODULE_REGISTER
static HI_VOID SAMPLE_AUDIO_AddLibPath(HI_VOID)
{
    HI_S32 s32Ret;
    HI_CHAR aszLibPath[FILE_NAME_LEN] = {0};
#if defined(__HuaweiLite__) && (!defined(__OHOS__))
    s32Ret = snprintf_s(aszLibPath, FILE_NAME_LEN, FILE_NAME_LEN - 1, "/sharefs/");
    if (s32Ret <= EOK)
    {
        printf("\n snprintf_s fail! ret = 0x%x", s32Ret);
        return;
    }
#else
#endif
    s32Ret = Audio_Dlpath(aszLibPath);
    if (s32Ret != HI_SUCCESS)
    {
        printf("%s: add lib path %s failed\n", __FUNCTION__, aszLibPath);
    }
    return;
}
#endif

static HI_S32 SAMPLE_AUDIO_Getchar(HI_VOID)
{
    int c;

    if (g_sample_audio_exit == HI_TRUE)
    {
        SAMPLE_COMM_AUDIO_DestroyAllTrd();
        SAMPLE_COMM_SYS_Exit();
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
        return -1;
    }

    c = getchar();

    if (g_sample_audio_exit == HI_TRUE)
    {
        SAMPLE_COMM_AUDIO_DestroyAllTrd();
        SAMPLE_COMM_SYS_Exit();
        printf("\033[0;31mprogram exit abnormally!\033[0;39m\n");
        return -1;
    }

    return c;
}

/* function : PT Number to String */
static char *SAMPLE_AUDIO_Pt2Str(PAYLOAD_TYPE_E enType)
{
    if (enType == PT_G711A)
    {
        return "g711a";
    }
    else if (enType == PT_G711U)
    {
        return "g711u";
    }
    else if (enType == PT_ADPCMA)
    {
        return "adpcm";
    }
    else if (enType == PT_G726)
    {
        return "g726";
    }
    else if (enType == PT_LPCM)
    {
        return "pcm";
    }
    else if (enType == PT_AAC)
    {
        return "aac";
    }
    else if (enType == PT_MP3)
    {
        return "mp3";
    }
    else
    {
        return "data";
    }
}

static hi_void sample_audio_adec_ao_init_param(AIO_ATTR_S *attr)
{
    attr->enSamplerate = AUDIO_SAMPLE_RATE_48000;
    attr->u32FrmNum = FPS_30;
    attr->enBitwidth = AUDIO_BIT_WIDTH_16;
    attr->enWorkmode = AIO_MODE_I2S_MASTER;
    attr->enSoundmode = AUDIO_SOUND_MODE_MONO; // AUDIO_SOUND_MODE_STEREO;
    attr->u32ChnCnt = 1;                       /* 2: chn num */
    attr->u32ClkSel = 1;
    attr->enI2sType = AIO_I2STYPE_INNERCODEC;
    attr->u32PtNumPerFrm = AACLC_SAMPLES_PER_FRAME;
    attr->u32EXFlag = 0;

    gs_bAioReSample = HI_FALSE;
    g_in_sample_rate = AUDIO_SAMPLE_RATE_BUTT;
    g_out_sample_rate = AUDIO_SAMPLE_RATE_BUTT;
}

static FILE *SAMPLE_AUDIO_OpenAdecFile(ADEC_CHN AdChn, PAYLOAD_TYPE_E enType, const char *path)
{
    FILE *pfd = NULL;
    // HI_CHAR aszFileName[FILE_NAME_LEN] = {0};
    HI_S32 s32Ret;
    // HI_CHAR path[PATH_MAX] = {0};

    /* create file for save stream */
    // s32Ret = snprintf_s(aszFileName, FILE_NAME_LEN, FILE_NAME_LEN - 1,
    //     "audio_chn%d.%s", AdChn, SAMPLE_AUDIO_Pt2Str(enType));
    // if (s32Ret <= EOK) {
    //     printf("\n snprintf_s fail! ret = 0x%x", s32Ret);
    //     return NULL;
    // }

    // if (realpath(aszFileName, path) == NULL) {
    //     printf("[func]:%s [line]:%d [info]:%s\n", __FUNCTION__, __LINE__, "adec file name realpath fail");
    //     return NULL;
    // }
    pfd = fopen(path, "rb");
    if (pfd == NULL)
    {
        printf("%s: open file %s failed\n", __FUNCTION__, path);
        return NULL;
    }

    printf("open stream file:\"%s\" for adec ok\n", path);
    return pfd;
}

static HI_VOID SAMPLE_AUDIO_AdecAoInner(AUDIO_DEV AoDev, AO_CHN AoChn, ADEC_CHN AdChn, const char *Audioname)
{
    HI_S32 s32Ret;
    FILE *pfd = NULL;

    s32Ret = SAMPLE_COMM_AUDIO_AoBindAdec(AoDev, AoChn, AdChn);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
        return;
    }

    pfd = SAMPLE_AUDIO_OpenAdecFile(AdChn, gs_enPayloadType, Audioname);
    if (pfd == NULL)
    {
        SAMPLE_DBG(HI_FAILURE);
        goto ADECAO_ERR0;
    }

    s32Ret = SAMPLE_COMM_AUDIO_CreateTrdFileAdec(AdChn, pfd);
    if (s32Ret != HI_SUCCESS)
    {
        (HI_VOID) fclose(pfd);
        pfd = HI_NULL;
        SAMPLE_DBG(s32Ret);
        goto ADECAO_ERR0;
    }

    HI_MPI_AO_SetVolume(AoDev, 0);
    printf("bind adec:%d to ao(%d,%d) ok \n", AdChn, AoDev, AoChn);
    printf("\nplease press twice ENTER to exit this sample\n");
    SAMPLE_AUDIO_Getchar();
    SAMPLE_AUDIO_Getchar();

    s32Ret = SAMPLE_COMM_AUDIO_DestroyTrdFileAdec(AdChn);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
        return;
    }

ADECAO_ERR0:
    s32Ret = SAMPLE_COMM_AUDIO_AoUnbindAdec(AoDev, AoChn, AdChn);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
    }

    return;
}

#if defined(HI_VQE_USE_STATIC_MODULE_REGISTER)
/* function : to register vqe module */
static HI_S32 SAMPLE_AUDIO_RegisterVQEModule(HI_VOID)
{
    HI_S32 s32Ret;
    AUDIO_VQE_REGISTER_S stVqeRegCfg = {0};
    /* Resample */
    stVqeRegCfg.stResModCfg.pHandle = HI_VQE_RESAMPLE_GetHandle();
    /* RecordVQE */
    stVqeRegCfg.stRecordModCfg.pHandle = HI_VQE_RECORD_GetHandle();
    /* TalkVQE */
    stVqeRegCfg.stHpfModCfg.pHandle = HI_VQE_HPF_GetHandle();
    stVqeRegCfg.stAecModCfg.pHandle = HI_VQE_AEC_GetHandle();
    stVqeRegCfg.stAgcModCfg.pHandle = HI_VQE_AGC_GetHandle();
    stVqeRegCfg.stAnrModCfg.pHandle = HI_VQE_ANR_GetHandle();
    stVqeRegCfg.stEqModCfg.pHandle = HI_VQE_EQ_GetHandle();

    s32Ret = HI_MPI_AUDIO_RegisterVQEModule(&stVqeRegCfg);
    if (s32Ret != HI_SUCCESS)
    {
        printf("%s: register vqe module fail with s32Ret = 0x%x!\n", __FUNCTION__, s32Ret);
        return HI_FAILURE;
    }
    return HI_SUCCESS;
}
#endif

/* function : file -> Adec -> Ao */
static HI_S32 SAMPLE_AUDIO_AdecAo(char *filename)
{
    HI_S32 s32Ret, s32AoChnCnt;
    const AO_CHN AoChn = 0;
    const ADEC_CHN AdChn = 0;
    AIO_ATTR_S stAioAttr;
    AUDIO_DEV AoDev = SAMPLE_AUDIO_INNER_AO_DEV;

    sample_audio_adec_ao_init_param(&stAioAttr);

    s32Ret = SAMPLE_COMM_AUDIO_StartAdec(AdChn, gs_enPayloadType);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
        goto ADECAO_ERR3;
    }

    s32AoChnCnt = stAioAttr.u32ChnCnt;
    s32Ret = SAMPLE_COMM_AUDIO_StartAo(AoDev, s32AoChnCnt, &stAioAttr, g_in_sample_rate, gs_bAioReSample);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
        goto ADECAO_ERR2;
    }

    s32Ret = SAMPLE_COMM_AUDIO_CfgAcodec(&stAioAttr);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
        goto ADECAO_ERR1;
    }

    SAMPLE_AUDIO_AdecAoInner(AoDev, AoChn, AdChn, filename);

ADECAO_ERR1:
    s32Ret = SAMPLE_COMM_AUDIO_StopAo(AoDev, s32AoChnCnt, gs_bAioReSample);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
    }
ADECAO_ERR2:
    s32Ret = SAMPLE_COMM_AUDIO_StopAdec(AdChn);
    if (s32Ret != HI_SUCCESS)
    {
        SAMPLE_DBG(s32Ret);
    }

ADECAO_ERR3:
    return s32Ret;
}

HI_VOID AudioAddLibPath(HI_VOID)
{
    SAMPLE_AUDIO_AddLibPath();
}

HI_S32 AudioPlay(char *AudioFileName)
{
    HI_S32 s32Ret = HI_SUCCESS;

    s32Ret = HI_MPI_SYS_Init();
    HI_MPI_ADEC_AacInit();
    
    s32Ret = SAMPLE_AUDIO_AdecAo(AudioFileName);

    HI_MPI_ADEC_AacDeInit();
    HI_MPI_SYS_Exit();
    return s32Ret;
}