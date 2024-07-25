

#ifndef AUDIO_PLAY_H
#define AUDIO_PLAY_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "sample_comm.h"

#if __cplusplus
extern "C"
{
#endif
    HI_VOID AudioAddLibPath(HI_VOID);
    HI_S32 AudioPlay(char *AudioFileName);

#ifdef __cplusplus
}
#endif

#endif