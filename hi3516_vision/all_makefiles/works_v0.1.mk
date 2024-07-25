#  Copyright (c) 2022 HiSilicon (Shanghai) Technologies CO., LIMITED.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# Hisilicon ai sample makefile

# 编译所需的公共文件
include ./mk.param

# 编译后生成的可执行文件名
TARGET := works_test

# 编译此工程需要依赖的.c文件
OPENCV_LIB_DIR = $(WORKS_V0.1_DIR)/third_party/output/opencv/lib
CHESS_DETECT_DIR = $(WORKS_V0.1_DIR)/scenario/chess_detect

SMP_SRCS += $(PLATRORM_COMMON_DIR)/sample_comm_isp.c
SMP_SRCS += $(PLATRORM_COMMON_DIR)/sample_comm_sys.c
SMP_SRCS += $(PLATRORM_COMMON_DIR)/sample_comm_vi.c
SMP_SRCS += $(PLATRORM_COMMON_DIR)/sample_comm_vo.c
SMP_SRCS += $(PLATRORM_COMMON_DIR)/sample_comm_vpss.c
SMP_SRCS += $(SVP_COMMON_DIR)/sample_comm_nnie.c
SMP_SRCS += $(SVP_COMMON_DIR)/sample_comm_svp.c
SMP_SRCS += $(NNIE_SAMPLE_DIR)/sample_nnie_software/sample_svp_nnie_software.c
SMP_SRCS += $(WORKS_V0.1_DIR)/ai_infer_process/ai_infer_process.c
SMP_SRCS += $(WORKS_V0.1_DIR)/ext_util/base_interface.c
SMP_SRCS += $(WORKS_V0.1_DIR)/ext_util/misc_util.c
SMP_SRCS += $(WORKS_V0.1_DIR)/ext_util/posix_help.c
SMP_SRCS += $(WORKS_V0.1_DIR)/interconnection_server/hisignalling.c
SMP_SRCS += $(WORKS_V0.1_DIR)/mpp_help/src/ive_img.c
SMP_SRCS += $(WORKS_V0.1_DIR)/mpp_help/src/osd_img.c
SMP_SRCS += $(WORKS_V0.1_DIR)/mpp_help/src/vgs_img.c
SMP_SRCS += $(WORKS_V0.1_DIR)/scenario/hand_classify/hand_classify.c
SMP_SRCS += $(WORKS_V0.1_DIR)/scenario/hand_classify/yolov2_hand_detect.c
SMP_SRCS += $(WORKS_V0.1_DIR)/third_party/src/iniparser/src/dictionary.c
SMP_SRCS += $(WORKS_V0.1_DIR)/third_party/src/iniparser/src/iniparser.c
SMP_SRCS += $(WORKS_V0.1_DIR)/smp/sample_media_ai.c

# 编译此工程需要依赖的.cpp文件
CPP_SRCS += $(WORKS_V0.1_DIR)/smp/sample_media_opencv.cpp
CPP_SRCS += $(CHESS_DETECT_DIR)/chess_detect.cpp
CPP_SRCS += $(WORKS_V0.1_DIR)/smp/sample_ai_main.cpp

# 编译此工程需要包含的头文件
CFLAGS += -I$(NNIE_SAMPLE_DIR)/sample_nnie_software
CFLAGS += -I$(NNIE_SAMPLE_DIR)/smp
CFLAGS += -I$(WORKS_V0.1_DIR)/ai_infer_process
CFLAGS += -I$(WORKS_V0.1_DIR)/smp
CFLAGS += -I$(WORKS_V0.1_DIR)/ext_util
CFLAGS += -I$(WORKS_V0.1_DIR)/interconnection_server
CFLAGS += -I$(WORKS_V0.1_DIR)/scenario/hand_classify
CFLAGS += -I$(CHESS_DETECT_DIR)
CFLAGS += -I$(WORKS_V0.1_DIR)/mpp_help/include
CFLAGS += -I$(WORKS_V0.1_DIR)/third_party/src/iniparser/src
CFLAGS += -Wno-psabi -I$(WORKS_V0.1_DIR)/third_party/output/opencv/include/opencv4

# 编译此工程需要依赖的库函数
MPI_LIBS += -L$(OPENCV_LIB_DIR)
MPI_LIBS += -lopencv_world -lpthread -lrt -ldl

# 根据实际的类型设置，可以用set_sensor脚本设置
SENSOR0_TYPE = -DSENSOR0_TYPE=SONY_IMX335_MIPI_4M_30FPS_12BIT
SENSOR1_TYPE = -DSENSOR1_TYPE=SONY_IMX335_MIPI_4M_30FPS_12BIT
CFLAGS += $(SENSOR0_TYPE)
CFLAGS += $(SENSOR1_TYPE)

# 编译工程所需的mk文件
include ./sdk_linux.mk