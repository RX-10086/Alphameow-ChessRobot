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

# Hisilicon workspace makefile

TARGETS := audio higv vio ive nnie ai_sample color_space_convert  helloworld  histreaming_server  light_sensor_demo  resize_crop  rtsp_sample  store_sample  switch_demo  uart_sample  works_v0.1

TARGETS_clean := ${addsuffix _clean,${TARGETS}}
TARGETS_rebuild := ${addsuffix _rebuild,${TARGETS}}

.PHONY: all clean rebuild $(TARGETS) $(TARGETS_clean) $(TARGETS_rebuild)
all: $(TARGETS)
rebuild: clean all
clean: ${TARGETS_clean}

$(TARGETS) :
	make -f ${addsuffix .mk,$@}
	@echo
	@echo  $@ build success
	@echo
${TARGETS_clean} :
	make -f ${patsubst %_clean,%,$@}.mk clean
	@echo
	@echo $@ clean success
	@echo
${TARGETS_rebuild} :
	make -f ${patsubst %_rebuild,%,$@}.mk clean
	make -f ${patsubst %_rebuild,%,$@}.mk
	@echo
	@echo   $@ rebuild success
	@echo
