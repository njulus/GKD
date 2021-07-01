# -*- coding: utf-8 -*-
"""
@Author: Su Lu

@Date: 2020-12-08 19:52:05
"""

import platform
import GPUtil

# determine the operating system and the GPUs available currently

if platform.platform().startswith('Windows'):
    WORKERS = 0
    n_gpus = 1
elif platform.platform().startswith('Linux'):
    WORKERS = 8
    n_gpus = 1

gpu_list = GPUtil.getAvailable(order='memory', limit=100, maxLoad=1.1, maxMemory=1.1)
DEVICES = [gpu_list[i] for i in range(0, n_gpus)]