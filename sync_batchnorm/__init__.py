# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from .batchnorm import (
    SynchronizedBatchNorm1d,
    SynchronizedBatchNorm2d,
    SynchronizedBatchNorm3d,
    convert_model,
    patch_sync_batchnorm,
    set_sbn_eps_mode,
)
from .replicate import DataParallelWithCallback, patch_replication_callback
