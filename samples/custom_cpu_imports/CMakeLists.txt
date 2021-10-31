# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(_IMPORTS_NAME "iree_samples_custom_cpu_imports")
add_library(${_IMPORTS_NAME} STATIC)
target_sources(${_IMPORTS_NAME}
  PRIVATE
    imports.c
)
target_link_libraries(${_IMPORTS_NAME}
  iree_base_base
  iree_hal_local_executable_loader
)
