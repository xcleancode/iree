// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/local/executable_loader.h"

static int iree_test_func(void* context, void* params_ptr, void* reserved) {
  typedef struct {
    int32_t result0;
    int32_t result1;
    int32_t arg0;
    int32_t arg1;
  } params_t;
  params_t* params = (params_t*)params_ptr;
  params->result0 = params->arg0 * 2;
  params->result1 = params->arg1 * 2;
  return 0;
}

static int iree_test_printi(void* context, void* params_ptr, void* reserved) {
  typedef struct {
    int32_t arg0;
  } params_t;
  params_t* params = (params_t*)params_ptr;
  fprintf(stdout, "from dispatch: %d\n", params->arg0);
  fflush(stdout);
  return 0;
}

static iree_status_t iree_samples_custom_cpu_import_provider_resolve(
    void* self, iree_string_view_t symbol_name, void** out_fn_ptr,
    void** out_fn_context) {
  if (iree_string_view_equal(symbol_name, IREE_SV("iree_test_func"))) {
    *out_fn_ptr = iree_test_func;
    return iree_ok_status();
  } else if (iree_string_view_equal(symbol_name, IREE_SV("iree_test_printi"))) {
    *out_fn_ptr = iree_test_printi;
    return iree_ok_status();
  }
  return iree_status_from_code(IREE_STATUS_NOT_FOUND);
}

iree_hal_executable_import_provider_t iree_samples_custom_cpu_import_provider(
    void) {
  iree_hal_executable_import_provider_t import_provider = {
      .self = NULL,
      .resolve = iree_samples_custom_cpu_import_provider_resolve,
  };
  return import_provider;
}
