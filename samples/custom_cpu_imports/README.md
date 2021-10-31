DO NOT SUBMIT
This hardcodes each dispatch function compiled to call two functions
imported at runtime. The automatic conversion supports any func.call to
an external function that may exist in the IR prior to ConvertToLLVM as
well as calls materialized in patterns during conversion.

To link in the library providing the imports and register the imports on
all loaded executables set the following cmake flags:
```
-DCMAKE_C_FLAGS=-DIREE_HAL_EXECUTABLE_IMPORT_PROVIDER_DEFAULT_FN=iree_samples_custom_cpu_import_provider
-DIREE_HAL_EXECUTABLE_LOADER_EXTRA_DEPS=iree_samples_custom_cpu_imports
```
