# Locate or download a prebuilt LiteRT C SDK + runtime and create imported target nn_litert.
# Sets NN_HAS_LITERT in the parent scope.

function(nn_litert_download url dest)
    unset(NN_LITERT_DOWNLOAD_ERROR PARENT_SCOPE)
    if(EXISTS "${dest}")
        return()
    endif()
    get_filename_component(_dir "${dest}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dir}")
    message(STATUS "Downloading ${url}")
    file(DOWNLOAD "${url}" "${dest}"
        SHOW_PROGRESS
        STATUS _dl
        TIMEOUT 180
        TLS_VERIFY ON
    )
    list(GET _dl 0 _dl_code)
    if(NOT _dl_code EQUAL 0)
        list(GET _dl 1 _dl_msg)
        file(REMOVE "${dest}")
        set(NN_LITERT_DOWNLOAD_ERROR "${_dl_msg}" PARENT_SCOPE)
    endif()
endfunction()

function(nn_litert_msvc_implib dll implib)
    if(EXISTS "${implib}")
        return()
    endif()
    set(_def "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/litert_exports.def")
    if(NOT EXISTS "${_def}")
        message(WARNING "LiteRT export list missing: ${_def}")
        return()
    endif()
    if(CMAKE_GENERATOR_PLATFORM MATCHES "ARM64|arm64")
        set(_mach ARM64)
    else()
        set(_mach x64)
    endif()
    if(NOT CMAKE_AR)
        message(WARNING "CMAKE_AR is unset; cannot generate a LiteRT import library.")
        return()
    endif()
    execute_process(
        COMMAND "${CMAKE_AR}" /nologo /def:${_def} /machine:${_mach} /name:libLiteRt.dll /out:${implib}
        RESULT_VARIABLE _lib_rc
        OUTPUT_VARIABLE _lib_out
        ERROR_VARIABLE _lib_err
    )
    if(NOT _lib_rc EQUAL 0 OR NOT EXISTS "${implib}")
        message(WARNING "Failed to generate LiteRT import library: ${_lib_out} ${_lib_err}")
        file(REMOVE "${implib}")
    endif()
endfunction()

function(nn_setup_litert)
    set(NN_HAS_LITERT FALSE PARENT_SCOPE)
    if(NOT NN_ENABLE_LITERT_RUNTIME)
        return()
    endif()

    set(_litert_ver "2.1.6")
    set(_sdk_header "litert/c/litert_compiled_model.h")

    if(DEFINED ENV{NN_LITERT_ROOT} AND EXISTS "$ENV{NN_LITERT_ROOT}/${_sdk_header}")
        set(_sdk_root "$ENV{NN_LITERT_ROOT}")
    elseif(DEFINED NN_LITERT_ROOT AND EXISTS "${NN_LITERT_ROOT}/${_sdk_header}")
        set(_sdk_root "${NN_LITERT_ROOT}")
    else()
        set(_sdk_dest "${CMAKE_BINARY_DIR}/_deps/litert_cc_sdk")
        set(_sdk_zip "${CMAKE_BINARY_DIR}/_deps/litert_cc_sdk.zip")
        if(NOT EXISTS "${_sdk_dest}/${_sdk_header}")
            nn_litert_download(
                "https://github.com/google-ai-edge/LiteRT/releases/download/v${_litert_ver}/litert_cc_sdk.zip"
                "${_sdk_zip}")
            if(NN_LITERT_DOWNLOAD_ERROR)
                message(WARNING "LiteRT C++ SDK download failed: ${NN_LITERT_DOWNLOAD_ERROR}. Building without the litert backend.")
                return()
            endif()
            file(ARCHIVE_EXTRACT INPUT "${_sdk_zip}" DESTINATION "${CMAKE_BINARY_DIR}/_deps")
        endif()
        if(EXISTS "${_sdk_dest}/${_sdk_header}")
            set(_sdk_root "${_sdk_dest}")
        elseif(EXISTS "${CMAKE_BINARY_DIR}/_deps/litert_cc_sdk/${_sdk_header}")
            set(_sdk_root "${CMAKE_BINARY_DIR}/_deps/litert_cc_sdk")
        else()
            message(WARNING "LiteRT SDK archive extracted but headers were not found. Building without the backend.")
            return()
        endif()
    endif()

    set(_proc "${CMAKE_SYSTEM_PROCESSOR}")
    string(TOLOWER "${_proc}" _proc)
    if(APPLE)
        if(_proc MATCHES "arm64|aarch64")
            set(_plat "macos_arm64")
            set(_libname "libLiteRt.dylib")
        else()
            message(WARNING "LiteRT has no prebuilt runtime for this macOS architecture (${CMAKE_SYSTEM_PROCESSOR}). Building without the backend.")
            return()
        endif()
    elseif(WIN32)
        if(_proc MATCHES "arm64|aarch64")
            message(WARNING "LiteRT has no prebuilt Windows ARM64 runtime. Building without the backend.")
            return()
        endif()
        set(_plat "windows_x86_64")
        set(_libname "libLiteRt.dll")
    else()
        if(_proc MATCHES "arm64|aarch64")
            set(_plat "linux_arm64")
        else()
            set(_plat "linux_x86_64")
        endif()
        set(_libname "libLiteRt.so")
    endif()

    if(DEFINED ENV{NN_LITERT_LIB} AND EXISTS "$ENV{NN_LITERT_LIB}")
        set(_litert_lib "$ENV{NN_LITERT_LIB}")
    elseif(DEFINED NN_LITERT_LIB AND EXISTS "${NN_LITERT_LIB}")
        set(_litert_lib "${NN_LITERT_LIB}")
    else()
        set(_lib_dir "${CMAKE_BINARY_DIR}/_deps/litert-${_litert_ver}-${_plat}/lib")
        set(_litert_lib "${_lib_dir}/${_libname}")
        if(NOT EXISTS "${_litert_lib}")
            nn_litert_download(
                "https://github.com/google-ai-edge/LiteRT/raw/v${_litert_ver}/litert/prebuilt/${_plat}/${_libname}"
                "${_litert_lib}")
            if(NN_LITERT_DOWNLOAD_ERROR)
                message(WARNING "LiteRT runtime download failed: ${NN_LITERT_DOWNLOAD_ERROR}. Building without the backend.")
                return()
            endif()
        endif()
        if(EXISTS "${_litert_lib}")
            file(SIZE "${_litert_lib}" _lib_sz)
            if(_lib_sz LESS 10000)
                file(REMOVE "${_litert_lib}")
                nn_litert_download(
                    "https://media.githubusercontent.com/media/google-ai-edge/LiteRT/v${_litert_ver}/litert/prebuilt/${_plat}/${_libname}"
                    "${_litert_lib}")
                if(NN_LITERT_DOWNLOAD_ERROR)
                    message(WARNING "LiteRT LFS runtime download failed: ${NN_LITERT_DOWNLOAD_ERROR}. Building without the backend.")
                    return()
                endif()
            endif()
        endif()
    endif()

    if(NOT EXISTS "${_litert_lib}")
        message(WARNING "LiteRT runtime library missing. Building without the backend.")
        return()
    endif()

    set(_gen "${CMAKE_BINARY_DIR}/_deps/litert_gen")
    file(MAKE_DIRECTORY "${_gen}/litert/build_common")
    file(WRITE "${_gen}/litert/build_common/build_config.h" [=[
#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_CONFIG_DISABLE_GPU 1
#define LITERT_BUILD_CONFIG_DISABLE_NPU 1
#if LITERT_BUILD_CONFIG_DISABLE_GPU
#define LITERT_DISABLE_GPU
#endif
#if LITERT_BUILD_CONFIG_DISABLE_NPU
#define LITERT_DISABLE_NPU
#endif
#endif
]=])

    if(WIN32)
        get_filename_component(_lib_dir "${_litert_lib}" DIRECTORY)
        set(_implib "${_lib_dir}/libLiteRt.lib")
        nn_litert_msvc_implib("${_litert_lib}" "${_implib}")
        if(NOT EXISTS "${_implib}")
            message(WARNING "LiteRT Windows import library was not created. Building without the backend.")
            return()
        endif()
    endif()

    add_library(nn_litert SHARED IMPORTED GLOBAL)
    if(WIN32)
        set_target_properties(nn_litert PROPERTIES
            IMPORTED_IMPLIB "${_implib}"
            IMPORTED_LOCATION "${_litert_lib}"
        )
        set(NN_LITERT_DLL "${_litert_lib}" PARENT_SCOPE)
    else()
        set_target_properties(nn_litert PROPERTIES
            IMPORTED_LOCATION "${_litert_lib}"
        )
    endif()

    get_filename_component(NN_LITERT_LIBDIR "${_litert_lib}" DIRECTORY)
    set(NN_LITERT_LIBDIR "${NN_LITERT_LIBDIR}" PARENT_SCOPE)
    set(NN_LITERT_INCLUDE "${_sdk_root}" PARENT_SCOPE)
    set(NN_LITERT_GEN_INCLUDE "${_gen}" PARENT_SCOPE)
    set(NN_LITERT_VERSION "${_litert_ver}" PARENT_SCOPE)
    set(NN_HAS_LITERT TRUE PARENT_SCOPE)
    message(STATUS "LiteRT: ${_litert_lib}")
endfunction()
