# Locate or download a prebuilt ONNX Runtime SDK and create imported target nn::onnxruntime.
# Sets NN_HAS_ONNXRUNTIME in the parent scope.

function(nn_setup_onnxruntime)
    set(NN_HAS_ONNXRUNTIME FALSE PARENT_SCOPE)
    if(NOT NN_ENABLE_ONNXRUNTIME)
        return()
    endif()

    set(_ort_ver "1.20.1")
    if(DEFINED ENV{NN_ONNXRUNTIME_ROOT} AND EXISTS "$ENV{NN_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        set(_ort_root "$ENV{NN_ONNXRUNTIME_ROOT}")
    elseif(DEFINED NN_ONNXRUNTIME_ROOT AND EXISTS "${NN_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        set(_ort_root "${NN_ONNXRUNTIME_ROOT}")
    else()
        set(_proc "${CMAKE_SYSTEM_PROCESSOR}")
        string(TOLOWER "${_proc}" _proc)
        if(APPLE)
            if(_proc MATCHES "arm64|aarch64")
                set(_pkg "onnxruntime-osx-arm64-${_ort_ver}")
            else()
                set(_pkg "onnxruntime-osx-x86_64-${_ort_ver}")
            endif()
            set(_ext "tgz")
        elseif(WIN32)
            if(_proc MATCHES "arm64|aarch64")
                set(_pkg "onnxruntime-win-arm64-${_ort_ver}")
            else()
                set(_pkg "onnxruntime-win-x64-${_ort_ver}")
            endif()
            set(_ext "zip")
        else()
            if(_proc MATCHES "arm64|aarch64")
                set(_pkg "onnxruntime-linux-aarch64-${_ort_ver}")
            else()
                set(_pkg "onnxruntime-linux-x64-${_ort_ver}")
            endif()
            set(_ext "tgz")
        endif()

        set(_url "https://github.com/microsoft/onnxruntime/releases/download/v${_ort_ver}/${_pkg}.${_ext}")
        set(_dest "${CMAKE_BINARY_DIR}/_deps/${_pkg}")
        set(_archive "${CMAKE_BINARY_DIR}/_deps/${_pkg}.${_ext}")
        if(NOT EXISTS "${_dest}/include/onnxruntime_cxx_api.h")
            message(STATUS "Downloading ONNX Runtime ${_ort_ver} from ${_url}")
            file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
            file(DOWNLOAD "${_url}" "${_archive}"
                SHOW_PROGRESS
                STATUS _dl
                TIMEOUT 120
                TLS_VERIFY ON
            )
            list(GET _dl 0 _dl_code)
            if(NOT _dl_code EQUAL 0)
                list(GET _dl 1 _dl_msg)
                message(WARNING "ONNX Runtime download failed: ${_dl_msg}. Building without the onnxruntime backend.")
                return()
            endif()
            file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${CMAKE_BINARY_DIR}/_deps")
        endif()
        if(EXISTS "${_dest}/include/onnxruntime_cxx_api.h")
            set(_ort_root "${_dest}")
        elseif(EXISTS "${CMAKE_BINARY_DIR}/_deps/${_pkg}/include/onnxruntime_cxx_api.h")
            set(_ort_root "${CMAKE_BINARY_DIR}/_deps/${_pkg}")
        else()
            message(WARNING "ONNX Runtime archive extracted but headers were not found. Building without the backend.")
            return()
        endif()
    endif()

    find_path(NN_ORT_INCLUDE onnxruntime_cxx_api.h
        PATHS "${_ort_root}/include" NO_DEFAULT_PATH)
    if(WIN32)
        find_library(NN_ORT_LIB onnxruntime
            PATHS "${_ort_root}/lib" "${_ort_root}/lib/Release" NO_DEFAULT_PATH)
        find_file(NN_ORT_DLL onnxruntime.dll
            PATHS "${_ort_root}/lib" "${_ort_root}/bin" "${_ort_root}/lib/Release" NO_DEFAULT_PATH)
    else()
        find_library(NN_ORT_LIB onnxruntime
            PATHS "${_ort_root}/lib" NO_DEFAULT_PATH)
    endif()

    if(NOT NN_ORT_INCLUDE OR NOT NN_ORT_LIB)
        message(WARNING "ONNX Runtime headers or library missing under ${_ort_root}. Building without the backend.")
        return()
    endif()

    add_library(nn_onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(nn_onnxruntime PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${NN_ORT_INCLUDE}"
        IMPORTED_LOCATION "${NN_ORT_LIB}"
    )
    if(WIN32 AND NN_ORT_DLL)
        set_target_properties(nn_onnxruntime PROPERTIES
            IMPORTED_IMPLIB "${NN_ORT_LIB}"
            IMPORTED_LOCATION "${NN_ORT_DLL}"
        )
        set(NN_ORT_DLL "${NN_ORT_DLL}" PARENT_SCOPE)
    endif()
    get_filename_component(NN_ORT_LIBDIR "${NN_ORT_LIB}" DIRECTORY)
    set(NN_ORT_LIBDIR "${NN_ORT_LIBDIR}" PARENT_SCOPE)
    set(NN_ORT_INCLUDE "${NN_ORT_INCLUDE}" PARENT_SCOPE)
    set(NN_HAS_ONNXRUNTIME TRUE PARENT_SCOPE)
    message(STATUS "ONNX Runtime: ${NN_ORT_LIB}")
endfunction()
