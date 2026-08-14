cmake_minimum_required(VERSION 3.20)
project(nn
    VERSION 0.1.0
    DESCRIPTION "Command-line neural-network engineering toolkit"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NN_ENABLE_ONNX "Enable ONNX format reader" ON)
option(NN_ENABLE_TFLITE "Enable TFLite/LiteRT format reader" ON)
option(NN_ENABLE_GGUF "Enable GGUF format reader" ON)
option(NN_ENABLE_SAFETENSORS "Enable SafeTensors format reader" ON)
option(NN_ENABLE_PYTORCH "Enable PyTorch artifact inspection (safe paths only)" ON)
option(NN_ENABLE_EXECUTORCH "Enable ExecuTorch .pte inspection" ON)
option(NN_ENABLE_COREML "Enable Core ML inspection" ON)
option(NN_ENABLE_OPENVINO "Enable OpenVINO IR inspection" ON)
option(NN_ENABLE_TENSORRT "Enable TensorRT engine metadata inspection" ON)
option(NN_ENABLE_NCNN "Enable NCNN inspection" ON)
option(NN_ENABLE_MNN "Enable MNN inspection" ON)
option(NN_ENABLE_TFJS "Enable TensorFlow.js inspection" ON)
option(NN_ENABLE_LEGACY "Enable legacy format readers (Caffe, Darknet, MXNet, Paddle)" ON)
option(NN_ENABLE_KERAS "Enable Keras/HDF5 inspection" ON)
option(NN_ENABLE_TENSORFLOW "Enable TensorFlow SavedModel inspection" ON)

option(NN_ENABLE_ONNXRUNTIME "Enable ONNX Runtime backend (downloads SDK if needed)" ON)
option(NN_ENABLE_LITERT_RUNTIME "Enable LiteRT/TFLite runtime backend (downloads SDK if needed)" ON)
option(NN_ENABLE_WERROR "Treat warnings as errors" ON)
option(NN_BUILD_TESTS "Build unit and integration tests" ON)
option(NN_BUILD_FUZZERS "Build libFuzzer targets" OFF)
option(NN_REQUIRE_RUNTIMES "Fail configure if enabled runtimes cannot be downloaded" OFF)

set(NN_SANITIZER "" CACHE STRING "Sanitizer: address, undefined, thread, or address,undefined")

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
if(MSVC)
    add_compile_options(/W4 /permissive- /Zc:__cplusplus)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    if(NN_ENABLE_WERROR)
        add_compile_options(/WX)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(NN_ENABLE_WERROR)
        add_compile_options(-Werror)
    endif()
endif()

if(NN_SANITIZER)
    if(NOT MSVC)
        string(REPLACE "," ";" _nn_san_list "${NN_SANITIZER}")
        foreach(_san IN LISTS _nn_san_list)
            add_compile_options(-fsanitize=${_san} -fno-omit-frame-pointer)
            add_link_options(-fsanitize=${_san})
        endforeach()
    endif()
endif()

# ---------------------------------------------------------------------------
# Version / build identity
# ---------------------------------------------------------------------------
find_package(Git QUIET)
set(NN_GIT_COMMIT "unknown")
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NN_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT NN_GIT_COMMIT)
        set(NN_GIT_COMMIT "unknown")
    endif()
endif()

set(NN_COMPILER_ID "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

configure_file(
    ${CMAKE_SOURCE_DIR}/src/util/version_info.cpp.in
    ${CMAKE_BINARY_DIR}/generated/version_info.cpp
    @ONLY
)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(NN_LIB_SOURCES
    src/util/error.cpp
    src/util/file.cpp
    src/util/mmap.cpp
    src/util/hash.cpp
    src/util/json.cpp
    src/util/yaml.cpp
    src/util/timer.cpp
    src/util/terminal.cpp
    src/util/logging.cpp
    src/util/binary.cpp
    src/util/protobuf.cpp
    src/util/flatbuffer.cpp
    src/util/format_text.cpp
    src/util/npy.cpp
    src/util/zip.cpp
    src/util/overflow.cpp
    src/ir/datatype.cpp
    src/ir/shape.cpp
    src/ir/quantization.cpp
    src/ir/tensor.cpp
    src/ir/node.cpp
    src/ir/graph.cpp
    src/ir/model.cpp
    src/ir/attribute.cpp
    src/ir/operator.cpp
    src/ir/canonical.cpp
    src/formats/fileset.cpp
    src/formats/registry.cpp
    src/formats/detect.cpp
    src/analysis/shape.cpp
    src/analysis/compute.cpp
    src/analysis/lifetime.cpp
    src/analysis/memory.cpp
    src/analysis/quantization.cpp
    src/analysis/sparsity.cpp
    src/analysis/graph_match.cpp
    src/analysis/diff.cpp
    src/analysis/lint.cpp
    src/analysis/hash_model.cpp
    src/analysis/inspect.cpp
    src/analysis/optimize.cpp
    src/analysis/compat.cpp
    src/analysis/convert.cpp
    src/runtime/registry.cpp
    src/runtime/tensor_io.cpp
    src/runtime/reference/backend.cpp
    src/runtime/reference/ops.cpp
    src/targets/target.cpp
    src/targets/database.cpp
    src/targets/compatibility.cpp
    src/targets/performance.cpp
    src/targets/partition.cpp
    src/cli/options.cpp
    src/cli/help.cpp
    src/cli/output.cpp
    src/cli/command.cpp
    src/cli/config.cpp
    src/commands/inspect.cpp
    src/commands/ops.cpp
    src/commands/tensors.cpp
    src/commands/io.cpp
    src/commands/metadata.cpp
    src/commands/hash.cpp
    src/commands/diff.cpp
    src/commands/graph.cpp
    src/commands/memory.cpp
    src/commands/compute.cpp
    src/commands/quant.cpp
    src/commands/sparsity.cpp
    src/commands/lint.cpp
    src/commands/run.cpp
    src/commands/compare.cpp
    src/commands/test.cpp
    src/commands/validate.cpp
    src/commands/regression.cpp
    src/commands/benchmark.cpp
    src/commands/profile.cpp
    src/commands/compat.cpp
    src/commands/target.cpp
    src/commands/partition.cpp
    src/commands/convert.cpp
    src/commands/optimize.cpp
    src/commands/extract.cpp
    src/commands/canonicalize.cpp
    src/commands/bisect.cpp
    src/commands/doctor.cpp
    src/commands/formats.cpp
    src/commands/backends.cpp
    src/commands/targets.cpp
    src/commands/config.cpp
    src/commands/version.cpp
    src/commands/help.cpp
    ${CMAKE_BINARY_DIR}/generated/version_info.cpp
)

if(NN_ENABLE_ONNX)
    list(APPEND NN_LIB_SOURCES
        src/formats/onnx/reader.cpp
        src/formats/onnx/writer.cpp
        src/formats/onnx/proto.cpp
    )
endif()
if(NN_ENABLE_TFLITE)
    list(APPEND NN_LIB_SOURCES src/formats/tflite/reader.cpp)
endif()
if(NN_ENABLE_GGUF)
    list(APPEND NN_LIB_SOURCES src/formats/gguf/reader.cpp)
endif()
if(NN_ENABLE_SAFETENSORS)
    list(APPEND NN_LIB_SOURCES src/formats/safetensors/reader.cpp)
endif()
if(NN_ENABLE_PYTORCH)
    list(APPEND NN_LIB_SOURCES src/formats/pytorch/reader.cpp)
endif()
if(NN_ENABLE_EXECUTORCH)
    list(APPEND NN_LIB_SOURCES src/formats/executorch/reader.cpp)
endif()
if(NN_ENABLE_COREML)
    list(APPEND NN_LIB_SOURCES src/formats/coreml/reader.cpp)
endif()
if(NN_ENABLE_OPENVINO)
    list(APPEND NN_LIB_SOURCES src/formats/openvino/reader.cpp)
endif()
if(NN_ENABLE_TENSORRT)
    list(APPEND NN_LIB_SOURCES src/formats/tensorrt/reader.cpp)
endif()
if(NN_ENABLE_NCNN)
    list(APPEND NN_LIB_SOURCES src/formats/ncnn/reader.cpp)
endif()
if(NN_ENABLE_MNN)
    list(APPEND NN_LIB_SOURCES src/formats/mnn/reader.cpp)
endif()
if(NN_ENABLE_TFJS)
    list(APPEND NN_LIB_SOURCES src/formats/tfjs/reader.cpp)
endif()
if(NN_ENABLE_LEGACY)
    list(APPEND NN_LIB_SOURCES
        src/formats/legacy/caffe.cpp
        src/formats/legacy/darknet.cpp
        src/formats/legacy/mxnet.cpp
        src/formats/legacy/paddle.cpp
    )
endif()
if(NN_ENABLE_KERAS)
    list(APPEND NN_LIB_SOURCES src/formats/keras/reader.cpp)
endif()
if(NN_ENABLE_TENSORFLOW)
    list(APPEND NN_LIB_SOURCES src/formats/tensorflow/reader.cpp)
endif()

# ---------------------------------------------------------------------------
# Optional ONNX Runtime SDK
# ---------------------------------------------------------------------------
include(${CMAKE_SOURCE_DIR}/cmake/onnxruntime.cmake)
nn_setup_onnxruntime()
if(NN_HAS_ONNXRUNTIME AND NN_ENABLE_ONNX)
    list(APPEND NN_LIB_SOURCES src/runtime/onnxruntime/backend.cpp)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/litert.cmake)
nn_setup_litert()
if(NN_HAS_LITERT AND NN_ENABLE_TFLITE)
    list(APPEND NN_LIB_SOURCES src/runtime/litert/backend.cpp)
endif()
if(NN_REQUIRE_RUNTIMES)
    if(NN_ENABLE_ONNXRUNTIME AND NOT NN_HAS_ONNXRUNTIME)
        message(FATAL_ERROR "ONNX Runtime was required (NN_REQUIRE_RUNTIMES) but could not be configured")
    endif()
    if(NN_ENABLE_LITERT_RUNTIME AND NOT NN_HAS_LITERT)
        message(FATAL_ERROR "LiteRT was required (NN_REQUIRE_RUNTIMES) but could not be configured")
    endif()
endif()

add_library(nn_core STATIC ${NN_LIB_SOURCES})
add_library(nn::core ALIAS nn_core)

target_include_directories(nn_core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_BINARY_DIR}/generated
)

target_compile_definitions(nn_core
    PUBLIC
        NN_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
        NN_VERSION_MINOR=${PROJECT_VERSION_MINOR}
        NN_VERSION_PATCH=${PROJECT_VERSION_PATCH}
        $<$<BOOL:${NN_ENABLE_ONNX}>:NN_ENABLE_ONNX=1>
        $<$<BOOL:${NN_ENABLE_TFLITE}>:NN_ENABLE_TFLITE=1>
        $<$<BOOL:${NN_ENABLE_GGUF}>:NN_ENABLE_GGUF=1>
        $<$<BOOL:${NN_ENABLE_SAFETENSORS}>:NN_ENABLE_SAFETENSORS=1>
        $<$<BOOL:${NN_ENABLE_PYTORCH}>:NN_ENABLE_PYTORCH=1>
        $<$<BOOL:${NN_ENABLE_EXECUTORCH}>:NN_ENABLE_EXECUTORCH=1>
        $<$<BOOL:${NN_ENABLE_COREML}>:NN_ENABLE_COREML=1>
        $<$<BOOL:${NN_ENABLE_OPENVINO}>:NN_ENABLE_OPENVINO=1>
        $<$<BOOL:${NN_ENABLE_TENSORRT}>:NN_ENABLE_TENSORRT=1>
        $<$<BOOL:${NN_ENABLE_NCNN}>:NN_ENABLE_NCNN=1>
        $<$<BOOL:${NN_ENABLE_MNN}>:NN_ENABLE_MNN=1>
        $<$<BOOL:${NN_ENABLE_TFJS}>:NN_ENABLE_TFJS=1>
        $<$<BOOL:${NN_ENABLE_LEGACY}>:NN_ENABLE_LEGACY=1>
        $<$<BOOL:${NN_ENABLE_KERAS}>:NN_ENABLE_KERAS=1>
        $<$<BOOL:${NN_ENABLE_TENSORFLOW}>:NN_ENABLE_TENSORFLOW=1>
)

if(NN_HAS_ONNXRUNTIME AND NN_ENABLE_ONNX)
    target_compile_definitions(nn_core PUBLIC NN_HAS_ONNXRUNTIME=1)
    target_link_libraries(nn_core PUBLIC nn_onnxruntime)
    target_include_directories(nn_core SYSTEM PRIVATE ${NN_ORT_INCLUDE})
    if(NN_ORT_LIBDIR)
        set_property(TARGET nn_core APPEND PROPERTY BUILD_RPATH "${NN_ORT_LIBDIR}")
        set_property(TARGET nn_core APPEND PROPERTY INSTALL_RPATH "${NN_ORT_LIBDIR}")
    endif()
endif()

if(NN_HAS_LITERT AND NN_ENABLE_TFLITE)
    target_compile_definitions(nn_core PUBLIC NN_HAS_LITERT=1 NN_LITERT_VERSION="${NN_LITERT_VERSION}")
    target_link_libraries(nn_core PUBLIC nn_litert)
    target_include_directories(nn_core SYSTEM PRIVATE ${NN_LITERT_INCLUDE} ${NN_LITERT_GEN_INCLUDE})
    if(NN_LITERT_LIBDIR)
        set_property(TARGET nn_core APPEND PROPERTY BUILD_RPATH "${NN_LITERT_LIBDIR}")
        set_property(TARGET nn_core APPEND PROPERTY INSTALL_RPATH "${NN_LITERT_LIBDIR}")
    endif()
endif()

if(UNIX)
    target_compile_definitions(nn_core PRIVATE NN_PLATFORM_POSIX=1)
endif()
if(APPLE)
    target_compile_definitions(nn_core PRIVATE NN_PLATFORM_APPLE=1)
endif()
if(WIN32)
    target_compile_definitions(nn_core PRIVATE NN_PLATFORM_WINDOWS=1)
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(nn src/main.cpp)
target_link_libraries(nn PRIVATE nn::core)
set_target_properties(nn PROPERTIES OUTPUT_NAME nn)
if(NN_HAS_ONNXRUNTIME AND NN_ORT_LIBDIR)
    set_property(TARGET nn APPEND PROPERTY BUILD_RPATH "${NN_ORT_LIBDIR}")
    set_property(TARGET nn APPEND PROPERTY INSTALL_RPATH "${NN_ORT_LIBDIR}")
endif()
if(NN_HAS_LITERT AND NN_LITERT_LIBDIR)
    set_property(TARGET nn APPEND PROPERTY BUILD_RPATH "${NN_LITERT_LIBDIR}")
    set_property(TARGET nn APPEND PROPERTY INSTALL_RPATH "${NN_LITERT_LIBDIR}")
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(nn_core PRIVATE pthread)
endif()

install(TARGETS nn RUNTIME DESTINATION bin)
install(DIRECTORY include/nn DESTINATION include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/docs/man/
    DESTINATION share/man/man1
    FILES_MATCHING PATTERN "*.1"
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(NN_BUILD_TESTS)
    enable_testing()

    add_executable(nn_unit_tests
        tests/unit/test_main.cpp
        tests/unit/test_datatype.cpp
        tests/unit/test_shape.cpp
        tests/unit/test_overflow.cpp
        tests/unit/test_hash.cpp
        tests/unit/test_json.cpp
        tests/unit/test_cli_options.cpp
        tests/unit/test_compute.cpp
        tests/unit/test_lifetime.cpp
        tests/unit/test_memory.cpp
        tests/unit/test_operator.cpp
        tests/unit/test_canonical.cpp
        tests/unit/test_quantization.cpp
        tests/unit/test_sparsity.cpp
        tests/unit/test_diff.cpp
        tests/unit/test_protobuf.cpp
        tests/unit/test_npy.cpp
        tests/unit/test_compare_align.cpp
        tests/unit/test_target.cpp
        tests/unit/test_runtime.cpp
        tests/unit/test_optimize.cpp
    )
    target_link_libraries(nn_unit_tests PRIVATE nn::core)
    target_include_directories(nn_unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)

    add_executable(nn_integration_tests
        tests/integration/test_main.cpp
        tests/integration/test_onnx.cpp
        tests/integration/test_tflite.cpp
        tests/integration/test_gguf.cpp
        tests/integration/test_safetensors.cpp
        tests/integration/test_inspect.cpp
        tests/integration/test_commands.cpp
        tests/integration/test_run.cpp
        tests/integration/golden_cli.cpp
    )
    target_link_libraries(nn_integration_tests PRIVATE nn::core)
    target_include_directories(nn_integration_tests PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_SOURCE_DIR}/include
    )
    target_compile_definitions(nn_integration_tests PRIVATE
        NN_TEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}/tests"
        NN_TEST_BINARY_DIR="${CMAKE_BINARY_DIR}"
        NN_TEST_NN_EXE="$<TARGET_FILE:nn>"
    )
    add_dependencies(nn_integration_tests nn)

    add_test(NAME unit COMMAND nn_unit_tests)
    add_test(NAME integration COMMAND nn_integration_tests)
    add_test(NAME nn_help COMMAND nn)
    add_test(NAME nn_formats COMMAND nn formats)
    add_test(NAME nn_doctor COMMAND nn doctor)
    add_test(NAME nn_version COMMAND nn --version)

    set_tests_properties(nn_help nn_formats nn_doctor nn_version PROPERTIES
        PASS_REGULAR_EXPRESSION "."
    )

    if(NN_HAS_ONNXRUNTIME AND NN_ORT_LIBDIR)
        set_property(TARGET nn_unit_tests APPEND PROPERTY BUILD_RPATH "${NN_ORT_LIBDIR}")
        set_property(TARGET nn_integration_tests APPEND PROPERTY BUILD_RPATH "${NN_ORT_LIBDIR}")
    endif()
    if(NN_HAS_LITERT AND NN_LITERT_LIBDIR)
        set_property(TARGET nn_unit_tests APPEND PROPERTY BUILD_RPATH "${NN_LITERT_LIBDIR}")
        set_property(TARGET nn_integration_tests APPEND PROPERTY BUILD_RPATH "${NN_LITERT_LIBDIR}")
    endif()
    if(WIN32 AND NN_HAS_ONNXRUNTIME AND NN_ORT_DLL)
        foreach(_tgt nn nn_unit_tests nn_integration_tests)
            add_custom_command(TARGET ${_tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NN_ORT_DLL}"
                        "$<TARGET_FILE_DIR:${_tgt}>")
        endforeach()
    endif()
    if(WIN32 AND NN_HAS_LITERT AND NN_LITERT_DLL)
        foreach(_tgt nn nn_unit_tests nn_integration_tests)
            add_custom_command(TARGET ${_tgt} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${NN_LITERT_DLL}"
                        "$<TARGET_FILE_DIR:${_tgt}>")
        endforeach()
    endif()

    set(NN_MNIST_ONNX "${CMAKE_BINARY_DIR}/testdata/mnist-8.onnx")
    if(NN_HAS_ONNXRUNTIME AND NOT EXISTS "${NN_MNIST_ONNX}")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/testdata")
        message(STATUS "Downloading mnist-8.onnx test fixture")
        file(DOWNLOAD
            "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-8.onnx"
            "${NN_MNIST_ONNX}"
            TIMEOUT 60
            TLS_VERIFY ON
            STATUS _mnist_dl
        )
        list(GET _mnist_dl 0 _mnist_code)
        if(NOT _mnist_code EQUAL 0)
            file(REMOVE "${NN_MNIST_ONNX}")
            message(WARNING "mnist-8.onnx download failed; skipping that fixture")
        endif()
    endif()
    if(EXISTS "${NN_MNIST_ONNX}")
        target_compile_definitions(nn_integration_tests PRIVATE
            NN_TEST_MNIST_ONNX="${NN_MNIST_ONNX}")
    endif()

    set(NN_TFLITE_ADD "${CMAKE_BINARY_DIR}/testdata/add.tflite")
    if(NN_HAS_LITERT AND NOT EXISTS "${NN_TFLITE_ADD}")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/testdata")
        message(STATUS "Downloading add.tflite test fixture")
        file(DOWNLOAD
            "https://raw.githubusercontent.com/tensorflow/tensorflow/v2.18.0/tensorflow/lite/testdata/add.bin"
            "${NN_TFLITE_ADD}"
            TIMEOUT 60
            TLS_VERIFY ON
            STATUS _tflite_dl
        )
        list(GET _tflite_dl 0 _tflite_code)
        if(NOT _tflite_code EQUAL 0)
            file(REMOVE "${NN_TFLITE_ADD}")
            message(WARNING "add.tflite download failed; skipping that fixture")
        endif()
    endif()
    if(EXISTS "${NN_TFLITE_ADD}")
        target_compile_definitions(nn_integration_tests PRIVATE
            NN_TEST_TFLITE_ADD="${NN_TFLITE_ADD}")
    endif()
endif()

if(NN_BUILD_FUZZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    foreach(_fuzz IN ITEMS fuzz_onnx fuzz_gguf fuzz_safetensors fuzz_tflite fuzz_cli)
        add_executable(${_fuzz} tests/fuzz/${_fuzz}.cpp)
        target_link_libraries(${_fuzz} PRIVATE nn::core)
        target_include_directories(${_fuzz} PRIVATE ${CMAKE_SOURCE_DIR}/src)
        target_compile_options(${_fuzz} PRIVATE -fsanitize=fuzzer,address,undefined)
        target_link_options(${_fuzz} PRIVATE -fsanitize=fuzzer,address,undefined)
    endforeach()
endif()
