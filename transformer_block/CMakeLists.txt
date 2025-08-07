cmake_minimum_required(VERSION 3.16)
project("test_model")

#SET(CMAKE_VERBOSE_MAKEFILE on)
#options, these can be pass in by cmake argument
option(USE_CXX11_ABI "USE_CXX11_ABI" OFF)

# 设置为Debug模式以便调试
set(CMAKE_BUILD_TYPE Debug)

# 添加调试信息和不优化，便于gdb调试
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")

if(USE_CXX11_ABI)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(
    $ENV{ATB_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR})

link_directories(
    $ENV{ATB_HOME_PATH}/lib
    $ENV{ASCEND_HOME_PATH}/lib64
    )

set(TEST_MODEL_CXX
    main.cpp
    aclnn/aclnn_gelu_operation.cpp
    aclnn/aclnn_operation_base.cpp
    utils/utils.cpp
    utils/log.cpp
    atb/atb_graph_op.cpp
    model/model.cpp
    memory/memorypool.cpp
    memory/memory_utils.cpp
)

set(TEST_MODEL2_CXX
    main2.cpp
    aclnn/aclnn_gelu_operation.cpp
    aclnn/aclnn_operation_base.cpp
    utils/utils.cpp
    utils/log.cpp
    memory/memorypool.cpp
    memory/memory_utils.cpp
)
file(GLOB ATB_SRC2 "atb/*.cpp")
list(APPEND TEST_MODEL2_CXX ${ATB_SRC2})
file(GLOB MODEL_SRC2 "model/*.cpp")
list(APPEND TEST_MODEL2_CXX ${MODEL_SRC2})



# 列出所有的头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/aclnn
    ${CMAKE_SOURCE_DIR}/atb
    ${CMAKE_SOURCE_DIR}/model
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/memory
)

# add_executable(test_model ${TEST_MODEL_CXX})
add_executable(test_model2 ${TEST_MODEL2_CXX})

# target_link_libraries(test_model PRIVATE atb ascendcl opapi nnopbase atb_graph pthread)
# target_link_libraries(test_model PRIVATE atb ascendcl opapi nnopbase pthread)
target_link_libraries(test_model2 PRIVATE atb ascendcl opapi nnopbase pthread)
