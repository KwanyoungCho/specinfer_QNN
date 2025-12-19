set(TARGET llama-simple-qnn)

# QNN SDK root; allow override via env or cache
if(NOT DEFINED QNN_SDK_ROOT)
  if(DEFINED ENV{QNN_SDK_ROOT})
    set(QNN_SDK_ROOT $ENV{QNN_SDK_ROOT})
  else()
    set(QNN_SDK_ROOT "/home/chokwans99/QNN_SDK/qairt/2.37.1.250807")
  endif()
endif()
message(STATUS "Using QNN_SDK_ROOT=${QNN_SDK_ROOT}")

add_executable(${TARGET} simple-qnn.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE
    common
    llama
    qnn_core
    ${CMAKE_THREAD_LIBS_INIT}
)
target_include_directories(${TARGET} PRIVATE
    ${QNN_SDK_ROOT}/include/QNN
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
