file(GLOB Yolo_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
file(GLOB Yolo_GPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)

# Note(ilijar): Since Detectron ops currently have no
# CPU implementation, we only build GPU ops for now.
if (USE_CUDA)
  CUDA_ADD_LIBRARY(
      caffe2_yolo_ops_gpu SHARED
      ${Yolo_CPU_SRCS}
      ${Yolo_GPU_SRCS})

  target_link_libraries(caffe2_yolo_ops_gpu caffe2_gpu)
  install(TARGETS caffe2_yolo_ops_gpu DESTINATION lib)
endif()
