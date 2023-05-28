cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(Protobuf_LIBRARIES
    ${CMAKE_SOURCE_DIR}/3rdparty/protobuf-3.21.4/libs/${ARCH}/libprotobuf-lite.a
)

add_library(quant_transformer_weight STATIC quant_transformer_weight.cc
                                            quant_transformer.pb.cc)
target_link_libraries(quant_transformer_weight PUBLIC ${Protobuf_LIBRARIES})
if(${CMAKE_SYSTEM_NAME} STREQUAL "Android")
  target_link_libraries(quant_transformer_weight PRIVATE log)
endif()
target_link_libraries(quant_transformer_weight PUBLIC ${Protobuf_LIBRARIES})
