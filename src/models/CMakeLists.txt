cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

add_library(quant_transformer_model quant_transformer.cc quant_encoder.cc
                                    quant_decoder.cc)
target_link_libraries(quant_transformer_model PUBLIC quant_transformer_weight
                                                     lightseq_ops utils)

if(BUILD_DYNAMIC)
    add_library(lightseq_model SHARED lightseq_model.cpp)
else()
    add_library(lightseq_model STATIC lightseq_model.cpp)
endif()
target_link_libraries(lightseq_model quant_transformer_model)
