cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(OPS embedding.cc layernorm.cc gemm.cc quantize.cc attention.cc
        activation_add.cc)
add_library(lightseq_ops ${OPS})

target_link_libraries(lightseq_ops chgemm)
