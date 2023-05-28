cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

add_executable(test_op "test_op.cc")
target_link_libraries(test_op lightseq_ops)
target_link_libraries(test_op chgemm)
