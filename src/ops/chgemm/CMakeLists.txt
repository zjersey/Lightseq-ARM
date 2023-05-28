cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
enable_language(ASM)

file(GLOB PATH_S *.S)
set(SOURCES MMult_4x8_22.c ${PATH_S})

add_library(chgemm ${SOURCES})
