#!/usr/bin/bash
rm -rf build-linux-arm
mkdir -p build-linux-arm
cd build-linux-arm


cmake -DCMAKE_SYSTEM_NAME=Linux \
	-DCMAKE_SYSTEM_PROCESSOR=arm \
	-DCMAKE_C_COMPILER="aarch64-linux-gnu-gcc" \
	-DCMAKE_CXX_COMPILER="aarch64-linux-gnu-g++" \
	-DARCH="linux-aarch64" \
	"$@" ..

make -j4
