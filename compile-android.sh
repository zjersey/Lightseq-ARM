#!/usr/bin/bash

mkdir -p build-android
rm -rf build-android/*
cd build-android

NDK=/usr/local/android-ndk-r25b

cmake -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
	-DANDROID_ABI="arm64-v8a" \
	-DANDROID_NDK=$NDK \
	-DANDROID_PLATFORM=android-22 \
	-DARCH="arm64-v8a" \
	"$@" ..

make -j4
