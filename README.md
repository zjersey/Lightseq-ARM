# Lightseq-ARM
Source code of paper "MobileNMT: Enabling Translation in 15MB and 30ms".

# Build and Run
```shell
bash compile-android.sh
export RUN_DIR=/data/local/tmp
adb push run.sh $RUN_DIR
adb shell sh $RUN_DIR/run.sh
```
# Acknowledgement
Lightseq-ARM refers to the following projects:
* [LightSeq](https://github.com/bytedance/lightseq)
* [chgemm](https://github.com/tpoisonooo/chgemm)
* [tvm](https://github.com/apache/tvm)