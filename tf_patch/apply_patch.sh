bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model
cp thread_pool.h $(find /root/.cache -name "thread_pool.h" | grep ruy)
bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model
