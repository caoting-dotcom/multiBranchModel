FROM tensorflow/build:latest-python3.11 AS tflite

ENV ANDROID_DEV_HOME /android
RUN mkdir -p ${ANDROID_DEV_HOME}

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jdk

# Install Android SDK.
ENV ANDROID_SDK_FILENAME commandlinetools-linux-6858069_latest.zip
ENV ANDROID_SDK_URL https://dl.google.com/android/repository/${ANDROID_SDK_FILENAME}
ENV ANDROID_API_LEVEL 23
ENV ANDROID_NDK_API_LEVEL 21
# Build Tools Version liable to change.
ENV ANDROID_BUILD_TOOLS_VERSION 31.0.0
ENV ANDROID_SDK_HOME ${ANDROID_DEV_HOME}/sdk
RUN mkdir -p ${ANDROID_SDK_HOME}/cmdline-tools
ENV PATH ${PATH}:${ANDROID_SDK_HOME}/cmdline-tools/latest/bin:${ANDROID_SDK_HOME}/platform-tools
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_SDK_URL} && \
    unzip ${ANDROID_SDK_FILENAME} -d /tmp && \
    mv /tmp/cmdline-tools ${ANDROID_SDK_HOME}/cmdline-tools/latest && \
    rm ${ANDROID_SDK_FILENAME}

# Install Android NDK.
ENV ANDROID_NDK_FILENAME android-ndk-r21e-linux-x86_64.zip
ENV ANDROID_NDK_URL https://dl.google.com/android/repository/${ANDROID_NDK_FILENAME}
ENV ANDROID_NDK_HOME ${ANDROID_DEV_HOME}/ndk
ENV PATH ${PATH}:${ANDROID_NDK_HOME}
RUN cd ${ANDROID_DEV_HOME} && \
    wget -q ${ANDROID_NDK_URL} && \
    unzip ${ANDROID_NDK_FILENAME} -d ${ANDROID_DEV_HOME} && \
    rm ${ANDROID_NDK_FILENAME} && \
    bash -c "ln -s ${ANDROID_DEV_HOME}/android-ndk-* ${ANDROID_NDK_HOME}"

# Make android ndk executable to all users.
RUN chmod -R go=u ${ANDROID_DEV_HOME} && \
    yes | sdkmanager \
        "build-tools;${ANDROID_BUILD_TOOLS_VERSION}" \
        "platform-tools" \
        "platforms;android-${ANDROID_API_LEVEL}"

# Build benchmark_model
WORKDIR /workspace/tensorflow
COPY src/tensorflow .tf_configure.bazelrc ./
RUN bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model


FROM mosaicml/pytorch_vision:1.13.1_cu117-python3.10-ubuntu20.04
COPY --from=tflite /android /workspace/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /android/
COPY --from=tflite /workspace/tensorflow/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /android/runtime/
COPY android_runtime /android/runtime

WORKDIR /app/mbm-pycls
COPY src/mbm-pycls ./
RUN pip install -r requirements.txt && \
    chmod 744 ./tools/*.py && \
    apt remove mlnx-tools -y && \
    python setup.py develop --user

RUN pip install pure-python-adb && \
    echo "export PATH=/android/sdk/platform-tools:/android/sdk/cmdline-tools/latest/bin:\$PATH" >> /root/.bashrc

WORKDIR /workspace
COPY src/tools ./
