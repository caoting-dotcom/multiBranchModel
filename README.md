## Build Docker (optional)

```
docker build -t kalineid/nn_stretch .
```

## Evaluation

```
docker run -it -v configs:/data --privileged -v /dev/bus/usb:/dev/bus/usb --name stretch-ae kalineid/nn_stretch /bin/bash
```

Then, in docker:
```
python run_pipeline.py --configs /data
```

### If your host is windows

If your host is windows, you need to install adb first, then running on host:

```
adb -a -P 5037 nodaemon server
```

And start the container with:

```
docker run -it -v configs:/data --name stretch-ae kalineid/nn_stretch /bin/bash
```

Then in docker:
```
python run_pipeline.py --configs /data --adb_host host.docker.internal
```

## Notes

## Code structure

- src/mbm-pycls: The model generation and training are based on [pycls](https://github.com/facebookresearch/pycls). It is included in this repo as a submodule. Our modifications involve commits from [fc4fc5](https://github.com/kaleid-liner/mbm-pycls/commit/fc4fc503e7d0f4f9fbf369b068d45af191d8c5e9), mainly including:
    - pycls/ir: build pytorch/tensorflow models from multi-branch configs.
    - pycls/models: define extra `nn.Module`s to build the models

- src/tensorflow: Implement the inference for multi-branch models based on tflite. It is included in this repo as a submodule. Our modifications involve commits from [be3db60](https://github.com/kaleid-liner/tensorflow/commit/be3db600d6fb9fcc185c5537f3cbbef3d558a0d0), mainly including:
    - tensorflow/lite/core/subgraph.cc: modify the inference procedure; partition the model into different devices using the name tags
    - tensorflow/lite/graph_info.cc: extra utils for model partition
    - tensorflow/lite/c/common.h: change the api of `TfLiteContext`
    - tensorflow/lite/delegates/gpu: implement api for asynchronous gpu execution and memory mapping
    - tensorflow/lite/energy_profiler.cc: monitor energy consumption

  The principle of our modifications is maintaining the function and structure of the code-base without harming its generability.

- src/tools: python script to run the pipeline.
