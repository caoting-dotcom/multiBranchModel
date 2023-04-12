This repo includes the implementation for paper "NN-Stretch: Automatic Neural Network Branching for
Parallel Inference on Heterogeneous Multi-Processors"

# End-to-End Pipeline

The following commands run on a host computer connected with an Android phone. The host computer OS prefers Ubuntu, since Docker on Windows may invoke some system issues. 

We also provide an **end-to-end video** to install the environment and run the pipeline. Check it [here](tutorial/end2end-pipeline.mp4)!

## Pull docker

We provide a docker container for the artifact. It can be directly downloaded by:

```
docker image pull kalineid/nn_stretch
```

## Build Docker (optional)

If you want to build the docker yourself, run the following command. Or you can skip this step to the next step.
```
docker build -t kalineid/nn_stretch .
```

## Evaluation

First, start the container with:

```
docker run -it -v $(pwd)/configs:/data --net host --name stretch-ae kalineid/nn_stretch /bin/bash
```

You should enter the container after the above command. Type in docker:

```
adb devices
```

to check if your Android device is successfully connected to the container.

Then, in docker:

```
python run_pipeline.py --configs /data
```

This script will automatically generate the model for the yaml configs in /data and profile the models on the Android device. The execution can consume over 30 mins.

If you see this on terminal:

```
Results generated at /workspace/data/result.csv
```

Then the script is completed successfully. Go and check the result at `/workspace/data/result.csv` in the container. Sample outputs are provided at [this folder](sample_output).

### If your host is windows

If your host is windows, you need to install adb first, then run on the host:

```
adb -a -P 5037 nodaemon server
```

And start the container with:

```
docker run -it -v absolute_path_to_configs:/data --name stretch-ae kalineid/nn_stretch /bin/bash
```

Then in docker:
```
python run_pipeline.py --configs /data --adb_host host.docker.internal
```

## Trouble shooting

### Could not create Hexagon delegate

It means your android smartphone is not equipped with a Hexagon DSP or your Hexagon architecture is unsupported by tensorflow. Currently, tensorflow only support Hexagon of architecture 680/682/685/690.

### Segmentation fault

This error is most likely caused by no root access. Root is required for energy measurement. If you don't want to profile the energy, please pass `--no_root` to `run_pipeline.py`.

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
