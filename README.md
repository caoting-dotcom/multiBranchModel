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

## Latency Evaluation

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
python run_pipeline.py --configs /data/pixel/ [--num_threads 2 --core_affinity c0]
```

This script will automatically generate the model for the yaml configs in /data and profile the models on the Android device. The execution can consume over 30 mins. `--num_threads 2 --core_affinity c0` is needed for device similar with Pixel6 (with two big cores).

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
python run_pipeline.py --configs /data/pixel --adb_host host.docker.internal
```

### Use a real input image

If you want to use a real input image for latency measurements, you can specify this argument for `run_pipeline.py`
```
--input_image images/ILSVRC2012_val_00000001.JPEG
```

### Evaluation output

Check the [video](tutorial/end2end-pipeline.mp4) to see the expected stdout output. The final results will be generated at `/workspace/data/result.csv` of the container.

The CSV table are consisted of three columns:
- `cfg`: Path of the config file. E.g., a config named R-50-CD.yaml means the model are generated with two branches, one for CPU and one for DSP. A config named R-50-C.yaml means the model is a baseline model generated for CPU.
- `latency`: Inference latency of the model.
- `energy`: Inference energy of the model. It could be 0 if the device is not rooted or hardware counters are unavailable. It could be less than 0 if the device is connected to the device via USB. To profile the energy accurately, WiFi ADB is recommended.

## Training and Accuracy Evluation

ImageNet is a very large dataset. If you haven't used ImageNet before, you need to first prepare the dataset. If you already have ImageNet on your host computer, you can skip Step 1.

### Step 1: Prepare Data

Download ImageNet from [kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data):
```
kaggle competitions download -c imagenet-object-localization-challenge
```

or image-net with:
```
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```

Then using [this script](src/tools/extract_ILSVRC.sh) to process the data.

After the above commands, the expected structure of `${path_to_imagenet}` should be:
```
 train/
 ├── n01440764
 │   ├── n01440764_10026.JPEG
 │   ├── n01440764_10027.JPEG
 │   ├── ......
 ├── ......
 val/
 ├── n01440764
 │   ├── ILSVRC2012_val_00000293.JPEG
 │   ├── ILSVRC2012_val_00002138.JPEG
 │   ├── ......
 ├── ......
```

### Step 2: Preprocess for FFCV

Our models are trained using [ffcv](https://github.com/libffcv/ffcv) dataloader. This needs extra steps to preprocess the dataset. Follow [the instructions](https://github.com/libffcv/ffcv-imagenet/blob/main/README.md) at [ffcv-imagenet](https://github.com/libffcv/ffcv-imagenet) to get `train.ffcv` and `val.ffcv`. You will need to execute the following commands:
```
# Required environmental variables for the script:
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/

# Starting in the root of the Git repo:
cd examples;

# Serialize images with:
# - 500px side length maximum
# - 50% JPEG encoded
# - quality=90 JPEGs
./write_imagenet.sh 500 0.50 90
```

After that, rename the files and move to `${path_to_imagenet}`:
```
mv ${WRITE_DIR}/train_500_0.50_90.ffcv ${path_to_imagenet}/train.ffcv
mv ${WRITE_DIR}/val_500_0.50_90.ffcv ${path_to_imagenet}/val.ffcv
```

### Step 3: Training and Evaluation

We provide trained model weights at [zenodo](https://zenodo.org/record/7923746). After downloading, extract the models and mount the models, configs and datasets to docker container with:
```
docker run -it -v $(pwd)/configs:/data -v ${path_to_imagenet}:/imagenet -v ${path_to_models}:/models --net host --name stretch-ae --shm-size=32g kalineid/nn_stretch /bin/bash
```

Note that `--shm-size=xxx` is needed. The default 64M shared memory is not enough for the pytorch dataloader.

In the container, create a symlink to `/imagenet` (follow [this doc](https://github.com/kaleid-liner/mbm-pycls/blob/main/docs/DATA.md) for more details):
```
mkdir -p /app/mbm-pycls/pycls/datasets/data
ln -sv /imagenet /app/mbm-pycls/pycls/datasets/data/imagenet
mkdir -p /app/mbm-pycls/pycls/datasets/ffcv
ln -sv /imagenet /app/mbm-pycls/pycls/datasets/ffcv/imagenet
```

To evaluate the accuracy using trained models:
```
python tools/eval_acc.py --cfg /data/mi/EN-B5-CD.yaml
```

An example output for EN-B5-CD is provided [here](sample_output/EN-B5-CD-ACC.log). Check the last line for accuracy:
```
[meters.py: 260]: json_stats: {"_type": "test_epoch", "epoch": "1/100", "mem": 1138, "min_top1_err": 21.2500, "min_top5_err": 5.7680, "time_avg": 0.0395, "time_epoch": 41.1608, "top1_err": 21.2500, "top5_err": 5.7680}
```

You can also train and evaluate a model yourself through the following command:
```
python /app/mbm-pycls/tools/run_net.py --cfg /data/mi/EN-B5-CD.yaml --mode train|test OUT_DIR ${out_dir} NUM_GPUS ${num_gpus} DATA_LOADER.MODE ffcv
```

If you want to evaluate latency and accuracy in a single run, use this script:
```
./eval_lat_acc.sh /data/mi/EN-B5-CD.yaml {arg1, arg2...}
```

The arguments are the same with `run_pipeline.py`.

#### Supernet Training and Evaluation

To train and evaluate the accuracy of the supernet, please use this [code base](src/pycls). Follow [README](https://github.com/kaleid-liner/pycls/blob/main/README.md) to prepare the environment and dataset, then train and evaluate the supernet with:

```
./tools/run_net.py --mode train \
    --cfg configs/elastic/R-50-2B.yaml \
    OUT_DIR R-50-2B \
    LOG_DEST file
```

## Compilation of `benchmark_model`

The `benchmark_model` used for profiling is compiled in the [Dockerfile](Dockerfile). In the case you want to modify the `benchmark_model`, we provide a seperate docker. It can be built with `docker build -f Dockerfile.tf .`.

In the container, you can trigger the compilation with:
```
bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model
```

## Trouble shooting

### Could not create Hexagon delegate

It means your android smartphone is not equipped with a Hexagon DSP or your Hexagon architecture is unsupported by tensorflow. Currently, tensorflow only support Hexagon of architecture 680/682/685/690.

### Segmentation fault

This error is most likely caused by no root access. Root is required for energy measurement. If you don't want to profile the energy, please pass `--no_root` to `run_pipeline.py`.

### Incorrect energy result

The energy result can be incorrect if your phone is connected via USB. Please use WiFi ADB if you want to profile the energy accurately.

### Unstable results on Pixel6

Pixel6 (or similar device) has two big cores and the results can be very unstable if using 2 big cores + 2 mid cores (varying from 20 ms to 100 ms). The unmodified TFLite2.9 has the same problem. Please specify "--num_theads=2 --core_affinity=c0" on such device.

## Code structure

- src/mbm-pycls: The model generation and training are based on [pycls](https://github.com/facebookresearch/pycls). It is included in this repo as a submodule. Our modifications involve commits from [fc4fc5](https://github.com/kaleid-liner/mbm-pycls/commit/fc4fc503e7d0f4f9fbf369b068d45af191d8c5e9) to latest, mainly including:
    - pycls/ir: build pytorch/tensorflow models from multi-branch configs.
    - pycls/models: define extra `nn.Module`s to build the models

- src/tensorflow: Implement the inference for multi-branch models based on tflite. It is included in this repo as a submodule. Our modifications involve commits from [be3db60](https://github.com/kaleid-liner/tensorflow/commit/be3db600d6fb9fcc185c5537f3cbbef3d558a0d0) to latest, mainly including:
    - tensorflow/lite/core/subgraph.cc: modify the inference procedure; partition the model into different devices using the name tags
    - tensorflow/lite/graph_info.cc: extra utils for model partition
    - tensorflow/lite/c/common.h: change the api of `TfLiteContext`
    - tensorflow/lite/delegates/gpu: implement api for asynchronous gpu execution and memory mapping
    - tensorflow/lite/energy_profiler.cc: monitor energy consumption

  The principle of our modifications is maintaining the function and structure of the code-base without harming its generability.

- src/tools: python script to run the pipeline.
