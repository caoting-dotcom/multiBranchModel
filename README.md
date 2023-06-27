This repo includes the implementation for our *MobiSys'23* paper "NN-Stretch: Automatic Neural Network Branching for
Parallel Inference on Heterogeneous Multi-Processors". 

The bibtex for the paper is:

```
@inproceedings{10.1145/3581791.3596870,
author = {Wei, Jianyu and Cao, Ting and Cao, Shijie and Jiang, Shiqi and Fu, Shaowei and Yang, Mao and Zhang, Yanyong and Liu, Yunxin},
title = {NN-Stretch: Automatic Neural Network Branching for Parallel Inference on Heterogeneous Multi-Processors},
year = {2023},
doi = {10.1145/3581791.3596870},
pages = {70–83},
location = {Helsinki, Finland},
series = {MobiSys '23}
}
```

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

We provide trained model weights for [configs/mi](configs/mi) at [zenodo](https://zenodo.org/record/7934193). After downloading, extract the models and mount the models, configs and datasets to docker container with:
```
docker run -it -v $(pwd)/configs:/data -v ${path_to_imagenet}:/imagenet -v ${path_to_models}:/models --net host --name stretch-ae --shm-size=32g --gpus all kalineid/nn_stretch /bin/bash
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
python eval_acc.py --cfg /data/mi/EN-B5-CD.yaml
```

> Note that the FFCV dataloader can only work for multiple GPUs. Please specify 2 or more `NUM_GPUS` in the config.

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

To train and evaluate the accuracy of the supernet, please use [src/pycls](https://github.com/kaleid-liner/pycls). Follow [README](https://github.com/kaleid-liner/pycls/blob/main/README.md) to prepare the environment and dataset, then train and evaluate the supernet with:

```
./tools/run_net.py --mode train \
    --cfg configs/elastic/R-50-2B.yaml \
    OUT_DIR R-50-2B \
    LOG_DEST file
```

## Compilation of `benchmark_model`

The `benchmark_model` used for profiling is compiled in the [Dockerfile](Dockerfile). In the case you want to modify the `benchmark_model`, we provide a seperate docker. It can be built with `docker build -f Dockerfile.tf -t tf .` and started with `docker run -it tf /bin/bash`.

In the container, you can trigger the compilation with:
```
bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model
```

The binaries in [android_runtime](android_runtime) were built using different branches of our implementation:

- The default `benchmark_model` uses [sync-cpu-no-copy](https://github.com/kaleid-liner/tensorflow/tree/sync-cpu-no-copy).
- The `benchmark_model_only_gpu` uses [only_gpu](https://github.com/kaleid-liner/tensorflow/tree/only_gpu).
- The `benchmark_model_dg` uses [sync-cpu-no-copy](https://github.com/kaleid-liner/tensorflow/tree/only_gpu) with a patch to fix [Google Ruy](https://github.com/google/ruy). Run `./apply_patch.sh` to build.

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

## Verify calculation correctness

We verified the calculation correctness for our changes by ensuring each kernel was executed in the proper sequence from the output logs. We provide a sample output for R-34-DG at [correctness.log](sample_output/correctness.log).

### How to get the output

More detailed profiling could be enabled. We add some profiling codes at [more-profiling](https://github.com/kaleid-liner/tensorflow/tree/more-profiling) branch. Please checkout and build first(in the container of [Compilation of `benchmark_model`](https://github.com/caoting-dotcom/multiBranchModel#compilation-of-benchmark_model) section).

```
git checkout more-profiling
bazel build -c opt --config=android_arm64 //tensorflow/lite/tools/benchmark:benchmark_model
adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp/benchmark_model
```

> Note that the profiling will result in degraded performance.

Then, run the following command to get the output log:
```
adb shell "chmod +x /data/local/tmp/benchmark_model && taskset f0 /data/local/tmp/benchmark_model --graph=/data/local/tmp/R-34-DG.tflite --use_hexagon=true --use_gpu=true --num_threads=4 --use_xnnpack=false --enable_op_profiling=true --max_delegated_partitions=100 --warmup_min_secs=0 --min_secs=0 --warmup_runs=1 --num_runs=2 --hexagon_powersave_level=128 --hexagon_support_head=false --input_layer='input1' --input_layer_shape='1,224,224,3' --input_layer_value_files='input1:/data/local/tmp/input.bin' --output_class_label=true"
```

### Analyze the output

First, for operator-wise profling, we can see a clear timeline for all of the operators:
```
Operator-wise Profiling Info for Regular Benchmark Runs:
============================== Run Order ==============================
                     [node type]                  [start]         [first]        [avg ms]            [%]          [cdf%]          [mem KB]      [times called]  [Name]
                        QUANTIZE                    0.024           0.094           0.093         0.424%          0.424%             0.000              1       [tfl.quantize]:0
                         CONV_2D                    0.117           3.730           3.640        16.512%         16.936%             0.000              1       [anynet/st_cpu_relu/Relu;anynet/st_cpu_bn/FusedBatchNormV3;anynet/s2_gpu_b1_0_bn/FusedBatchNormV3/ReadVariableOp;anynet/st_cpu_conv/BiasAdd;anynet/s2_gpu_b3_1_bn/FusedBatchNormV3;anynet/st_cpu_conv/Conv2D]:1
                     MAX_POOL_2D                    3.758           0.176           0.176         0.798%         17.734%             0.000              1       [anynet/st_cpu_mp_start_pool/MaxPool]:2
           TfLiteHexagonDelegate                    3.934           0.009           0.009         0.041%         17.775%             0.000              1       [anynet/s1_dsp_b3_relu/Relu;anynet/s1_dsp_b3_add/add]:96
             TfLiteGpuDelegateV2                    3.943           0.384           0.388         1.760%         19.535%             0.000              1       [anynet/s1_gpu_b2_relu/Relu;anynet/s1_gpu_b2_add/add]:100
                   CONCATENATION                    4.332           3.750           3.809        17.281%         36.816%             0.000              1       [anynet/s1_cpu_mp_start_mp_end_concat/concat]:20
           TfLiteHexagonDelegate                    8.142           0.010           0.010         0.045%         36.861%             0.000              1       [anynet/s2_dsp_b3_relu/Relu;anynet/s2_dsp_b3_add/add]:97
             TfLiteGpuDelegateV2                    8.152           0.482           0.484         2.196%         39.056%             0.000              1       [anynet/s2_gpu_b3_relu/Relu;anynet/s2_gpu_b3_add/add]:101
                   CONCATENATION                    8.636           2.746           2.788        12.645%         51.701%             0.000              1       [anynet/s2_cpu_mp_start_mp_end_concat/concat]:41
           TfLiteHexagonDelegate                   11.424           0.009           0.009         0.043%         51.744%             0.000              1       [anynet/s3_dsp_b6_relu/Relu;anynet/s3_dsp_b6_add/add]:98
             TfLiteGpuDelegateV2                   11.434           0.325           0.336         1.522%         53.266%             0.000              1       [anynet/s3_gpu_b4_relu/Relu;anynet/s3_gpu_b4_add/add]:102
                   CONCATENATION                   11.770           3.639           3.714        16.847%         70.113%             0.000              1       [anynet/s3_cpu_mp_start_mp_end_concat/concat]:74
             TfLiteGpuDelegateV2                   15.484           0.249           0.256         1.159%         71.272%             0.000              1       [anynet/s4_gpu_b2_relu/Relu;anynet/s4_gpu_b2_add/add]:103
           TfLiteHexagonDelegate                   15.739           0.007           0.011         0.052%         71.325%             0.000              1       [anynet/s4_dsp_b3_relu/Relu;anynet/s4_dsp_b3_add/add]:99
                   CONCATENATION                   15.752           2.648           2.684        12.177%         83.502%             0.000              1       [anynet/s4_cpu_mp_end_concat/concat]:92
                            MEAN                   18.436           0.453           0.492         2.232%         85.734%             0.000              1       [anynet/head_cpu_gap/Mean]:93
                 FULLY_CONNECTED                   18.930           2.121           3.142        14.253%         99.986%             0.000              1       [StatefulPartitionedCall:01]:94
                        QUANTIZE                   22.072           0.003           0.003         0.014%        100.000%             0.000              1       [StatefulPartitionedCall:0]:95
```

From the above timeline, we can get that:
- The sequential part of models are executed on CPU in the correct order.
- For the multi-branch part, the subgraphs (4 subgraphs in each processor) are pushed to the corresponding processors for execution.
- The concatenation is the meeting point where CPU waits each subgraph for completion. This is approximately the latency of the slower branch of GPU and DSP.

By analyzing this timeline, we can asure that all of the subgraphs are executed in the proper sequence.

Then, we can get the detailed inference of branches on different processors by:

```
GPU stage 0: waiting for 3774
GPU stage 1: waiting for 2764
GPU stage 2: waiting for 3748
GPU stage 3: waiting for 2712
DSP stage 0: waiting for 1448
DSP stage 1: waiting for 865
DSP stage 2: waiting for 1330
DSP stage 3: waiting for 1343
```

For example, the above result shows that after pushing the subgraph to GPU and DSP, CPU waits the GPU/DSP for completion. The larger one of GPU and DSP is consistent with the latency of  `CONCATENATION` in the operator-wise profiling (stage 0 corrresponding to s1, stage 1 corresponding to s2, and so on):

```
CONCATENATION                    3.851           3.738           3.800        17.606%         35.322%             0.000              1       [anynet/s1_cpu_mp_start_mp_end_concat/concat]:20
```

Futhermore, to asure that kernels, including computation kernels and memory mapping kernels are executed in GPU in correct order, we also use OpenCL events to profile the operator-wise info on GPU:
```
Map::Finished: 1684051998328978176
Unmap::Queued: 1684051998333059072
Kernel::Started: 1684051998333369856
Kernel::Finished: 1684051998334188032
Kernel::Started: 1684051998334189056
Kernel::Finished: 1684051998334907904
Kernel::Started: 1684051998334908928
Kernel::Finished: 1684051998335014144
Kernel::Started: 1684051998335015168
Kernel::Finished: 1684051998335734016
Kernel::Started: 1684051998335734016
Kernel::Finished: 1684051998336459008
Map::Finished: 1684051998336641024
Unmap::Queued: 1684051998328762112
```

The above log shows the operator-wise profiling info for stage#1 of GPU branch, the kernels are executed in sequential order. From top to bottom, the kernels are:
- Map
- Unmap
- b1_0_conv
- b1_1_conv
- proj_conv
- b2_0_conv
- b2_1_conv
- Map
- Unmap

The add/relu operators are fused into conv.

From the above analysis, we can asure that all kernels are executed in the proper sequence, while any precision loss in the different processors won't influence our profiled latency.
