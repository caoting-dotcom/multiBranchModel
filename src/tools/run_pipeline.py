import argparse
import os
from pathlib import Path
import logging
import glob
import os
import re
import gc

import pandas as pd
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from ppadb.client import Client as AdbClient
from tensorflow.python.framework.ops import disable_eager_execution

import pycls.core.config as config
from pycls.core.config import cfg
from pycls.ir.constructor.tensorflow.anynet import anynet
import pycls.core.builders as builders


logger = logging.getLogger("NN-Stretch")


def get_representative_dataset(input_shape):
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape)
            yield [data.astype(np.float32)]
    return representative_dataset


class MeasurementBasedLatencyPredictor:
    def __init__(self, runtime_path, device_tmp_dir, host="127.0.0.1", serial="", as_root=True):
        self.host = host
        self.serial = serial
        self.device_tmp_dir = device_tmp_dir
        self.as_root = as_root

        logger.info("Initializing the connection with android device...")
        self.client = AdbClient(self.host, port=5037)
        if self.serial:
            self.device = self.client.device(self.serial)
        else:
            self.device = self.client.devices()[0]
        
        self.device_bin_path = os.path.join(self.device_tmp_dir, "benchmark_model")
        self.device_gpu_bin_path = os.path.join(self.device_tmp_dir, "benchmark_model_only_gpu")
        for filename in glob.glob(os.path.join(runtime_path, "*")):
            device_path = os.path.join(self.device_tmp_dir, os.path.basename(filename))
            self.push(filename, device_path)
        
        self.device.shell("chmod +x {} && chmod +x {}".format(self.device_bin_path, self.device_gpu_bin_path))

    def gen_tflite_model(self, cfg_path, model_path):
        logger.info("Generating tflite model for {} at {}".format(cfg_path, model_path))
        config.load_cfg(cfg_path)
        config.assert_cfg()
        cfg.freeze()
        train_im_size = cfg.TRAIN.IM_SIZE
        test_im_size = 224
        cx = {"h": train_im_size, "w": train_im_size, "flops": 0, "params": 0, "acts": 0}
        cx = builders.get_model().complexity(cx)

        if not os.path.exists(model_path):
            input_shape = (test_im_size, test_im_size, 3)
            net = anynet(input_shape)

            converter = tf.lite.TFLiteConverter.from_keras_model(net)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = get_representative_dataset(input_shape)
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            tflite_model = converter.convert()

            with open(model_path, "wb") as f:
                f.write(tflite_model)
        
        return cx

    @staticmethod
    def _parse(res: str):
        latency_pattern = "min=(-?\d+(?:\.\d+)?)"
        power_pattern = "(-?\d+(?:\.\d+)?)"
        latency = 0
        power = 0
        for line in reversed(res.splitlines()):
            if line.startswith("Timings (microseconds):"):
                m = re.search(latency_pattern, line)
                if m:
                    latency = float(m[1])
            elif line.startswith("#Power (moving):"):
                m = re.search(power_pattern, line)
                if m:
                    power = float(m[1])
        
        energy = power * latency
        return latency, energy

    def push(self, host_path, device_path):
        logger.info("Pushing from {} to {}".format(host_path, device_path))
        self.device.push(host_path, device_path)

    def _error_handle(res):
        for line in res.splitlines():
            if line.startswith("Could not create Hexagon delegate:"):
                logger.error(line)


    def predict(self, device_model_path):
        logger.info("Measure {} on device".format(device_model_path))
        use_gpu = "true" if "gpu" in cfg.ANYNET.DEVICES else "false"
        use_hexagon = "true" if "dsp" in cfg.ANYNET.DEVICES else "false"
        powersave_level = cfg.TEST.POWERSAVE_LEVEL

        if len(cfg.ANYNET.DEVICES) == 1 and use_gpu == "true":
            bin_path = self.device_gpu_bin_path
        else:
            bin_path = self.device_bin_path

        command = (
            "taskset f0 {} "
            "--graph={} "
            "--use_hexagon={} "
            "--use_gpu={} "
            "--num_threads=4 "
            "--use_xnnpack=false "
            "--enable_op_profiling=true "
            "--max_delegated_partitions=100 "
            "--warmup_min_secs=0 "
            "--min_secs=0 "
            "--warmup_runs=5 "
            "--num_runs=50 "
            "--hexagon_powersave_level={}".format(bin_path, device_model_path, use_hexagon, use_gpu, powersave_level)
        )
        if self.as_root:
            command = "su -c " + "'" + command + "'"
        logger.info("Running command {} on device".format(command))
        res = self.device.shell(command)

        return MeasurementBasedLatencyPredictor._parse(res)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", type=str, default="",
        help="Device serial of the android phone. Leave empty if only one device is connected to your host")
    parser.add_argument("--adb_host", type=str, default="127.0.0.1",
        help="Set to `host.docker.internal` if connecting to windows host from docker container")
    parser.add_argument("--workdir", type=str, default="/workspace",
        help="Root of the project.")
    parser.add_argument("--mbm_pycls_path", type=str, default="/app/mbm-pycls")
    parser.add_argument("--host_tmp_dir", type=str, default="")
    parser.add_argument("--device_tmp_dir", type=str, default="/data/local/tmp")
    parser.add_argument("--output_csv", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--no_root", dest="as_root", action="store_false")
    parser.add_argument("--android_runtime", type=str, default="/android/runtime")
    parser.set_defaults(as_root=True)
    return parser.parse_args()


def main():
    logging.basicConfig()
    logger.setLevel(logging.INFO)
    disable_eager_execution()

    args = parse_args()
    runtime_path = args.android_runtime
    host_tmp_dir = args.host_tmp_dir or os.path.join(args.workdir, "data")
    device_tmp_dir = args.device_tmp_dir
    Path(host_tmp_dir).mkdir(parents=True, exist_ok=True)
    output_csv = args.output_csv or os.path.join(args.workdir, "data", "result.csv")
    mbm_pycls_path = args.mbm_pycls_path
    configs_path = args.configs or os.path.join(mbm_pycls_path, "configs", "stretch")
    serial = args.serial
    adb_host = args.adb_host
    as_root = args.as_root
    if os.path.isdir(configs_path):
        configs = glob.glob(os.path.join(configs_path, "*.yaml"))
    else:
        configs = [configs_path]

    predictor = MeasurementBasedLatencyPredictor(runtime_path, args.device_tmp_dir, serial=serial, host=adb_host, as_root=as_root)
    
    if os.path.exists(output_csv):
        results = pd.read_csv(output_csv)
    else:
        results = pd.DataFrame({
            "cfg": [],
            "latency": [],
            "energy": [],
        })
    for cfg_path in configs:
        if cfg_path in results["cfg"].values:
            logger.info("Experiment {} already runned".format(cfg_path))
            continue

        logger.info("Running experiment: {}".format(cfg_path))
        model_name = Path(cfg_path).stem
        model_basename = model_name + ".tflite"
        model_path = os.path.join(host_tmp_dir, model_basename)

        cx = predictor.gen_tflite_model(cfg_path, model_path)
        logger.info("Complexity: {}".format(cx))

        device_model_path = os.path.join(device_tmp_dir, model_basename)
        predictor.push(model_path, device_model_path)

        latency, energy = predictor.predict(device_model_path)
        results = results.append({
            "cfg": cfg_path,
            "latency": latency,
            "energy": energy,
        }, ignore_index=True)
        results.to_csv(output_csv, index=False)
        logger.info("Result appended to {}".format(output_csv))

        # Reclaim memory
        tf.keras.backend.clear_session()
        gc.collect()

    logger.info("Results generated at {}".format(output_csv))


if __name__ == "__main__":
    main()
