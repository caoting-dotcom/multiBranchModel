import argparse
import os
from pathlib import Path
import logging
import glob
import os
import re

import pandas as pd
import numpy as np
import tensorflow as tf
from ppadb.client import Client as AdbClient

import pycls.core.config as config
from pycls.core.config import cfg
from pycls.ir.constructor.tensorflow.anynet import anynet
import pycls.core.builders as builders


logger = logging.getLogger(__name__)


def get_representative_dataset(input_shape):
    def representative_dataset():
        for _ in range(100):
            data = np.random.rand(1, *input_shape)
            yield [data.astype(np.float32)]
    return representative_dataset


class MeasurementBasedLatencyPredictor:
    def __init__(self, bin_path, device_tmp_dir, host="127.0.0.1", serial="", as_root=True):
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
        
        self.device_bin_path = os.path.join(device_tmp_dir, os.path.basename(bin_path))
        self.push(bin_path, self.device_bin_path)
        self.device.shell("chmod +x {}".format(self.device_bin_path))

    def gen_tflite_model(self, cfg_path, model_path):
        logger.info("Generating tflite model for {} at {}".format(cfg_path, model_path))
        config.load_cfg(cfg_path)
        config.assert_cfg()
        cfg.freeze()
        im_size = cfg.TRAIN.IM_SIZE
        cx = {"h": im_size, "w": im_size, "flops": 0, "params": 0, "acts": 0}
        cx = builders.get_model().complexity(cx)
        cx = {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}

        if not os.path.exists(model_path):
            input_shape = (im_size, im_size, 3)
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
        pattern = "avg=(-?\d+(?:\.\d+)?)"
        for line in reversed(res.splitlines()):
            if line.startswith("Timings (microseconds):"):
                m = re.search(pattern, line)
                if m:
                    return float(m[1])

    def push(self, host_path, device_path):
        logger.info("Pushing from {} to {}".format(host_path, device_path))
        self.device.push(host_path, device_path)

    def predict(self, device_model_path):
        use_gpu = "true" if "gpu" in cfg.ANYNET.DEVICES else "false"
        use_hexagon = "true" if "dsp" in cfg.ANYNET.DEVICES else "false"

        command = (
            "taskset f0 {} "
            "--graph={} "
            "--use_gpu={} "
            "--use_hexagon={} "
            "--num_threads=4 "
            "--use_xnnpack=false "
            "--enable_op_profiling=true "
            "--max_delegated_partitions=100 "
            "--warmup_min_secs=0 "
            "--min_secs=0 "
            "--warmup_runs=5 "
            "--num_runs=50".format(self.device_bin_path, device_model_path, use_gpu, use_hexagon)
        )
        if self.as_root:
            command = "su -c " + "'" + command + "'"
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
    parser.add_argument("--bin_path", type=str, default="/android/benchmark_model")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument("--no_root", dest="as_root", action="store_false")
    parser.set_defaults(as_root=True)
    return parser.parse_args()


def main():
    logger.setLevel(logging.INFO)
    args = parse_args()

    bin_path = args.bin_path

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

    predictor = MeasurementBasedLatencyPredictor(bin_path, args.device_tmp_dir, serial=serial, host=adb_host, as_root=as_root)
    
    results = []
    for cfg_path in configs:
        logger.info("Running experiment: {}".format(cfg_path))
        model_name = Path(cfg_path).stem
        model_basename = model_name + ".tflite"
        model_path = os.path.join(host_tmp_dir, model_basename)

        cx = predictor.gen_tflite_model(cfg_path, model_path)
        logger.info("Complexity: {}".format(cx))

        device_model_path = os.path.join(device_tmp_dir, model_basename)
        predictor.push(model_path, device_model_path)

        latency = predictor.predict(device_model_path)
        results.append({
            "cfg": cfg_path,
            "latency": latency,
            "energy": 0,
        })

    df = pd.DataFrame.from_dict(results)
    df.to_csv(output_csv, index=False)
    logger.info("Results generated at {}".format(output_csv))


if __name__ == "__main__":
    main()
