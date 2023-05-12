import argparse
import logging
import subprocess
from pathlib import Path
import os


logger = logging.getLogger("NN-Stretch")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str)
    parser.add_argument("--models", type=str, default="/models")
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default="/models")
    parser.add_argument("--workdir", type=str, default="/app/mbm-pycls")
    return parser.parse_args()


def main():
    logging.basicConfig()
    logger.setLevel(logging.INFO)

    args = parse_args()
    model_name = Path(args.cfg).stem
    model_path = os.path.join(args.models, model_name)
    out_path = os.path.join(args.out_dir, model_name)
    run_net_path = os.path.join(args.workdir, "tools/run_net.py")
    if os.path.exists(model_path):
        logger.info("Starting evaluation...")
        command = "python {} --cfg {} --mode test OUT_DIR {} NUM_GPUS {} TEST.WEIGHTS {} DATA_LOADER.MODE ffcv".format(
            run_net_path, args.cfg, out_path, args.num_gpus, os.path.join(model_path, "model.pyth")
        )
        subprocess.run(command, shell=True)
    else:
        logger.info("Starting training...")
        command = "python {} --cfg {} --mode train OUT_DIR {} NUM_GPUS {} DATA_LOADER.MODE ffcv".format(
            run_net_path, args.cfg, out_path, args.num_gpus
        )
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
