#!/bin/bash

python eval_acc.py --cfg $1
mkdir -p tmp
cp $1 tmp
python run_pipeline.py --configs tmp ${@:2}
