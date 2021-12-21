#!/usr/bin/bash

# Se exporta el PythonPath
export PYTHONPATH=${PWD}:${PYTHONPATH}

# Se activa el entorno
source env/bin/activate

# Ejecutamos
python unet3d/scripts/train.py --config_filename path_to_results/config.json --model_filename path_to_results/model.h5 --training_log_filename path_to_results/output_log.csv