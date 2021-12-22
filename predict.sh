#!/usr/bin/bash

# Se exporta el PythonPath
export PYTHONPATH=${PWD}:${PYTHONPATH}

# Se activa el entorno
source env/bin/activate

# Ejecutamos
python unet3d/scripts/predict.py --config_filename path_to_results/config.json --model_filename path_to_results/model_best.h5 --group validation --output_directory path_to_results/predictions
