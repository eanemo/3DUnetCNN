#!/usr/bin/bash

# Se exporta el PythonPath
export PYTHONPATH=${PWD}:${PYTHONPATH}

# Se activa el entorno
source env/bin/activate

# Se define la ruta del experimento
EXPERIMENT_PATH="path_to_results"

# Ejecutamos
python unet3d/scripts/train.py --config_filename ${EXPERIMENT_PATH}/config.json --model_filename ${EXPERIMENT_PATH}/model.h5 --training_log_filename ${EXPERIMENT_PATH}/output_log.csv