import torch
import argparse
import json
import numpy as np
from unet3d.models.pytorch.build import build_or_load_model

def convert(args):
    dummy_input = torch.tensor(torch.randn(2, 1, 96, 96, 96, device='cuda')).float()
    fp_json = open(args.config)
    config = json.load(fp=fp_json)      # Devuelve un diccionario
    model_name = config['model_name']
    n_features = config['n_features']
    window = np.asarray(config['window'])
    input_shape = tuple(window.tolist() + [config['n_features']])
    if "n_outputs" in config:
        num_outputs = config['n_outputs']
    else:
        num_outputs = len(np.concatenate(config['metric_names']))

    model_kwargs = config['model_kwargs']

    model = build_or_load_model(model_name=model_name, model_filename=args.model_filename, n_outputs=num_outputs,
                                n_features=n_features, n_gpus=1, strict=True, **model_kwargs)

    torch.onnx.export(model, dummy_input, "model.onnx", verbose=True, input_names="encoder.layers.0.blocks.0.conv1.norm1",
                      output_names="final_convolution")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Fichero de configuración usado durante el entrenamiento", required=True)
    parser.add_argument("--model_filename", type=str, help="Fichero que contiene el modelo guardado que queremos convertir", required=True)
    parser.add_argument("--converted_filename", type=str,
                        help="Fichero que contiene el modelo convertido a formato ONNX", required=True)

    args = parser.parse_args()
    convert(args)