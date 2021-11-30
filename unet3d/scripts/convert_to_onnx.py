import torch
import argparse
import json
import numpy as np
from unet3d.models.pytorch.build import build_or_load_model
import keras2onnx


def convert(args):
    # Modelo anterior
    #dummy_input = torch.tensor(torch.randn(
    #    2, 1, 96, 96, 96, device='cuda')).float()
    # Modelo nuevo
    dummy_input = torch.tensor(torch.randn(
        1, 1, 176, 224, 144, device='cuda')).float()
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
    
    if args.debug:
        print("model_name:", model_name)
        print("model_filename:", args.model_filename)
        print("num_outputs:", num_outputs)
        print("n_features:", n_features)
        print("model_kwargs:", model_kwargs)

    model = build_or_load_model(model_name=model_name, model_filename=args.model_filename, n_outputs=num_outputs,
                                n_features=n_features, n_gpus=args.gpus, strict=True, **model_kwargs)
    
    torch.onnx.export(model, dummy_input, args.converted_filename, verbose=True, input_names=[
                      args.input], output_names=[args.output], opset_version=11)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, help="Fichero de configuración usado durante el entrenamiento", required=True)
    parser.add_argument("--model_filename", type=str,
                        help="Fichero que contiene el modelo guardado que queremos convertir", required=True)
    parser.add_argument("--converted_filename", type=str,
                        help="Fichero que contiene el modelo convertido a formato ONNX", required=True)
    parser.add_argument("--input", type=str, help="Nombre del nodo de entrada de la red", required=False, default="module.encoder.layers.0.blocks.0.conv1.norm1")
    parser.add_argument("--output", type=str, help="Nombre del nodo de salida de la red", required=False, default="module.final_convolution")
    parser.add_argument("--gpus", type=int, help="Número de GPUS usadas durante la conversión (Se recomienda usar el mismo número que durante el entrenamiento).", default=1)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    
    if args.debug:
        print(args)
    convert(args)
