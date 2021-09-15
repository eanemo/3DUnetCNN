import keras2onnx
import argparse
from unet3d.models.keras.load import load_model
from unet3d.models.keras.build import build_model
import json
import numpy as np

def convert(args):

    fp_json = open(args.config)
    config = json.load(fp=fp_json)      # Devuelve un diccionario
    model_name = config['model_name']
    window = np.asarray(config['window'])
    input_shape = tuple(window.tolist() + [config['n_features']])
    if "n_outputs" in config:
        num_outputs = config['n_outputs']
    else:
        num_outputs = len(np.concatenate(config['metric_names']))

    activation = config['model_kwargs']['activation']
    model = build_model(model_name, input_shape=input_shape, num_outputs=num_outputs,
                            activation=activation)
    #model = load_model(args.model_filename)
    model.load_weights(args.model_filename)
    # Se guarda en formato ProtoBuffer
    model.save('./', save_format="tf")
    onnx_model = keras2onnx.convert_keras(model, name="3DUNet", channel_first_inputs=args.channel_first_inputs)
    #fp = open(args.converted_filename, "wb")
    #keras2onnx.save_model(onnx_model, fp)
    #fp.close()
    #model = build_model(model_name, input_shape=input_shape, num_outputs=num_outputs,  activation=config['activation'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Fichero de configuración usado durante el entrenamiento", required=True)
    parser.add_argument("--model_filename", type=str, help="Fichero que contiene el modelo guardado que queremos convertir", required=True)
    parser.add_argument("--converted_filename", type=str,
                        help="Fichero que contiene el modelo convertido a formato ONNX", required=True)

    args = parser.parse_args()
    convert(args)