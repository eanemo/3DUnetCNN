import keras2onnx
import argparse
from unet3d.models.keras.load import load_model

def convert(args):
    model = load_model(args.model_filename)
    onnx_model = keras2onnx.convert_keras(model, name="3DUNet", channel_first_inputs=args.channel_first_inputs)
    #fp = open(args.converted_filename, "wb")
    #keras2onnx.save_model(onnx_model, fp)
    #fp.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_filename", type=str, help="Fichero que contiene el modelo guardado que queremos convertir", required=True)
    parser.add_argument("--converted_filename", type=str,
                        help="Fichero que contiene el modelo convertido a formato ONNX", required=True)
