import glob
import os
from os.path import join, isdir, basename
import hashlib
import argparse
import shutil

import numpy as np
from tqdm import tqdm
from PIL import Image


def main(args):
    dirs = ["train", "val"]
    os.makedirs(args.target, exist_ok=True)
    for dir in dirs:
        source_dir = join(args.source, dir)
        target_dir = join(args.target, dir)
        cases = glob.glob(source_dir + "/" + "*")
        pbar = tqdm(total=len(cases))
        for case in cases:
            if isdir(case):
                base_dir = basename(case)
                # Es un directorio y por tanto un caso
                input_files = glob.glob(join(case, "*_cut_*.png"))
                input_segmentation = glob.glob(
                    join(case, "*_segmentation_*.png"))
                input_original_xml = sorted(
                    glob.glob(join(case, "*_original_*.xml")))

                output_case = hashlib.sha256(
                    base_dir.encode('utf-8')).hexdigest()
                output_dir = join(target_dir, output_case)
                os.makedirs(output_dir, exist_ok=True)

                for input_file in input_files:
                    file = basename(input_file)
                    new_file = file.replace(base_dir, output_case)
                    new_file_output = join(output_dir, new_file)
                    shutil.copy(input_file, new_file_output)

                for input_file in input_original_xml:
                    file = basename(input_file)
                    new_file = file.replace(base_dir, output_case)
                    new_file_output = join(output_dir, new_file)
                    shutil.copy(input_file, new_file_output)

                for input_file in input_segmentation:
                    file = basename(input_file)
                    image = Image.open(input_file)
                    array = np.asarray(image)
                    #array[array > 0] = 1
                    label_modified = Image.fromarray(array)
                    new_file = file.replace(base_dir, output_case)
                    new_file_output = join(output_dir, new_file)
                    #shutil.copy(input_file, new_file_output)
                    label_modified.save(new_file_output)
            pbar.update(1)
            pbar.set_description(case)
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocesado del dataset.')
    parser.add_argument('source',
                        help='Directorio origen del dataset')
    parser.add_argument('target',
                        help='Directorio dónde vamos a escribir el dataset modificado en formato NII')
    parser.add_argument('--verbose', dest='verbose',
                        action='store_true', help="Mostrar informacion del proceso")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    main(args)
