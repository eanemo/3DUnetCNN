import glob
import os
from os.path import join, isdir, basename
import SimpleITK as sitk
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import hashlib


def create_dcm_from_mat(mat : np.array, filename : str):
    img = sitk.GetImageFromArray(mat)
    sitk.WriteImage(img, filename)

def convert_mat_to_dicom(files: list):
    old_basedir = os.getcwd()
    for file in files:
        splitted_path = os.path.split(file)
        basedir = splitted_path[0]
        filename = splitted_path[1]
        os.chdir(basedir)
        # Leemos el fichero XML con la matriz
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        image = fs.getNode("image")
        mat = image.mat()
        file_dcm = filename.replace(".xml", ".dcm")
        create_dcm_from_mat(mat, file_dcm)

    os.chdir(old_basedir)


def createVol(volFiles, outputVolumeFile):
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(volFiles)
    vol = reader.Execute()
    sitk.WriteImage(vol, outputVolumeFile)


def main(args):
    dirs = ["train", "val"]
    os.makedirs(args.target, exist_ok=True)
    for dir in dirs:
        source_dir = join(args.source, dir)
        target_dir = join(args.target, dir)
        os.makedirs(target_dir, exist_ok=True)
        cases = glob.glob(source_dir + "/" + "*")
        pbar = tqdm(total=len(cases))
        for case in cases:
            if isdir(case):
                base_dir = basename(case)
                output_case = hashlib.sha256(
                    base_dir.encode('utf-8')).hexdigest()
                # Previamente necesitamos transformar las matrices de OpenCV a ficheros DICOM compatibles con SimpleITK
                input_original_xml = sorted(
                    glob.glob(join(case, "*_original_*.xml")))
                convert_mat_to_dicom(input_original_xml)
                input_original_dcm = sorted(
                    glob.glob(join(case, "*_original_*.dcm")))
                # Es un directorio y por tanto un caso
                input_files = sorted(glob.glob(join(case, "*_cut_*.png")))
                input_segmentation = sorted(
                    glob.glob(join(case, "*_segmentation_*.png")))
                output_nii = join(target_dir, output_case + "_cut.nii.gz")
                segmentation_nii = join(
                    target_dir, output_case + "_segmentation.nii.gz")
                original_nii = join(
                    target_dir, output_case + "_original.nii.gz"
                )
                #createVol(input_files, output_nii)
                createVol(input_segmentation, segmentation_nii)
                createVol(input_original_dcm, original_nii)
                pbar.update(1)
                pbar.set_description(case)
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convertir dataset a formato NII para usar en las redes de segmentación 3D de Monai.')
    parser.add_argument('source',
                        help='Directorio origen del dataset')
    parser.add_argument('target',
                        help='Directorio dónde vamos a escribir el dataset modificado en formato NII')
    parser.add_argument('--verbose', dest='verbose',
                        action='store_true', help="Mostrar informacion del proceso")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    main(args)
