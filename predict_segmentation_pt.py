import os
import sys
import math
import numpy as np
from glob import glob
import random
from tqdm import tqdm
import time
import pandas as pd
import SimpleITK as sitk
import torch

from utils.configuration import config
from networks.ctcf_unet_3d import CTCFUNet3D
from networks.dbtc_net_3d import DBTCNet3D
from networks.mtc_net_3d import MTCNet3D
from utils.preprocessing import *
from utils.sw import sw

seed_value = 0

os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

torch.manual_seed(seed_value)

torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)


class Prediction(object):
    def __init__(self, test_type, prediction_model_name, device, result_dir):
        self.test_type = test_type
        self.prediction_model_name = prediction_model_name
        self.device = device
        self.result_dir = result_dir

        model_dir = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]), 'Segmentation',
                                 config["seg_model_name"])
        checkpoint_dir = os.path.join(model_dir, 'checkpoint')
        self.checkpoint_path = os.path.join(checkpoint_dir,
                                            'checkpoint_epoch_{:04d}.pth'.format(config["seg_pred_epoch"]))

        self.prediction_model = load_pred_model(self.checkpoint_path, self.device)

    def predict_model_2d_axial(self, test_image):
        prediction_model_2d = self.prediction_model

        test_image_2d = np.transpose(test_image, (1, 0, 2, 3))

        pred_2d = predict_model_2d(test_image_2d, prediction_model_2d, self.device).detach().cpu().numpy()

        pred_2d = np.transpose(pred_2d, (0, 2, 3, 1))

        return pred_2d

    def predict_model_3d_axial(self, test_image):
        prediction_model_3d = self.prediction_model

        test_image_3d = test_image

        pred_3d = np.zeros_like(test_image_3d)

        pred_3d = predict_model_3d(test_image_3d, prediction_model_3d,
                                   self.device).detach().cpu().numpy()

        pred_3d = np.transpose(pred_3d, (1, 2, 3, 0))

        return pred_3d

    def predict_volume(self, filepath_patient):
        t1 = glob(filepath_patient + '/*_t1.nii.gz')
        t1ce = glob(filepath_patient + '/*_t1ce.nii.gz')
        t2 = glob(filepath_patient + '/*_t2.nii.gz')
        flair = glob(filepath_patient + '/*_flair.nii.gz')

        assert ((len(t1) + len(t1ce) + len(t2) + len(flair)) == config["num_modalities"]), \
            "There is a problem here. The problem lies in this patient: {}".format(filepath_patient)
        scans_test = [t1[0], t1ce[0], t2[0], flair[0]]

        test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i]))
                   for i in range(len(scans_test))]
        test_im = np.array(test_im).astype(np.float32)

        test_image = test_im[0:4]
        gt = []

        test_image = normalize(test_image)

        if config["network_dims"] == "2D":
            if config["network_view"] == "Axial":
                pred = self.predict_model_2d_axial(test_image)
        elif config["network_dims"] == "3D":
            if config["network_view"] == "Axial":
                pred = self.predict_model_3d_axial(test_image)

        pred = np.argmax(pred, axis=-1)
        pred = pred.astype(np.uint8)

        pred[pred == 3] = 4

        return gt, pred

    def evaluate_segmented_volume(self, filepath_patient, patient_id):
        gt, pred = self.predict_volume(filepath_patient)

        img = sitk.GetImageFromArray(pred)

        save_path = os.path.join(self.result_dir, '{}.nii.gz'.format(patient_id))
        sitk.WriteImage(img, save_path)

    def predict_multiple_volumes(self, filepaths_patients):
        for patient in tqdm(filepaths_patients, desc="Prediction on data"):
            patient_id = patient.split(os.sep)[-2]
            self.evaluate_segmented_volume(patient, patient_id)


def load_pred_model(checkpoint_path, device):
    if config["seg_model_name"] == "3D CTCF-UNet":
        model = CTCFUNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])
    elif config["seg_model_name"] == "3D DBTC-Net":
        model = DBTCNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])
    elif config["seg_model_name"] == "3D MTC-Net":
        model = MTCNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])

    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    return model


def predict_model_2d(test_image_2d, prediction_model_2d, device):
    test_image_2d = torch.from_numpy(test_image_2d).to(device)

    with torch.no_grad():
        pred = prediction_model_2d(test_image_2d)

    return pred


def predict_model_3d(test_image_3d, prediction_model_3d, device):
    batch_test_image_3d = np.array([test_image_3d])

    batch_test_image_3d = torch.from_numpy(batch_test_image_3d).to(device)

    if not config["sw"]:
        with torch.no_grad():
            batch_prediction = prediction_model_3d(batch_test_image_3d)
    else:
        batch_prediction = sw(prediction_model_3d, batch_test_image_3d, device)

    pred = batch_prediction[0]

    return pred


if __name__ == '__main__':

    if config["gpu_cpu"] == "GPU" and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    test_type = "Validation"
    result_dir = os.path.join(os.getcwd(), 'results', str(config["prediction_year"]),
                              'Segmentation', config["seg_model_name"])
    result_dir = os.path.join(result_dir, test_type)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    paths_test = glob(config["test_path"] + "/*/")

    brain_seg_pred = Prediction(test_type=test_type,
                                prediction_model_name=config["seg_model_name"],
                                device=device,
                                result_dir=result_dir)

    brain_seg_pred.predict_multiple_volumes(paths_test)
