import random
import os
import sys
import math
from glob import glob
from tqdm import tqdm
import time
import SimpleITK as sitk
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.configuration import config

if config["gpu_cpu"] == "CPU":
    os.environ["CUDA_DEVICE_ORDER"] = "0000:01:00.0"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif config["gpu_cpu"] == "GPU":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if config["gpu_mem"] == "memory growth":
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            elif config["gpu_mem"] == "hard limit":
                for gpu in gpus:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=config["gpu_mem_limit"])])

            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            print(e)

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow_addons.layers import GroupNormalization
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.framework import convert_to_constants

from utils.preprocessing import *
from utils.losses_tf import *

K.set_image_data_format("channels_last")

seed_value = 0

os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)


class Prediction(object):
    def __init__(self, test_type, pred_clear_session, prediction_model_name, result_dir):
        self.test_type = test_type
        self.pred_clear_session = pred_clear_session
        self.prediction_model_name = prediction_model_name
        self.result_dir = result_dir
        self.model_dir = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]),
                                      'Segmentation', config["seg_model_name"])
        self.custom_objects = {'GroupNormalization': GroupNormalization,
                               'dice_ET_metric': dice_ET_metric,
                               'dice_WT_metric': dice_WT_metric,
                               'dice_TC_metric': dice_TC_metric}
        if config["prediction_type"] == "Individual":
            model_path = os.path.join(self.model_dir,
                                      config["network_view"] + "_" + str(config["seg_val_fold"]))
            if config["gpu_mem"] == "hard limit":
                model_path = model_path + "_" + str(config["gpu_mem_limit"]) + "MB"
            self.prediction_model_path = os.path.join(model_path, self.prediction_model_name)
            if not self.pred_clear_session:
                self.prediction_model = load_pred_model(self.prediction_model_path, self.custom_objects)
        elif config["prediction_type"] == "Ensemble":
            if not self.pred_clear_session:
                for model_num, model_net_view in config["ensemble_models"].items():
                    model_path = os.path.join(self.model_dir, model_num, self.prediction_model_name)
                    prediction_view = model_net_view[1]
                    if model_net_view[0] == "2D":
                        if prediction_view == "Axial":
                            self.model_2d_ax = load_pred_model(model_path, self.custom_objects)
                        elif prediction_view == "Sagittal":
                            self.model_2d_sg = load_pred_model(model_path, self.custom_objects)
                        elif prediction_view == "Coronal":
                            self.model_2d_cr = load_pred_model(model_path, self.custom_objects)
                    elif model_net_view[0] == "3D":
                        if prediction_view == "Axial":
                            self.model_3d_ax = load_pred_model(model_path, self.custom_objects)
                        elif prediction_view == "Sagittal":
                            self.model_3d_sg = load_pred_model(model_path, self.custom_objects)
                        elif prediction_view == "Coronal":
                            self.model_3d_cr = load_pred_model(model_path, self.custom_objects)

            proportions = config["ensemble_proportions"]
            mean_proportions = np.mean(proportions)
            diff_proportions = proportions - mean_proportions
            proportions = proportions + config["ensemble_variation"] * diff_proportions
            self.weights = proportions / np.sum(proportions)
            path = os.path.split(self.result_dir)[0]

    def predict_model_2d_axial(self, model_path, test_image):
        if config["prediction_type"] == "Individual":
            if self.pred_clear_session:
                prediction_model_2d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_2d = self.prediction_model
        elif config["prediction_type"] == "Ensemble":
            if self.pred_clear_session:
                prediction_model_2d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_2d = self.model_2d_ax

        test_image_2d = np.transpose(test_image, (1, 2, 3, 0))

        if not config["tta"]:
            pred_2d = predict_model_2d(test_image_2d, prediction_model_2d)
        else:
            pred_2d = predict_model_2d_tta_mirroring(test_image_2d, prediction_model_2d)

        if self.pred_clear_session:
            K.clear_session()

        return pred_2d

    def predict_model_2d_sagittal(self, model_path, test_image):
        if config["prediction_type"] == "Individual":
            if self.pred_clear_session:
                prediction_model_2d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_2d = self.prediction_model
        elif config["prediction_type"] == "Ensemble":
            if self.pred_clear_session:
                prediction_model_2d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_2d = self.model_2d_sg

        test_image_2d = np.transpose(test_image, (3, 1, 2, 0))

        pred_2d = np.zeros_like(test_image_2d)

        if not config["tta"]:
            pred_2d = predict_model_2d(test_image_2d, prediction_model_2d)
        else:
            pred_2d = predict_model_2d_tta_mirroring(test_image_2d, prediction_model_2d)

        pred_2d = np.transpose(pred_2d, (1, 2, 0, 3))

        if self.pred_clear_session:
            K.clear_session()

        return pred_2d

    def predict_model_2d_coronal(self, model_path, test_image):
        if config["prediction_type"] == "Individual":
            if self.pred_clear_session:
                prediction_model_2d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_2d = self.prediction_model
        elif config["prediction_type"] == "Ensemble":
            if self.pred_clear_session:
                prediction_model_2d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_2d = self.model_2d_cr

        test_image_2d = np.transpose(test_image, (2, 1, 3, 0))

        pred_2d = np.zeros_like(test_image_2d)

        if not config["tta"]:
            pred_2d = predict_model_2d(test_image_2d, prediction_model_2d)
        else:
            pred_2d = predict_model_2d_tta_mirroring(test_image_2d, prediction_model_2d)

        pred_2d = np.transpose(pred_2d, (1, 0, 2, 3))

        if self.pred_clear_session:
            K.clear_session()

        return pred_2d

    def predict_model_3d_axial(self, model_path, test_image):
        if config["prediction_type"] == "Individual":
            if self.pred_clear_session:
                prediction_model_3d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_3d = self.prediction_model
        elif config["prediction_type"] == "Ensemble":
            if self.pred_clear_session:
                prediction_model_3d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_3d = self.model_3d_ax

        test_image_3d = np.transpose(test_image, (1, 2, 3, 0))

        pred_3d = np.zeros_like(test_image_3d)

        if not config["tta"]:
            pred_3d = predict_model_3d(test_image_3d, prediction_model_3d)
        else:
            pred_3d = predict_model_3d_tta_mirroring(test_image_3d, prediction_model_3d)

        if self.pred_clear_session:
            K.clear_session()

        return pred_3d

    def predict_model_3d_sagittal(self, model_path, test_image):
        if config["prediction_type"] == "Individual":
            if self.pred_clear_session:
                prediction_model_3d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_3d = self.prediction_model
        elif config["prediction_type"] == "Ensemble":
            if self.pred_clear_session:
                prediction_model_3d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_3d = self.model_3d_sg

        test_image_3d = np.transpose(test_image, (3, 1, 2, 0))

        pred_3d = np.zeros_like(test_image_3d)

        if not config["tta"]:
            pred_3d = predict_model_3d(test_image_3d, prediction_model_3d)
        else:
            pred_3d = predict_model_3d_tta_mirroring(test_image_3d, prediction_model_3d)

        pred_3d = np.transpose(pred_3d, (1, 2, 0, 3))

        if self.pred_clear_session:
            K.clear_session()

        return pred_3d

    def predict_model_3d_coronal(self, model_path, test_image):
        if config["prediction_type"] == "Individual":
            if self.pred_clear_session:
                prediction_model_3d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_3d = self.prediction_model
        elif config["prediction_type"] == "Ensemble":
            if self.pred_clear_session:
                prediction_model_3d = load_pred_model(model_path, self.custom_objects)
            else:
                prediction_model_3d = self.model_3d_cr

        test_image_3d = np.transpose(test_image, (2, 1, 3, 0))

        pred_3d = np.zeros_like(test_image_3d)

        if not config["tta"]:
            pred_3d = predict_model_3d(test_image_3d, prediction_model_3d)
        else:
            pred_3d = predict_model_3d_tta_mirroring(test_image_3d, prediction_model_3d)

        pred_3d = np.transpose(pred_3d, (1, 0, 2, 3))

        if self.pred_clear_session:
            K.clear_session()

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

        if config["prediction_type"] == "Individual":
            prediction_model_path = self.prediction_model_path
            if config["network_dims"] == "2D":
                if config["network_view"] == "Axial":
                    pred = self.predict_model_2d_axial(prediction_model_path, test_image)
                elif config["network_view"] == "Sagittal":
                    pred = self.predict_model_2d_sagittal(prediction_model_path, test_image)
                elif config["network_view"] == "Coronal":
                    pred = self.predict_model_2d_coronal(prediction_model_path, test_image)
            elif config["network_dims"] == "3D":
                if config["network_view"] == "Axial":
                    pred = self.predict_model_3d_axial(prediction_model_path, test_image)
                elif config["network_view"] == "Sagittal":
                    pred = self.predict_model_3d_sagittal(prediction_model_path, test_image)
                elif config["network_view"] == "Coronal":
                    pred = self.predict_model_3d_coronal(prediction_model_path, test_image)
        elif config["prediction_type"] == "Ensemble":
            pred = np.zeros((config["d_input"], config["h_input"], config["w_input"],
                             config["num_classes"])).astype(np.float32)
            i = 0
            for model_num, model_net_view in config["ensemble_models"].items():
                prediction_model_path = os.path.join(self.model_dir, model_num, self.prediction_model_name)
                prediction_view = model_net_view[1]
                if model_net_view[0] == "2D":
                    if prediction_view == "Axial":
                        pred += self.weights[i] * self.predict_model_2d_axial(prediction_model_path, test_image)
                    elif prediction_view == "Sagittal":
                        pred += self.weights[i] * self.predict_model_2d_sagittal(prediction_model_path, test_image)
                    elif prediction_view == "Coronal":
                        pred += self.weights[i] * self.predict_model_2d_coronal(prediction_model_path, test_image)
                elif model_net_view[0] == "3D":
                    if prediction_view == "Axial":
                        pred += self.weights[i] * self.predict_model_3d_axial(prediction_model_path, test_image)
                    elif prediction_view == "Sagittal":
                        pred += self.weights[i] * self.predict_model_3d_sagittal(prediction_model_path, test_image)
                    elif prediction_view == "Coronal":
                        pred += self.weights[i] * self.predict_model_3d_coronal(prediction_model_path, test_image)
                i += 1
            pred = pred / np.sum(self.weights)

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


def load_pred_model(model_path, custom_objects):
    if config["pred_model_type"] == "native TF FP32":
        prediction_model = load_model(filepath=model_path,
                                      custom_objects=custom_objects,
                                      compile=False)
    else:
        if config["pred_model_type"] == "TF-TRT FP32":
            model_path = model_path + "_TFTRT_FP32"
        elif config["pred_model_type"] == "TF-TRT FP16":
            model_path = model_path + "_TFTRT_FP16"
        prediction_model = tf.saved_model.load(export_dir=model_path, tags=[tag_constants.SERVING])
        prediction_model = prediction_model.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        prediction_model = convert_to_constants.convert_variables_to_constants_v2(prediction_model)
    return prediction_model


def predict_model_2d(test_image_2d, prediction_model_2d):
    pred_batch_size = config["pred_batch_size"]
    if config["pred_model_type"] == "native TF FP32":
        pred = prediction_model_2d.predict(test_image_2d, batch_size=pred_batch_size, verbose=0)
    else:
        num_samples = test_image_2d.shape[0]
        pred = np.zeros_like(test_image_2d)
        x = 0
        while x < num_samples:
            y = x + pred_batch_size
            pred[x:y, :, :, :] = prediction_model_2d(tf.convert_to_tensor(
                test_image_2d[x:y, :, :, :], dtype='float32'))[0].numpy()
            x = y

    return pred


def predict_model_2d_tta_mirroring(test_image_2d, prediction_model_2d):
    test_image_2d_1 = test_image_2d
    pred_1 = predict_model_2d(test_image_2d_1, prediction_model_2d)

    test_image_2d_2 = test_image_2d[:, :, ::-1, :]
    pred_2 = predict_model_2d(test_image_2d_2, prediction_model_2d)
    pred_2 = pred_2[:, :, ::-1, :]

    test_image_2d_3 = test_image_2d[:, ::-1, :, :]
    pred_3 = predict_model_2d(test_image_2d_3, prediction_model_2d)
    pred_3 = pred_3[:, ::-1, :, :]

    test_image_2d_4 = test_image_2d[:, ::-1, ::-1, :]
    pred_4 = predict_model_2d(test_image_2d_4, prediction_model_2d)
    pred_4 = pred_4[:, ::-1, ::-1, :]

    pred_stack = np.stack((pred_1, pred_2, pred_3, pred_4), axis=0)

    pred = np.mean(pred_stack, axis=0)
    return pred


def predict_model_3d(test_image_3d, prediction_model_3d):
    pred_batch_size = config["pred_batch_size"]
    batch_test_image_3d = np.array([test_image_3d])

    if config["pred_model_type"] == "native TF FP32":
        batch_prediction = prediction_model_3d.predict(batch_test_image_3d, batch_size=pred_batch_size, verbose=0)
    else:
        batch_prediction = prediction_model_3d(tf.convert_to_tensor(batch_test_image_3d, dtype='float32'))[0].numpy()

    pred = batch_prediction[0]

    return pred


def predict_model_3d_tta_mirroring(test_image_3d, prediction_model_3d):
    test_image_3d_1 = test_image_3d
    pred_1 = predict_model_3d(test_image_3d_1, prediction_model_3d)

    test_image_3d_2 = test_image_3d[:, :, ::-1, :]
    pred_2 = predict_model_3d(test_image_3d_2, prediction_model_3d)
    pred_2 = pred_2[:, :, ::-1, :]

    test_image_3d_3 = test_image_3d[:, ::-1, :, :]
    pred_3 = predict_model_3d(test_image_3d_3, prediction_model_3d)
    pred_3 = pred_3[:, ::-1, :, :]

    test_image_3d_4 = test_image_3d[:, ::-1, ::-1, :]
    pred_4 = predict_model_3d(test_image_3d_4, prediction_model_3d)
    pred_4 = pred_4[:, ::-1, ::-1, :]

    test_image_3d_5 = test_image_3d[::-1, :, :, :]
    pred_5 = predict_model_3d(test_image_3d_5, prediction_model_3d)
    pred_5 = pred_5[::-1, :, :, :]

    test_image_3d_6 = test_image_3d[::-1, :, ::-1, :]
    pred_6 = predict_model_3d(test_image_3d_6, prediction_model_3d)
    pred_6 = pred_6[::-1, :, ::-1, :]

    test_image_3d_7 = test_image_3d[::-1, ::-1, :, :]
    pred_7 = predict_model_3d(test_image_3d_7, prediction_model_3d)
    pred_7 = pred_7[::-1, ::-1, :, :]

    test_image_3d_8 = test_image_3d[::-1, ::-1, ::-1, :]
    pred_8 = predict_model_3d(test_image_3d_8, prediction_model_3d)
    pred_8 = pred_8[::-1, ::-1, ::-1, :]

    pred_stack = np.stack((pred_1, pred_2, pred_3, pred_4, pred_5, pred_6, pred_7, pred_8), axis=0)

    pred = np.mean(pred_stack, axis=0)
    return pred


if __name__ == "__main__":
    test_type = "Validation"
    pred_clear_session = True
    if config["prediction_type"] == "Individual":
        result_dir = os.path.join(os.getcwd(), 'results', str(config["prediction_year"]),
                                  'Segmentation', config["seg_model_name"],
                                  config["network_view"] + "_" + str(config["seg_val_fold"]))
    elif config["prediction_type"] == "Ensemble":
        result_dir = os.path.join(os.getcwd(), 'results', str(config["prediction_year"]),
                                  'Segmentation', config["seg_model_name"],
                                  config["ensemble_name"])
    if config["tta"]:
        result_dir = result_dir + "_tta"

    if config["postprocessing"]:
        result_dir = result_dir + "_post"

    if config["gpu_cpu"] == "GPU" and config["gpu_mem"] == "hard limit":
        result_dir = result_dir + "_" + str(config["gpu_mem_limit"]) + "MB"
    elif config["gpu_cpu"] == "CPU":
        result_dir = result_dir + "_cpu"

    if config["pred_model_type"] == "TF-TRT FP32":
        result_dir = result_dir + "_TFTRT_FP32"
    elif config["pred_model_type"] == "TF-TRT FP16":
        result_dir = result_dir + "_TFTRT_FP16"

    result_dir = os.path.join(result_dir, test_type)
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    paths_test = glob(config["test_path"] + "/*/")

    num_patients = len(paths_test)

    pred_epoch = config["seg_pred_epoch"]
    prediction_model_name = config["seg_prediction_model_name"] + f"-{pred_epoch:04d}"
    brain_seg_pred = Prediction(test_type=test_type,
                                pred_clear_session=pred_clear_session,
                                prediction_model_name=prediction_model_name,
                                result_dir=result_dir)

    brain_seg_pred.predict_multiple_volumes(paths_test)
