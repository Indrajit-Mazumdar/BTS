import random
import os
import math
import sys
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
from skimage.transform import resize
import xgboost as xgb
import torch

from utils.configuration import config
from networks.mtc_net_3d import MTCNet3D
from utils.preprocessing import *
from utils.feature_extraction import *
from utils.feature_selection import *
from utils.survival_utils import *

seed_value = 0

os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

torch.manual_seed(seed_value)

torch.cuda.manual_seed_all(seed_value)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.use_deterministic_algorithms(True)

activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def load_pred_model(checkpoint_path, device):
    model = MTCNet3D(in_channels=config["num_modalities"], out_channels=config["num_classes"])

    model = model.to(device)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.bottleneck_block.register_forward_hook(get_activation('bottleneck'))

    model.eval()

    return model


if __name__ == "__main__":
    if config["gpu_cpu"] == "GPU" and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    test_type = 'Validation'

    surv_train_file_name = 'survival_validation.csv'
    BraTS_ID = 'Brats' + str(config["training_year"])[-2:] + 'ID'
    surv_col_name = 'Survival_days'
    resection_col_name = 'Extent_of_Resection'

    model_general_dir = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]),
                                     'Survival', config["feature_type"])

    model_specific_dir = os.path.join(model_general_dir, config["survival_model"])

    res_dir = os.path.join(os.getcwd(), 'results', str(config["prediction_year"]),
                           'Survival', test_type, config["feature_type"], config["survival_model"])
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    survival_data = pd.DataFrame()
    test_data_path = config["test_data_path"]
    test_seg_path = config["test_seg_path"]
    survival_data = pd.read_csv(os.path.join(test_data_path, surv_train_file_name),
                                na_filter=False)

    test_patients_paths = []
    for dirpath, dirnames, files in os.walk(test_data_path):
        test_patients_paths.append(dirpath)

    test_features = pd.DataFrame()
    if config["feature_type"] == "Handcrafted features":
        for index, row in tqdm(survival_data.iterrows(), desc="Feature Extraction", total=survival_data.shape[0]):
            feature_vector = row.drop(labels=[resection_col_name])

            resection_feature_dict = extract_resection_feature(row[resection_col_name])
            resection_feature = pd.Series(resection_feature_dict)
            feature_vector = feature_vector.append(resection_feature)

            t1_image_path, t1ce_image_path, t2_image_path, flair_image_path, seg_mask_path = "", "", "", "", ""
            for path in test_patients_paths:
                patient_id = os.path.basename(path)
                if row[BraTS_ID] == os.path.basename(path):
                    t1_image_path = os.path.join(path, patient_id + '_t1.nii.gz')
                    t1ce_image_path = os.path.join(path, patient_id + '_t1ce.nii.gz')
                    t2_image_path = os.path.join(path, patient_id + '_t2.nii.gz')
                    flair_image_path = os.path.join(path, patient_id + '_flair.nii.gz')
                    seg_mask_path = os.path.join(test_seg_path, patient_id + '.nii.gz')
                    break

            modalities = OrderedDict()
            modalities['T1'] = t1_image_path
            modalities['T1ce'] = t1ce_image_path
            modalities['T2'] = t2_image_path
            modalities['FLAIR'] = flair_image_path

            handcrafted_features_dict = extract_handcrafted_features(modalities, seg_mask_path)
            handcrafted_features = pd.Series(handcrafted_features_dict)
            feature_vector = feature_vector.append(handcrafted_features)

            test_features = test_features.join(feature_vector)

        test_features = test_features.T
        test_features.reset_index(drop=True, inplace=True)
        test_features = test_features.replace(['NA', np.inf], np.nan)
    elif config["feature_type"] == "Deep features":
        seg_model_path = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]),
                                      'Segmentation', config["seg_model_name"], 'checkpoint',
                                      'checkpoint_epoch_{:04d}.pth'.format(config["seg_pred_epoch"]))

        seg_model = load_pred_model(seg_model_path, device)

        for index, row in tqdm(survival_data.iterrows(), desc="Feature Extraction", total=survival_data.shape[0]):
            feature_vector = row.drop(labels=[resection_col_name])

            resection_feature_dict = extract_resection_feature(row[resection_col_name])
            resection_feature = pd.Series(resection_feature_dict)
            feature_vector = feature_vector.append(resection_feature)

            t1_image_path, t1ce_image_path, t2_image_path, flair_image_path, seg_mask_path = "", "", "", "", ""
            for path in test_patients_paths:
                patient_id = os.path.basename(path)
                if row[BraTS_ID] == os.path.basename(path):
                    t1_image_path = os.path.join(path, patient_id + '_t1.nii.gz')
                    t1ce_image_path = os.path.join(path, patient_id + '_t1ce.nii.gz')
                    t2_image_path = os.path.join(path, patient_id + '_t2.nii.gz')
                    flair_image_path = os.path.join(path, patient_id + '_flair.nii.gz')
                    seg_mask_path = os.path.join(test_seg_path, patient_id + '.nii.gz')
                    break

            scans_test = [t1_image_path, t1ce_image_path, t2_image_path, flair_image_path, seg_mask_path]

            test_im = [sitk.GetArrayFromImage(sitk.ReadImage(scans_test[i]))
                       for i in range(len(scans_test))]
            test_im = np.array(test_im).astype(np.float32)

            test_image = test_im[0:4]

            test_image = normalize(test_image)

            seg_mask = test_im[-1]

            indices_tumor = np.nonzero(seg_mask)

            min_d_idx = np.min(indices_tumor[0])
            max_d_idx = np.max(indices_tumor[0])
            min_h_idx = np.min(indices_tumor[1])
            max_h_idx = np.max(indices_tumor[1])
            min_w_idx = np.min(indices_tumor[2])
            max_w_idx = np.max(indices_tumor[2])

            test_image_cropped = test_image[min_d_idx:max_d_idx + 1, min_h_idx:max_h_idx + 1,
                                 min_w_idx:max_w_idx + 1, :]

            test_image_resized = resize(test_image_cropped,
                                        (config["d_resized"], config["h_resized"], config["w_resized"],
                                         config["num_modalities"]),
                                        mode='constant',
                                        preserve_range=True)

            batch_test_image_3d = np.array([test_image_resized])

            batch_test_image_3d = torch.from_numpy(batch_test_image_3d).to(device)

            with torch.no_grad():
                batch_prediction = seg_model(batch_test_image_3d)
                enc_end_activations = activation['bottleneck'].detach().cpu().numpy()

            enc_end_features = enc_end_activations[0]

            enc_end_flattened_features = enc_end_features.flatten()

            deep_features = pd.Series(enc_end_flattened_features)
            feature_vector = feature_vector.append(deep_features)

            test_features = test_features.join(feature_vector)

        test_features = test_features.T
        test_features.reset_index(drop=True, inplace=True)
        cols = test_features.columns.tolist()
        test_features.columns = ['DF' + str(x) if isinstance(x, int) else x for x in cols]
        test_features = test_features.replace(['NA', np.inf], np.nan)

    test_patient_ids = test_features[BraTS_ID]
    test_patient_ids.reset_index(drop=True, inplace=True)

    x_test = test_features.drop(columns=[BraTS_ID])

    imputer = joblib.load(os.path.join(model_general_dir, config["imputer_name"]))

    x_test = pd.DataFrame(imputer.transform(x_test), columns=x_test.columns)

    scaler = joblib.load(os.path.join(model_general_dir, config["scaler_name"]))

    x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)

    feature_selector = joblib.load(os.path.join(model_specific_dir, config["feature_selector_name"]))

    mask = feature_selector.get_support()
    selected_features = x_test.columns[mask]
    x_test_tr = pd.DataFrame(feature_selector.transform(x_test), columns=selected_features)

    model = xgb.Booster()
    model.load_model(os.path.join(model_specific_dir, config["xgb_model_name"]))

    dtest = xgb.DMatrix(data=x_test_tr.to_numpy(), label=None, missing=np.nan)

    y_pred = model.predict(data=dtest)

    survival_test = pd.concat([test_patient_ids, pd.DataFrame(y_pred)],
                              axis=1,
                              ignore_index=True)
    survival_test.columns = [BraTS_ID, surv_col_name]
    survival_test.to_csv(os.path.join(res_dir, 'predicted_survival_values.csv'),
                         header=False,
                         index=False)
