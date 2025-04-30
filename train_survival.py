import random
import os
import math
import sys
import time
from collections import OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import joblib
import json
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression, SelectFromModel
from sklearn.pipeline import Pipeline
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

    surv_train_file_name = 'survival_train.csv'
    BraTS_ID = 'Brats' + str(config["training_year"])[-2:] + 'ID'
    surv_col_name = 'Survival_days'
    resection_col_name = 'Extent_of_Resection'

    model_general_dir = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]),
                                     'Survival', config["feature_type"])
    if not os.path.isdir(model_general_dir):
        os.makedirs(model_general_dir)

    model_specific_dir = os.path.join(model_general_dir, config["survival_model"])
    if not os.path.isdir(model_specific_dir):
        os.makedirs(model_specific_dir)

    training_data_path = config["training_data_path"]

    training_patients_paths = []
    paths = []
    for dirpath, dirnames, files in os.walk(training_data_path):
        training_patients_paths.append(dirpath)

    survival_data = pd.read_csv(os.path.join(training_data_path, surv_train_file_name),
                                na_filter=False)
    survival_data = survival_data[survival_data[surv_col_name].astype(str).str.isnumeric()]

    training_features = pd.DataFrame()
    if config["feature_type"] == "Handcrafted features":

        for index, row in tqdm(survival_data.iterrows(), desc="Feature Extraction", total=survival_data.shape[0]):
            feature_vector = row.drop(labels=[resection_col_name])

            resection_feature_dict = extract_resection_feature(row[resection_col_name])
            resection_feature = pd.Series(resection_feature_dict)
            feature_vector = feature_vector.append(resection_feature)

            t1_image_path, t1ce_image_path, t2_image_path, flair_image_path, seg_mask_path = "", "", "", "", ""
            for path in training_patients_paths:
                patient_id = os.path.basename(path)
                if row[BraTS_ID] == os.path.basename(path):
                    t1_image_path = os.path.join(path, patient_id + '_t1.nii.gz')
                    t1ce_image_path = os.path.join(path, patient_id + '_t1ce.nii.gz')
                    t2_image_path = os.path.join(path, patient_id + '_t2.nii.gz')
                    flair_image_path = os.path.join(path, patient_id + '_flair.nii.gz')
                    seg_mask_path = os.path.join(path, patient_id + '_seg.nii.gz')
                    break

            modalities = OrderedDict()
            modalities['T1'] = t1_image_path
            modalities['T1ce'] = t1ce_image_path
            modalities['T2'] = t2_image_path
            modalities['FLAIR'] = flair_image_path

            handcrafted_features_dict = extract_handcrafted_features(modalities, seg_mask_path)
            handcrafted_features = pd.Series(handcrafted_features_dict)
            feature_vector = feature_vector.append(handcrafted_features)

            training_features = training_features.join(feature_vector)

        training_features.reset_index(drop=True, inplace=True)
        training_features = training_features.replace(['NA', np.inf], np.nan)
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
            for path in training_patients_paths:
                patient_id = os.path.basename(path)
                if row[BraTS_ID] == os.path.basename(path):
                    t1_image_path = os.path.join(path, patient_id + '_t1.nii.gz')
                    t1ce_image_path = os.path.join(path, patient_id + '_t1ce.nii.gz')
                    t2_image_path = os.path.join(path, patient_id + '_t2.nii.gz')
                    flair_image_path = os.path.join(path, patient_id + '_flair.nii.gz')
                    seg_mask_path = os.path.join(path, patient_id + '_seg.nii.gz')
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

            training_features = training_features.join(feature_vector)

        training_features = training_features.T
        training_features.reset_index(drop=True, inplace=True)
        cols = training_features.columns.tolist()
        training_features.columns = ['DF' + str(x) if isinstance(x, int) else x for x in cols]
        training_features = training_features.replace(['NA', np.inf], np.nan)

    num_extracted_features = len(
        training_features.columns) - 2

    x_train = training_features.drop(columns=[BraTS_ID, surv_col_name])
    y_train = training_features[surv_col_name]

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=x_train.columns)

    joblib.dump(imputer, os.path.join(model_general_dir, config["imputer_name"]))

    scaler = StandardScaler()
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns)

    joblib.dump(scaler, os.path.join(model_general_dir, config["scaler_name"]))

    if config["feature_selection"] == "Pearson correlation coefficient":
        feature_selector = SelectKBest(score_func=Pearson_correlation_coefficient)
        feature_selector_scores = SelectKBest(score_func=Pearson_correlation_coefficient, k=num_extracted_features)
    elif config["feature_selection"] == "Mutual information":
        feature_selector = SelectKBest(
            score_func=partial(mutual_info_regression, discrete_features='auto', n_neighbors=3))
        feature_selector_scores = SelectKBest(
            score_func=partial(mutual_info_regression, discrete_features='auto', n_neighbors=3),
            k=num_extracted_features)

    feature_names = pd.DataFrame(x_train.columns.tolist())
    feature_selector_scores.fit(x_train, y_train)
    scores = pd.DataFrame(feature_selector_scores.scores_)
    feature_scores = pd.concat([feature_names, scores], axis=1, ignore_index=True)
    feature_scores.columns = ['Feature', 'Score']
    feature_scores = feature_scores.sort_values(by=['Score'], ascending=False)

    params = {'booster': config["xgb_booster"],
              'disable_default_eval_metric': config["xgb_disable_default_eval_metric"],
              'eta': config["xgb_eta"],
              'gamma': config["xgb_gamma"],
              'max_depth': config["xgb_max_depth"],
              'min_child_weight': config["xgb_min_child_weight"],
              'subsample': config["xgb_subsample"],
              'colsample_bytree': config["xgb_colsample_bytree"],
              'lambda': config["xgb_lambda"],
              'objective': config["xgb_objective"]}
    num_boost_round = config["xgb_num_boost_round"]

    min_MSE = float('inf')

    feature_selector__k = config["num_selected_features"]
    feature_selector.fit(x_train, y_train)
    joblib.dump(feature_selector, os.path.join(model_specific_dir, config["feature_selector_name"]))
    mask = feature_selector.get_support()
    selected_features = x_train.columns[mask]
    x_train = pd.DataFrame(feature_selector.transform(x_train), columns=selected_features)
    feature_names = x_train.columns.tolist()
    dtrain = xgb.DMatrix(data=x_train.to_numpy(),
                         label=y_train.to_numpy(),
                         missing=np.nan,
                         feature_names=feature_names)
    model = xgb.train(params=params,
                      dtrain=dtrain,
                      num_boost_round=num_boost_round)

    model.save_model(os.path.join(model_specific_dir, config["xgb_model_name"]))
