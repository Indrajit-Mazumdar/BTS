import os
from collections import OrderedDict
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor


def extract_resection_feature(resection_status):
    dict = OrderedDict()

    if resection_status == "GTR":
        dict["ResectionStatus"] = 2
    elif resection_status == "STR":
        dict["ResectionStatus"] = 1
    elif resection_status == "NA":
        dict["ResectionStatus"] = 0

    return dict


def extract_handcrafted_features(modalities, seg_mask_path):
    params = os.path.join(os.getcwd(), 'utils', 'Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    params_shape = os.path.join(os.getcwd(), 'utils', 'Params_shape.yaml')
    extractor_shape = featureextractor.RadiomicsFeatureExtractor(params_shape)

    seg_img = sitk.ReadImage(seg_mask_path)
    seg_arr = sitk.GetArrayFromImage(seg_img)

    labels = ['ET', 'NCR', 'ED', 'TC', 'WT']

    dict = OrderedDict()
    for label in labels:
        seg = np.zeros_like(seg_arr)
        if label == 'ET':
            seg[seg_arr == 4] = 1
        elif label == 'NCR':
            seg[seg_arr == 1] = 1
        elif label == 'ED':
            seg[seg_arr == 2] = 1
        elif label == 'TC':
            for lbl in [1, 4]:
                seg[seg_arr == lbl] = 1
        elif label == 'WT':
            seg[seg_arr > 0] = 1
        seg_mask = sitk.GetImageFromArray(seg)
        if np.count_nonzero(seg) > 0:
            for mod_key, mod_value in modalities.items():

                radiomic_dict = extractor.execute(imageFilepath=mod_value,
                                                  maskFilepath=seg_mask)
                for key in radiomic_dict:
                    feature = key.split("_", 1)[1]
                    dict[label + "_" + mod_key + "_" + feature] = radiomic_dict[key]
            radiomic_shape_dict = extractor_shape.execute(imageFilepath=modalities['T1'],
                                                          maskFilepath=seg_mask)
            for key in radiomic_shape_dict:
                feature = key.split("_", 1)[1]
                dict[label + "_Seg_" + feature] = radiomic_shape_dict[key]
    return dict
