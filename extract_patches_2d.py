import os
import sys
import math
from glob import glob
import SimpleITK as sitk
import random
import numpy as np
from tqdm import tqdm

from utils.configuration import config
from utils.preprocessing import *

seed_value = 0

os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)


class PatchExtraction(object):
    def __init__(self, save_dir):
        self.save_images_dir = os.path.join(save_dir, "images")
        if not os.path.isdir(self.save_images_dir):
            os.makedirs(self.save_images_dir)
        self.save_masks_dir = os.path.join(save_dir, "masks")
        if not os.path.isdir(self.save_masks_dir):
            os.makedirs(self.save_masks_dir)
        self.extraction_type = config["extraction_type"]
        self.num_patches = 0

    def extract_patches_cob(self, patient_images):
        d_image = patient_images.shape[1]
        h_image = patient_images.shape[2]
        w_image = patient_images.shape[3]
        patient_gt = patient_images[4]
        for slice_idx in range(d_image):
            arr_lbl = patient_gt[slice_idx, :, :]

            indices_tumor = np.nonzero(arr_lbl)

            num_tumor_elements_img = indices_tumor[0].size
            if num_tumor_elements_img == 0:
                continue

            brain_row_center = h_image // 2
            brain_col_center = w_image // 2

            p_y = (brain_row_center - config["h_patch"] / 2, brain_row_center + config["h_patch"] / 2)
            p_x = (brain_col_center - config["w_patch"] / 2, brain_col_center + config["w_patch"] / 2)
            p_y = list(map(int, p_y))
            p_x = list(map(int, p_x))

            min_row_idx = np.min(indices_tumor[0])
            max_row_idx = np.max(indices_tumor[0])
            min_col_idx = np.min(indices_tumor[1])
            max_col_idx = np.max(indices_tumor[1])

            if min_row_idx < p_y[0]:
                p_y[0] = min_row_idx
                p_y[1] = p_y[0] + config["h_patch"]
            elif max_row_idx >= p_y[1]:
                p_y[1] = max_row_idx + 1
                p_y[0] = p_y[1] - config["h_patch"]
            if min_col_idx < p_x[0]:
                p_x[0] = min_col_idx
                p_x[1] = p_x[0] + config["w_patch"]
            elif max_col_idx >= p_x[1]:
                p_x[1] = max_col_idx + 1
                p_x[0] = p_x[1] - config["w_patch"]

            x_patch = patient_images[0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
            y_label = patient_gt[slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]

            num_WT_patch = np.count_nonzero(y_label > 0)
            if num_WT_patch <= config["2D_tumor_skip_num"]:
                continue

            x_patch = np.transpose(x_patch, (1, 2, 0)).astype(np.float32)

            y_label[y_label == 4] = 3

            y_label = y_label.reshape(-1)

            y_label = one_hot_encoder(y_label, num_classes=config["num_classes"]).astype(np.uint8)
            y_label = y_label.reshape(config["h_patch"], config["w_patch"], config["num_classes"])

            np.save(
                os.path.join(self.save_images_dir, 'x_{}'.format(str(self.num_patches + 1).zfill(7))),
                x_patch)
            np.save(
                os.path.join(self.save_masks_dir, 'y_{}'.format(str(self.num_patches + 1).zfill(7))),
                y_label)
            del x_patch, y_label

            self.num_patches += 1

    def extract_patches_random(self, patient_images):
        patient_patches, patient_labels = [], []
        patient_gt = patient_images[4]
        cnt_patient_patches = 0
        while cnt_patient_patches < config["2D_patches_per_patient"]:
            slice_idx = np.random.randint(1, config["d_patch"])
            arr_lbl = patient_gt[slice_idx, :, :]
            indices_tumor = np.nonzero(arr_lbl)
            num_elements = indices_tumor[0].size
            if num_elements <= config["2D_tumor_skip_num"]:
                continue

            min_tumor_h = np.min(indices_tumor[0])
            max_tumor_h = np.max(indices_tumor[0])
            min_tumor_w = np.min(indices_tumor[1])
            max_tumor_w = np.max(indices_tumor[1])

            patch_h_center = np.random.randint(min_tumor_h, max_tumor_h + 1)
            patch_w_center = np.random.randint(min_tumor_w, max_tumor_w + 1)

            p_y = (patch_h_center - config["h_patch"] / 2, patch_h_center + config["h_patch"] / 2)
            p_x = (patch_w_center - config["w_patch"] / 2, patch_w_center + config["w_patch"] / 2)
            p_y = list(map(int, p_y))
            p_x = list(map(int, p_x))

            if p_y[0] < 1:
                p_y[0] = 1
                p_y[1] = p_y[0] + config["h_patch"]
            elif p_y[1] > config["h_input"]:
                p_y[1] = config["h_input"]
                p_y[0] = p_y[1] - config["h_patch"]
            if p_x[0] < 1:
                p_x[0] = 1
                p_x[1] = p_x[0] + config["w_patch"]
            elif p_x[1] > config["w_input"]:
                p_x[1] = config["w_input"]
                p_x[0] = p_x[1] - config["w_patch"]

            gt = patient_gt[slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]
            indices_tumor = np.nonzero(gt)
            num_elements = indices_tumor[0].size
            if num_elements <= config["2D_tumor_skip_num"]:
                continue

            img = patient_images[0:4, slice_idx, p_y[0]:p_y[1], p_x[0]:p_x[1]]

            patient_patches.append(img)
            patient_labels.append(gt)
            cnt_patient_patches += 1

        x_patches = np.array(patient_patches)
        y_labels = np.array(patient_labels)

        x_patches = np.transpose(x_patches, (0, 2, 3, 1)).astype(np.float32)

        y_labels[y_labels == 4] = 3

        shp = y_labels.shape[0]
        y_labels = y_labels.reshape(-1)
        y_labels = one_hot_encoder(y_labels, num_classes=config["num_classes"]).astype(np.uint8)
        y_labels = y_labels.reshape(shp, config["h_patch"], config["w_patch"], config["num_classes"])

        shuffle = list(zip(x_patches, y_labels))
        np.random.shuffle(shuffle)
        x_patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
        y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
        del shuffle

        for i in range(cnt_patient_patches):
            x_patch = x_patches[i]
            y_label = y_labels[i]
            np.save(
                os.path.join(self.save_images_dir, 'x_{}'.format(str(self.num_patches + 1).zfill(5))),
                x_patch)
            np.save(
                os.path.join(self.save_masks_dir, 'y_{}'.format(str(self.num_patches + 1).zfill(5))),
                y_label)
            del x_patch, y_label
            self.num_patches += 1
        del x_patches, y_labels

    def extract_patches(self, patient_images):
        if self.extraction_type == "center_of_brain":
            self.extract_patches_cob(patient_images)
        elif self.extraction_type == "random_locations":
            self.extract_patches_random(patient_images)

    def generate_patches(self, paths):
        num_patients = len(paths)

        for patient_num in tqdm(range(num_patients), desc="Sampling patches"):

            patient_images = read_images(paths[patient_num])

            if config["network_view"] == "Axial":
                patient_images_ax = patient_images
                self.extract_patches(patient_images_ax)
            elif config["network_view"] == "Sagittal":
                patient_images_sg = np.transpose(patient_images, axes=(0, 3, 1, 2))
                self.extract_patches(patient_images_sg)
            elif config["network_view"] == "Coronal":
                patient_images_cr = np.transpose(patient_images, axes=(0, 2, 1, 3))
                self.extract_patches(patient_images_cr)

        return self.num_patches


def read_images(image_path):
    t1 = glob(image_path + '/*_t1.nii.gz')
    t1ce = glob(image_path + '/*_t1ce.nii.gz')
    t2 = glob(image_path + '/*_t2.nii.gz')
    flair = glob(image_path + '/*_flair.nii.gz')
    gt = glob(image_path + '/*_seg.nii.gz')
    assert ((len(t1) + len(t1ce) + len(t2) + len(flair) + len(gt)) == config["num_mod_gt"]), \
        "There is a problem here. The problem lies in this patient: {}".format(image_path)
    scans = [t1[0], t1ce[0], t2[0], flair[0], gt[0]]

    tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))]

    normalized_img = tmp
    normalized_img[:-1] = normalize(tmp[:-1])

    return np.array(normalized_img)


if __name__ == '__main__':
    paths_train = glob(config["train_brats_path"] + "/*/")

    np.random.shuffle(paths_train)

    save_dir = os.path.join(config["preprocessed_data_dir"], "2D", config["network_view"])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pipe = PatchExtraction(save_dir=save_dir)
    num_patches = pipe.generate_patches(paths=paths_train)
