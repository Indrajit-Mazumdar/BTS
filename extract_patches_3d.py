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
        self.num_patches = 0

        self.radius = (config["d_patch"] / 32, config["h_patch"] / 32, config["w_patch"] / 32)
        self.base_locs_ax = generate_patch_locations(
            bbox_shape=(75, 85, 90),
            input_shape=(config["d_input"], config["h_input"], config["w_input"]),
            radius=self.radius)
        self.base_locs_sg = generate_patch_locations(
            bbox_shape=(90, 75, 85),
            input_shape=(config["w_input"], config["d_input"], config["h_input"]),
            radius=self.radius)
        self.base_locs_cr = generate_patch_locations(
            bbox_shape=(85, 75, 90),
            input_shape=(config["h_input"], config["d_input"], config["w_input"]),
            radius=self.radius)

    def extract_patches(self, patient_images):
        patient_gt = patient_images[4]

        if config["network_view"] == "Axial":
            base_locs = self.base_locs_ax
        elif config["network_view"] == "Sagittal":
            base_locs = self.base_locs_sg
        elif config["network_view"] == "Coronal":
            base_locs = self.base_locs_cr

        z, y, x = perturb_patch_locations(base_locs, self.radius)
        probs = generate_patch_probs(patient_gt, (z, y, x))
        selections = np.random.choice(range(len(probs)),
                                      size=config["3D_patches_per_patient"],
                                      replace=False,
                                      p=probs)
        for sel in selections:
            k, i, j = np.unravel_index(sel, (len(z), len(y), len(x)))
            x_patch = patient_images[
                      0:4,
                      int(z[k] - config["d_patch"] / 2): int(z[k] + config["d_patch"] / 2),
                      int(y[i] - config["h_patch"] / 2): int(y[i] + config["h_patch"] / 2),
                      int(x[j] - config["w_patch"] / 2): int(x[j] + config["w_patch"] / 2)]
            y_label = patient_gt[
                      int(z[k] - config["d_patch"] / 2): int(z[k] + config["d_patch"] / 2),
                      int(y[i] - config["h_patch"] / 2): int(y[i] + config["h_patch"] / 2),
                      int(x[j] - config["w_patch"] / 2): int(x[j] + config["w_patch"] / 2)]

            x_patch = np.transpose(x_patch, (1, 2, 3, 0)).astype(np.float32)

            y_label[y_label == 4] = 3

            y_label = y_label.reshape(-1)
            y_label = one_hot_encoder(y_label, num_classes=config["num_classes"]).astype(np.uint8)
            y_label = y_label.reshape(config["d_patch"], config["h_patch"], config["w_patch"], config["num_classes"])

            np.save(
                os.path.join(self.save_images_dir, 'x_{}'.format(str(self.num_patches + 1).zfill(5))),
                x_patch)
            np.save(
                os.path.join(self.save_masks_dir, 'y_{}'.format(str(self.num_patches + 1).zfill(5))),
                y_label)
            del x_patch, y_label
            self.num_patches += 1

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


def generate_patch_locations(bbox_shape, input_shape, radius):
    nz = round(
        (config["3D_patches_per_patient"] * 8 * input_shape[0] *
         input_shape[0] / input_shape[1] / input_shape[2]) ** (1.0 / 3))
    ny = round(nz * input_shape[1] / input_shape[0])
    nx = round(nz * input_shape[2] / input_shape[0])

    bbox_shape = tuple(map(sum, zip(bbox_shape, radius)))
    z = np.rint(np.linspace(bbox_shape[0], input_shape[0] - bbox_shape[0], num=nz))
    y = np.rint(np.linspace(bbox_shape[1], input_shape[1] - bbox_shape[1], num=ny))
    x = np.rint(np.linspace(bbox_shape[2], input_shape[2] - bbox_shape[2], num=nx))
    return z, y, x


def perturb_patch_locations(patch_locations, radius):
    z, y, x = patch_locations
    z = np.rint(z + np.random.uniform(-radius[0], radius[0], len(z)))
    y = np.rint(y + np.random.uniform(-radius[1], radius[1], len(y)))
    x = np.rint(x + np.random.uniform(-radius[2], radius[2], len(x)))
    return z, y, x


def generate_patch_probs(patient_gt, patch_locations):
    z, y, x = patch_locations
    p = []
    for k in range(len(z)):
        for i in range(len(y)):
            for j in range(len(x)):
                patch = patient_gt[
                        int(z[k] - config["d_patch"] / 2): int(z[k] + config["d_patch"] / 2),
                        int(y[i] - config["h_patch"] / 2): int(y[i] + config["h_patch"] / 2),
                        int(x[j] - config["w_patch"] / 2): int(x[j] + config["w_patch"] / 2)]
                patch = (patch > 0).astype(np.float32)
                percent = np.sum(patch) / (config["d_patch"] * config["h_patch"] * config["w_patch"])
                p.append((1 - np.abs(percent - 0.5)) * percent)
    p = np.asarray(p, dtype=np.float32)
    p[p == 0] = np.amin(p[np.nonzero(p)])
    p = p / np.sum(p)
    return p


if __name__ == '__main__':
    paths_train = glob(config["train_brats_path"] + "/*/")

    np.random.shuffle(paths_train)

    save_dir = os.path.join(config["preprocessed_data_dir"], "3D", config["network_view"])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    pipe = PatchExtraction(save_dir=save_dir)
    num_patches = pipe.generate_patches(paths=paths_train)
