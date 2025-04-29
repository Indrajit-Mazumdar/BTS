import warnings
from nipype.interfaces.ants import N4BiasFieldCorrection
import SimpleITK as sitk
import numpy as np

from utils.configuration import config


def normalize(original_img):
    normalized_img = np.zeros((config["num_modalities"], config["d_input"],
                               config["h_input"], config["w_input"])).astype(np.float32)
    for mod_idx in range(config["num_modalities"]):
        img = original_img[mod_idx]

        img_nonzero = img[np.nonzero(img)]

        if np.std(img) == 0 or np.std(img_nonzero) == 0:
            normalized_img[mod_idx] = img
        else:
            norm_img = (img - np.mean(img_nonzero)) / np.std(img_nonzero)
            norm_img[np.where(img == 0)] = 0
            normalized_img[mod_idx] = norm_img

    return normalized_img


def correct_bias(input_path, output_path):
    n4 = N4BiasFieldCorrection()
    n4.inputs.input_image = input_path
    n4.inputs.output_image = output_path
    try:
        n4.run()
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found. "
                                     "Will try using SimpleITK for bias field correction "
                                     "which will take much longer. To fix this problem, "
                                     "add N4BiasFieldCorrection to your PATH system variable "
                                     "(example: EXPORT ${PATH}:/path/to/ants/bin)."))
        output_image = sitk.N4BiasFieldCorrection(sitk.ReadImage(input_path))
        sitk.WriteImage(output_image, output_path)


def one_hot_encoder(y, num_classes):
    y = np.array(y, dtype="int64")
    num_locations = y.shape[0]
    one_hot_array = np.zeros((num_locations, num_classes))
    one_hot_array[np.arange(num_locations), y] = 1
    return one_hot_array
