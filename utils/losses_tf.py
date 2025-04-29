import numpy as np
from statistics import mean
from scipy.ndimage import distance_transform_edt
import tensorflow as tf
from tensorflow_addons.image import euclidean_dist_transform
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

from utils.configuration import config


class LossWeightsUpdater(Callback):
    def __init__(self, lambda_1, lambda_2):
        super(LossWeightsUpdater, self).__init__()

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.lst_loss_0 = []
        self.lst_loss_1 = []
        self.lst_loss_2 = []

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())

        mean_loss_0 = mean(self.lst_loss_0)

        mean_loss_1 = mean(self.lst_loss_1)

        mean_loss_2 = mean(self.lst_loss_2)

        K.set_value(self.lambda_2, mean_loss_0 / mean_loss_2)

        self.lst_loss_0 = []
        self.lst_loss_1 = []
        self.lst_loss_2 = []


lambda_1 = K.variable(config["lambda_1"])
lambda_2 = K.variable(config["lambda_2"])

loss_wt_instance = LossWeightsUpdater(lambda_1, lambda_2)


def dice_score_metric(gt, pred):
    sum_gp = K.sum(gt * pred)
    numerator = 2 * sum_gp
    sum_g = K.sum(gt)
    sum_p = K.sum(pred)
    denominator = sum_g + sum_p

    dice_score = numerator / (denominator + K.epsilon())
    return dice_score


def dice_ET_metric(gt, pred):
    num_classes = K.int_shape(pred)[-1]
    gt_reshaped = K.reshape(gt, shape=(-1, num_classes))
    pred_reshaped = K.reshape(pred, shape=(-1, num_classes))
    gt_ET = gt_reshaped[:, -1]
    pred_ET = pred_reshaped[:, -1]
    dice_ET = dice_score_metric(gt_ET, pred_ET)
    return dice_ET


def dice_WT_metric(gt, pred):
    num_classes = K.int_shape(pred)[-1]
    gt_reshaped = K.reshape(gt, shape=(-1, num_classes))
    pred_reshaped = K.reshape(pred, shape=(-1, num_classes))
    gt_WT = K.sum(gt_reshaped[:, 1:], axis=1)
    pred_WT = K.sum(pred_reshaped[:, 1:], axis=1)
    dice_WT = dice_score_metric(gt_WT, pred_WT)
    return dice_WT


def dice_TC_metric(gt, pred):
    num_classes = K.int_shape(pred)[-1]
    gt_reshaped = K.reshape(gt, shape=(-1, num_classes))
    pred_reshaped = K.reshape(pred, shape=(-1, num_classes))

    gt_TC = K.sum(tf.gather(gt_reshaped, [1, 3], axis=1), axis=1)
    pred_TC = K.sum(tf.gather(pred_reshaped, [1, 3], axis=1), axis=1)

    dice_TC = dice_score_metric(gt_TC, pred_TC)
    return dice_TC


def soft_dice_loss(gt, pred):
    gt_flattened = K.flatten(gt)
    pred_flattened = K.flatten(pred)
    sum_gp = K.sum(gt_flattened * pred_flattened)
    numerator = 2 * sum_gp
    sum_g_square = K.sum(K.square(gt_flattened))
    sum_p_square = K.sum(K.square(pred_flattened))
    denominator = sum_g_square + sum_p_square

    dice = numerator / (denominator + K.epsilon())

    loss = 1 - dice
    return loss


def focal_loss(gt, pred):
    pred /= K.sum(pred, axis=-1, keepdims=True)

    pred = K.clip(pred, min_value=K.epsilon(), max_value=1 - K.epsilon())

    gamma = config["focusing_param_FL"]
    modulating_factor = K.pow((1 - pred), gamma)

    loss = -modulating_factor * gt * K.log(pred)
    loss = K.mean(K.sum(loss, axis=-1))
    return loss


def np_euclidean_distance_transform_hd(segmentation):
    seg_shp = segmentation.shape
    dt = np.zeros(seg_shp)
    for i in range(seg_shp[0]):
        img = segmentation[i]
        positive_mask = img
        negative_mask = ~positive_mask
        dt[i] = distance_transform_edt(positive_mask) + distance_transform_edt(negative_mask)
    return dt


def hausdorff_distance_dt_loss(gt, pred):
    num_classes = K.int_shape(pred)[-1]

    pred_idx_top = tf.math.top_k(pred, k=1).indices
    pred_idx_top = tf.squeeze(pred_idx_top, axis=[-1])
    pred_one_hot = tf.one_hot(indices=pred_idx_top, depth=num_classes)

    loss = tf.Variable([])
    for c in range(num_classes):
        gt_c = gt[..., c]

        pred_c = pred[..., c]

        mismatch_c = K.square(gt_c - pred_c)

        pred_bar_c = pred_one_hot[..., c]

        dt_gt_c = np_euclidean_distance_transform_hd(gt_c.numpy().astype(np.uint8))
        dt_gt_c = tf.convert_to_tensor(dt_gt_c, dtype=tf.float32)
        dt_pred_c = np_euclidean_distance_transform_hd(pred_bar_c.numpy().astype(np.uint8))
        dt_pred_c = tf.convert_to_tensor(dt_pred_c, dtype=tf.float32)

        alpha = config["alpha_param_HD"]
        dist_c = K.pow(dt_gt_c, alpha) + K.pow(dt_pred_c, alpha)
        loss_c = K.mean(mismatch_c * dist_c)

        loss = tf.concat([loss, [loss_c]], axis=0)
    loss = K.mean(loss)
    return loss


def loss_function(gt, pred):
    lambda_1 = loss_wt_instance.lambda_1
    lambda_2 = loss_wt_instance.lambda_2

    loss_0 = soft_dice_loss(gt, pred)
    loss_1 = focal_loss(gt, pred)
    loss_2 = hausdorff_distance_dt_loss(gt, pred)
    loss_wt_instance.lst_loss_0.append(K.get_value(loss_0))
    loss_wt_instance.lst_loss_1.append(K.get_value(loss_1))
    loss_wt_instance.lst_loss_2.append(K.get_value(loss_2))
    loss = loss_0 + lambda_1 * loss_1 + lambda_2 * loss_2

    return loss
