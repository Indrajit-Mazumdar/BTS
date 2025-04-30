import json
import os
import sys
import math
import numpy as np
from glob import glob
import random
import time
import matplotlib.pyplot as plt
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

from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import mixed_precision

from utils.data_augmentations import *
from utils.losses_tf import *
from networks.esa_net_2d import EncoderDecoderModel

K.set_image_data_format("channels_last")

seed_value = 0

os.environ['PYTHONHASHSEED'] = str(seed_value)

random.seed(seed_value)

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

if config["mixed_precision_training"]:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)


class LearningRateDecay(Callback):
    def __init__(self, initial_learning_rate, max_epochs, poly_power):
        super(LearningRateDecay, self).__init__()

        self.max_epochs = max_epochs
        self.initial_learning_rate = initial_learning_rate
        self.poly_power = poly_power

    def on_epoch_end(self, epoch, logs=None):
        if config["lr_schedule"] is not None:
            if config["lr_schedule"] == "Polynomial":
                decay = (1 - (epoch / self.max_epochs)) ** self.poly_power
            elif config["lr_schedule"] == "Linear":
                decay = (1 - (epoch / self.max_epochs)) ** 1.0
            decayed_learning_rate = self.initial_learning_rate * decay

            K.set_value(self.model.optimizer.learning_rate, decayed_learning_rate)


initial_learning_rate = K.variable(config["init_lr"])
max_epochs = K.variable(config["num_epochs"])
poly_power = K.variable(config["poly_power"])

lr_poly_decay_instance = LearningRateDecay(initial_learning_rate, max_epochs, poly_power)


class Training(object):
    def __init__(self, model_dir, training_model_name, csv_logger_name):
        self.model_dir = model_dir
        self.training_model_name = training_model_name
        self.csv_logger_name = csv_logger_name
        resume_training = False
        if config["save_only_best_model"]:
            training_model_path = os.path.join(model_dir, self.training_model_name)
            if os.path.isdir(training_model_path):
                resume_training = True
        else:
            epochs = np.arange(config["save_period"], config["num_epochs"] + 1, config["save_period"])
            for epoch in np.flip(epochs):
                training_model_path = os.path.join(model_dir, config["seg_training_model_name"] + f"-{epoch:04d}")
                if os.path.isdir(training_model_path):
                    resume_training = True
                    self.init_epoch = epoch
                    break
        if resume_training:
            custom_objects = {'GroupNormalization': GroupNormalization,
                              'dice_ET_metric': dice_ET_metric,
                              'dice_WT_metric': dice_WT_metric,
                              'dice_TC_metric': dice_TC_metric}
            self.training_model = load_model(filepath=training_model_path,
                                             custom_objects=custom_objects,
                                             compile=False)

            lr = K.get_value(self.training_model.optimizer.learning_rate)
        else:
            training_network = EncoderDecoderModel(
                img_shape=(config["h_patch"], config["w_patch"], config["num_modalities"]))
            self.training_model = training_network.model

            self.init_epoch = 0
            lr = config["init_lr"]

        opt = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

        self.training_model.compile(optimizer=opt,
                                    loss=loss_function,
                                    metrics=[dice_ET_metric, dice_WT_metric, dice_TC_metric])

    def data_generator(self, x_paths, y_paths, batch_size, augment):
        sample_size = len(x_paths)

        batch_idx_start = 0
        batch_idx_end = batch_idx_start + batch_size
        while True:
            batch_x, batch_y = [], []
            idx = batch_idx_start
            cnt = 0
            while cnt < batch_size:
                if idx == 0:
                    shuffle = list(zip(x_paths, y_paths))
                    np.random.shuffle(shuffle)
                    x_paths = list([shuffle[i][0] for i in range(len(shuffle))])
                    y_paths = list([shuffle[i][1] for i in range(len(shuffle))])
                    del shuffle

                    if augment:
                        rotate_prob = np.random.randint(2, size=sample_size)
                        horizontal_flip_prob = np.random.randint(2, size=sample_size)
                        vertical_flip_prob = np.random.randint(2, size=sample_size)

                image = np.load(x_paths[idx]).astype(np.float32)
                mask = np.load(y_paths[idx]).astype(np.float32)
                if augment:
                    if rotate_prob[idx]:
                        image, mask = rotate_2D(image, mask)
                    if horizontal_flip_prob[idx]:
                        image, mask = horizontal_flip_2D(image, mask)
                    if vertical_flip_prob[idx]:
                        image, mask = vertical_flip_2D(image, mask)
                batch_x.append(image)
                batch_y.append(mask)
                idx = idx + 1
                cnt = cnt + 1
                if idx == sample_size:
                    idx = 0

            x_batch = np.stack(batch_x, axis=0)
            y_batch = np.stack(batch_y, axis=0)

            batch_idx_start = batch_idx_end
            if batch_idx_end == sample_size:
                batch_idx_start = 0
            batch_idx_end = batch_idx_start + batch_size
            if batch_idx_end > sample_size:
                batch_idx_end = batch_idx_end - sample_size

            yield x_batch, y_batch

    def fit_model(self, x_train_paths, y_train_paths, x_val_paths, y_val_paths):
        num_train_patches = len(x_train_paths)
        num_val_patches = len(x_val_paths)

        steps_per_epoch = config["steps_per_epoch"]

        if config["save_only_best_model"]:
            checkpoint = ModelCheckpoint(filepath=os.path.join(self.model_dir, self.training_model_name),
                                         monitor='val_loss',
                                         verbose=0,
                                         save_best_only=True,
                                         save_weights_only=False,
                                         mode='min',
                                         save_freq='epoch')
        else:
            checkpoint = ModelCheckpoint(
                filepath=os.path.join(self.model_dir, self.training_model_name) + "-{epoch:04d}",
                monitor='val_loss',
                verbose=0,
                save_best_only=False,
                save_weights_only=False,
                mode='min',
                save_freq=int(config["save_period"] * steps_per_epoch))
        csv_logger = CSVLogger(os.path.join(self.model_dir, self.csv_logger_name), append=True)

        train_generator = self.data_generator(x_paths=x_train_paths,
                                              y_paths=y_train_paths,
                                              batch_size=config["batch_size"],
                                              augment=True)

        validation_generator = self.data_generator(x_paths=x_val_paths,
                                                   y_paths=y_val_paths,
                                                   batch_size=config["batch_size"],
                                                   augment=True)

        self.training_model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=config["num_epochs"],
            verbose=2,
            callbacks=[checkpoint, csv_logger, loss_wt_instance, lr_poly_decay_instance],
            validation_data=validation_generator,
            validation_steps=num_val_patches // config["batch_size"],
            initial_epoch=self.init_epoch,
            workers=1,
            use_multiprocessing=False)

        df = pd.read_csv(os.path.join(self.model_dir, self.csv_logger_name))

        best_model_epoch = df['val_loss'].argmin()


if __name__ == "__main__":
    model_dir = os.path.join(os.getcwd(), 'trained_models', str(config["training_year"]),
                             'Segmentation', config["seg_model_name"],
                             config["network_view"] + "_" + str(config["seg_val_fold"]))
    if config["gpu_mem"] == "hard limit":
        model_dir = model_dir + "_" + str(config["gpu_mem_limit"]) + "MB"
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    csv_logger_name = "training.log"

    if config["network_view"] == "Axial":
        prediction_network = EncoderDecoderModel(
            img_shape=(config["h_input"], config["w_input"], config["num_modalities"]))
    elif config["network_view"] == "Sagittal":
        prediction_network = EncoderDecoderModel(
            img_shape=(config["d_input"], config["h_input"], config["num_modalities"]))
    elif config["network_view"] == "Coronal":
        prediction_network = EncoderDecoderModel(
            img_shape=(config["d_input"], config["w_input"], config["num_modalities"]))
    prediction_model = prediction_network.model

    x_val_paths = glob(config["val_path"] + "/images/*")
    y_val_paths = glob(config["val_path"] + "/masks/*")
    x_train_paths = glob(config["train_path"] + "/images/*")
    y_train_paths = glob(config["train_path"] + "/masks/*")

    brain_seg = Training(model_dir=model_dir,
                         training_model_name=config["seg_training_model_name"],
                         csv_logger_name=csv_logger_name)

    brain_seg.fit_model(x_train_paths, y_train_paths, x_val_paths, y_val_paths)

    if config["save_only_best_model"]:
        prediction_model.set_weights(brain_seg.training_model.get_weights())
        assert len(brain_seg.training_model.weights) == len(prediction_model.weights)
        for a, b in zip(brain_seg.training_model.weights, prediction_model.weights):
            np.testing.assert_allclose(a.numpy(), b.numpy())

        prediction_model.save(filepath=os.path.join(model_dir, config["seg_prediction_model_name"]))
    else:
        epochs = np.arange(config["save_period"], config["num_epochs"] + 1, config["save_period"])
        custom_objects = {'GroupNormalization': GroupNormalization,
                          'dice_ET_metric': dice_ET_metric,
                          'dice_WT_metric': dice_WT_metric,
                          'dice_TC_metric': dice_TC_metric}
        for epoch in epochs:
            training_model = load_model(
                filepath=os.path.join(model_dir, config["seg_training_model_name"] + f"-{epoch:04d}"),
                custom_objects=custom_objects,
                compile=False)

            prediction_model.set_weights(training_model.get_weights())
            assert len(training_model.weights) == len(prediction_model.weights)
            for a, b in zip(training_model.weights, prediction_model.weights):
                np.testing.assert_allclose(a.numpy(), b.numpy())

            prediction_model.save(
                filepath=os.path.join(model_dir, config["seg_prediction_model_name"]) + f"-{epoch:04d}")
