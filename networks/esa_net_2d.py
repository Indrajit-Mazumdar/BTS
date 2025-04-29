import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dropout, SpatialDropout2D, SpatialDropout3D
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import Conv2D, Conv3D, SeparableConv2D, Dense
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D, GlobalAveragePooling2D, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv2DTranspose, Conv3DTranspose, UpSampling2D, UpSampling3D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate, add, multiply, dot, Reshape, Lambda
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

from utils.configuration import config
from utils.padding import *

K.set_image_data_format("channels_last")


class EncoderDecoderModel(object):
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=self.img_shape)
        outputs = self.network(inputs=inputs)
        model = Model(inputs=inputs, outputs=outputs)

        return model

    def network(self, inputs):
        num_filters = config["base_channels"]
        num_filters_lst = [num_filters]
        for _ in range(1, config["num_levels"]):
            num_filters = int(num_filters * 2)
            num_filters_lst.append(num_filters)

        enc_block_1 = self.encoder_block(inputs, num_filters_lst[0])
        dn_smpl_1 = self.downsampling_block(enc_block_1, num_filters_lst[0])

        enc_block_2 = self.encoder_block(dn_smpl_1, num_filters_lst[1])
        dn_smpl_2 = self.downsampling_block(enc_block_2, num_filters_lst[1])

        enc_block_3 = self.encoder_block(dn_smpl_2, num_filters_lst[2])
        dn_smpl_3 = self.downsampling_block(enc_block_3, num_filters_lst[2])

        enc_block_4 = self.encoder_block(dn_smpl_3, num_filters_lst[3])

        up_smpl_3 = self.upsampling_block(enc_block_4, num_filters_lst[2])
        skip_3 = self.skip_connection(enc_block_3, up_smpl_3)
        dec_block_3 = self.decoder_block(skip_3, num_filters_lst[2])

        up_smpl_2 = self.upsampling_block(dec_block_3, num_filters_lst[1])
        skip_2 = self.skip_connection(enc_block_2, up_smpl_2)
        dec_block_2 = self.decoder_block(skip_2, num_filters_lst[1])

        up_smpl_1 = self.upsampling_block(dec_block_2, num_filters_lst[0])
        skip_1 = self.skip_connection(enc_block_1, up_smpl_1)
        dec_block_1 = self.decoder_block(skip_1, num_filters_lst[0])

        conv_out = Conv2D(filters=config["num_classes"],
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          padding="same",
                          kernel_initializer="he_normal",
                          bias_initializer="zeros",
                          kernel_regularizer=regularizers.l2(config["weight_decay"]))(dec_block_1)

        outputs = Activation('softmax', dtype='float32')(conv_out)

        return outputs

    def conv_block(self, i, num_filters):
        j = SeparableConv2D(filters=num_filters,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            depth_multiplier=1,
                            depthwise_initializer="he_normal",
                            pointwise_initializer="he_normal",
                            bias_initializer="zeros",
                            depthwise_regularizer=regularizers.l2(config["weight_decay"]),
                            pointwise_regularizer=regularizers.l2(config["weight_decay"]))(i)

        if config["norm"] == "Batch Normalization":
            j = BatchNormalization()(j)
        elif config["norm"] == "Instance Normalization":
            j = InstanceNormalization()(j)
        elif config["norm"] == "Group Normalization":
            j = GroupNormalization(groups=config["groups"])(j)

        if config["activation"] == "PReLU":
            j = PReLU(shared_axes=[1, 2])(j)
        elif config["activation"] == "LeakyReLU":
            j = LeakyReLU(alpha=0.01)(j)
        else:
            j = Activation(config["activation"])(j)

        return j

    def encoder_block(self, i, num_filters):
        j = self.conv_block(i, num_filters)
        j = self.conv_block(j, num_filters)

        j = self.SA_module(j)

        return j

    def decoder_block(self, i, num_filters):
        j = self.conv_block(i, num_filters)
        j = self.conv_block(j, num_filters)

        j = self.SA_module(j)

        return j

    def downsampling_block(self, i, num_filters):
        if config["downsampling"] == "Max pooling":
            dn_smpl = MaxPooling2D(pool_size=(2, 2),
                                   strides=(2, 2))(i)
        elif config["downsampling"] == "Strided convolution":
            dn_smpl = Conv2D(filters=num_filters,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             padding="same",
                             kernel_initializer="he_normal",
                             bias_initializer="zeros",
                             kernel_regularizer=regularizers.l2(config["weight_decay"]))(i)

        return dn_smpl

    def upsampling_block(self, i, num_filters):
        if config["upsampling"] == "Nearest neighbor unpooling":
            up_smpl = UpSampling2D(size=(2, 2))(i)
        elif config["upsampling"] == "Bilinear upsampling":
            up_smpl = UpSampling2D(size=(2, 2), interpolation='bilinear')(i)
        elif config["upsampling"] == "Strided transpose convolution":
            up_smpl = Conv2DTranspose(filters=num_filters,
                                      kernel_size=(2, 2),
                                      strides=(2, 2),
                                      padding="same",
                                      kernel_initializer="he_normal",
                                      bias_initializer="zeros",
                                      kernel_regularizer=regularizers.l2(config["weight_decay"]))(i)

        return up_smpl

    def SA_module(self, i):
        channel_avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(i)
        attn_map = Conv2D(filters=1,
                          kernel_size=(3, 3),
                          strides=(1, 1),
                          padding="same",
                          activation="sigmoid",
                          kernel_initializer="he_normal",
                          bias_initializer="zeros")(channel_avg_pool)

        spatial_attn_features = multiply([attn_map, i])

        return spatial_attn_features

    def skip_connection(self, enc_block, up_smpl):
        if config["skip_connection"] == "Concatenate":
            skip = concatenate([enc_block, up_smpl], axis=-1)
        elif config["skip_connection"] == "Add":
            skip = add([enc_block, up_smpl])

        return skip
