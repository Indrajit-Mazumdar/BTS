from tensorflow import pad
from tensorflow.keras.layers import Layer


class ConstantPadding1D(Layer):

    def __init__(self, padding=(1, 1), constant=0, **kwargs):
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_left, padding_right],
                    [0, 0]],
                   mode='CONSTANT',
                   constant_values=self.constant)


class ConstantPadding2D(Layer):

    def __init__(self, padding=(1, 1), constant=0, **kwargs):
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_height, padding_height],
                    [padding_width, padding_width],
                    [0, 0]],
                   mode='CONSTANT',
                   constant_values=self.constant)


class ConstantPadding3D(Layer):

    def __init__(self, padding=(1, 1), constant=0, **kwargs):
        self.padding = tuple(padding)
        self.constant = constant
        super(ConstantPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3] + 2 * self.padding[2],
                input_shape[4])

    def call(self, input_tensor, mask=None):
        padding_depth, padding_width, padding_height = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_depth, padding_depth],
                    [padding_height, padding_height],
                    [padding_width, padding_width],
                    [0, 0]],
                   mode='CONSTANT',
                   constant_values=self.constant)


class ReflectionPadding1D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_left, padding_right],
                    [0, 0]],
                   mode='REFLECT')


class ReflectionPadding2D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_height, padding_height],
                    [padding_width, padding_width],
                    [0, 0]],
                   mode='REFLECT')


class ReflectionPadding3D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3] + 2 * self.padding[2],
                input_shape[4])

    def call(self, input_tensor, mask=None):
        padding_depth, padding_width, padding_height = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_depth, padding_depth],
                    [padding_height, padding_height],
                    [padding_width, padding_width],
                    [0, 0]],
                   mode='REFLECT')


class ReplicationPadding1D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding1D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[1] + self.padding[0] + self.padding[1]

    def call(self, input_tensor, mask=None):
        padding_left, padding_right = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_left, padding_right],
                    [0, 0]],
                   mode='SYMMETRIC')


class ReplicationPadding2D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_height, padding_height],
                    [padding_width, padding_width],
                    [0, 0]],
                   mode='SYMMETRIC')


class ReplicationPadding3D(Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReplicationPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3] + 2 * self.padding[2],
                input_shape[4])

    def call(self, input_tensor, mask=None):
        padding_depth, padding_width, padding_height = self.padding
        return pad(input_tensor,
                   [[0, 0],
                    [padding_depth, padding_depth],
                    [padding_height, padding_height],
                    [padding_width, padding_width],
                    [0, 0]],
                   mode='SYMMETRIC')
