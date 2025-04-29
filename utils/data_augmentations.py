import numpy as np
from scipy import ndimage


def rotate_3D(image, mask):
    angles = np.array([90, 180, 270])
    rotation_angle = np.random.choice(a=angles, size=None, replace=False)

    axes = (0, 1, 2)
    axes = np.random.choice(a=axes, size=2, replace=False)
    axes.sort()

    image = ndimage.rotate(image,
                           angle=rotation_angle,
                           axes=axes,
                           reshape=False,
                           mode='constant',
                           cval=0.0)
    mask = ndimage.rotate(mask,
                          angle=rotation_angle,
                          axes=axes,
                          reshape=False,
                          mode='constant',
                          cval=0.0)
    return image, mask


def rotate_2D(image, mask):
    angles = np.array([90, 180, 270])
    rotation_angle = np.random.choice(a=angles, size=None, replace=False)

    axes = (0, 1)

    image = ndimage.rotate(image,
                           angle=rotation_angle,
                           axes=axes,
                           reshape=False,
                           mode='constant',
                           cval=0.0)
    mask = ndimage.rotate(mask,
                          angle=rotation_angle,
                          axes=axes,
                          reshape=False,
                          mode='constant',
                          cval=0.0)
    return image, mask


def horizontal_flip_3D(image, mask):
    image[:, :, :, :] = image[:, :, ::-1, :]
    mask[:, :, :, :] = mask[:, :, ::-1, :]
    return image, mask


def horizontal_flip_2D(image, mask):
    image[:, :, :] = image[:, ::-1, :]
    mask[:, :, :] = mask[:, ::-1, :]
    return image, mask


def vertical_flip_3D(image, mask):
    image[:, :, :, :] = image[:, ::-1, :, :]
    mask[:, :, :, :] = mask[:, ::-1, :, :]
    return image, mask


def vertical_flip_2D(image, mask):
    image[:, :, :] = image[::-1, :, :]
    mask[:, :, :] = mask[::-1, :, :]
    return image, mask


def depthwise_flip_3D(image, mask):
    image[:, :, :, :] = image[::-1, :, :, :]
    mask[:, :, :, :] = mask[::-1, :, :, :]
    return image, mask
