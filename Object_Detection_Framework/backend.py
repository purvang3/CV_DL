import tensorflow
import keras
import keras.callbacks


# =============================================================================================

def convert_to_tensor(*args, **kwargs):
    return tensorflow.convert_to_tensor(*args, **kwargs)


def ones(*args, **kwargs):
    return tensorflow.ones(*args, **kwargs)


def transpose(*args, **kwargs):
    return tensorflow.transpose(*args, **kwargs)


def map_fn(*args, **kwargs):
    return tensorflow.map_fn(*args, **kwargs)


def pad(*args, **kwargs):
    return tensorflow.pad(*args, **kwargs)


def top_k(*args, **kwargs):
    return tensorflow.nn.top_k(*args, **kwargs)


def clip_by_value(*args, **kwargs):
    return tensorflow.clip_by_value(*args, **kwargs)


def resize_images(images, size, method='bilinear', align_corners=False):
    methods = {
        'bilinear': tensorflow.image.ResizeMethod.BILINEAR,
        'nearest': tensorflow.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tensorflow.image.ResizeMethod.BICUBIC,
        'area': tensorflow.image.ResizeMethod.AREA,
    }
    return tensorflow.compat.v1.image.resize_images(images, size, methods[method], align_corners)


def non_max_suppression(*args, **kwargs):
    return tensorflow.image.non_max_suppression(*args, **kwargs)


def range(*args, **kwargs):
    return tensorflow.range(*args, **kwargs)


def scatter_nd(*args, **kwargs):
    return tensorflow.scatter_nd(*args, **kwargs)


def gather_nd(*args, **kwargs):
    return tensorflow.gather_nd(*args, **kwargs)


def meshgrid(*args, **kwargs):
    return tensorflow.meshgrid(*args, **kwargs)


def where(*args, **kwargs):
    return tensorflow.where(*args, **kwargs)


def unstack(*args, **kwargs):
    return tensorflow.unstack(*args, **kwargs)


# =============================================================================================

def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                        dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5,
                                                                                                        dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(
        keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors

