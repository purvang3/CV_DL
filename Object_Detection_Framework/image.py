import numpy as np
import cv2
from PIL import Image
from transform import change_transform_origin


def read_image_bgr(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def _check_range(val_range, min_val=None, max_val=None):
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


def _clip(image):
    return np.clip(image, 0, 255).astype(np.uint8)


def adjust_contrast(image, factor):
    mean = image.mean(axis=0).mean(axis=0)
    return _clip((image - mean) * factor + mean)


def adjust_brightness(image, delta):
    return _clip(image + delta * 255)


def adjust_hue(image, delta):
    image[..., 0] = np.mod(image[..., 0] + delta * 180, 180)
    return image


def adjust_saturation(image, factor):
    image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
    return image


def _uniform(val_range):
    return np.random.uniform(val_range[0], val_range[1])


class VisualEffect:

    def __init__(
            self,
            contrast_factor,
            brightness_delta,
            hue_delta,
            saturation_factor,
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        if self.contrast_factor:
            image = adjust_contrast(image, self.contrast_factor)
        if self.brightness_delta:
            image = adjust_brightness(image, self.brightness_delta)

        if self.hue_delta or self.saturation_factor:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            if self.hue_delta:
                image = adjust_hue(image, self.hue_delta)
            if self.saturation_factor:
                image = adjust_saturation(image, self.saturation_factor)

            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image


def random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)
):
    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, -1, 1)
    _check_range(saturation_range, 0)

    def _generate():
        while True:
            yield VisualEffect(
                contrast_factor=_uniform(contrast_range),
                brightness_delta=_uniform(brightness_range),
                hue_delta=_uniform(hue_range),
                saturation_factor=_uniform(saturation_range),
            )

    return _generate()


def preprocess_image(x, mode='tf'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x -= [103.939, 116.779, 123.68]

    return x


class TransformParameters:

    def __init__(
            self,
            fill_mode='nearest',
            interpolation='linear',
            cval=0,
            relative_translation=True,
    ):
        self.fill_mode = fill_mode
        self.cval = cval
        self.interpolation = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params):
    """
    Apply a transformation to an image.

    The origin of transformation is at the top left corner of the image.

    The matrix is interpreted such that a point (x, y) on the original image is moved to transform * (x, y) in the generated image.
    Mathematically speaking, that means that the matrix is a transformation from the transformed image space to the original image space.

    Args
      matrix: A homogeneous 3 by 3 matrix holding representing the transformation to apply.
      image:  The image to transform.
      params: The transform parameters (see TransformParameters)
    """
    output = cv2.warpAffine(
        image,
        matrix[:2, :],
        dsize=(image.shape[1], image.shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )
    return output


def compute_square_resize_scale(image_shape, min_side=800, max_side=1333):
    (rows, cols, _) = image_shape  # 720, 1280

    smallest_side = min(rows, cols)  # 720

    scale_min = min_side / smallest_side  # 300/720

    largest_side = max(rows, cols)  # 1280
    # if largest_side * scale > max_side:  # 1280 * (300/720) > 1280
    #     scale = max_side / largest_side  # 300/1280

    scale_max = max_side / largest_side  # 300/1280
    return scale_min, scale_max


def compute_resize_scale(image_shape, min_side=800, max_side=1333):
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=800, max_side=1333, is_square_resize=False):
    """ Resize an image such that the size is constrained to min_side and max_side.

        Args
            min_side: The image's min side will be equal to min_side after resizing.
            max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

        Returns
            A resized image.
        """

    if is_square_resize:
        # compute scale to resize the image
        scale_min, scale_max = compute_square_resize_scale(img.shape, min_side=min_side, max_side=max_side)

        # resize the image with the computed scale
        img = cv2.resize(img, None, fx=scale_max, fy=scale_min)

        return img, scale_min, scale_max

    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale, scale


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image.shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result
