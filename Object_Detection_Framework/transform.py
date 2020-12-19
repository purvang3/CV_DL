import numpy as np

DEFAULT_PROB = np.random


def transform_aabb(transform, aabb):
    """ Apply a transformation to an axis aligned bounding box.

    The result is a new AABB in the same coordinate system as the original AABB.
    The new AABB contains all corner points of the original AABB after applying the given transformation.

    Args
        transform: The transformation to apply.
        x1:        The minimum x value of the AABB.
        y1:        The minimum y value of the AABB.
        x2:        The maximum x value of the AABB.
        y2:        The maximum y value of the AABB.
    Returns
        The new AABB as tuple (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = aabb
    # Transform all 4 corners of the AABB.
    points = transform.dot([
        [x1, x2, x1, x2],
        [y1, y2, y2, y1],
        [1,  1,  1,  1 ],
    ])

    # Extract the min and max corners again.
    min_corner = points.min(axis=1)
    max_corner = points.max(axis=1)

    return [min_corner[0], min_corner[1], max_corner[0], max_corner[1]]

def _random_vector(min, max, rndn=DEFAULT_PROB):
    min = np.array(min)
    max = np.array(max)
    assert min.shape == max.shape
    assert len(min.shape) == 1
    return rndn.uniform(min, max)


def rotation(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_rotation(min, max, rndn):
    return rotation(rndn.uniform(min, max))


def translation(translation):
    return np.array([
        [1, 0, translation[0]],
        [0, 1, translation[1]],
        [0, 0, 1]
    ])


def random_translation(min, max, rndn):
    return translation(_random_vector(min, max, rndn))


def shear(angle):
    return np.array([
        [1, -np.sin(angle), 0],
        [0, np.cos(angle), 0],
        [0, 0, 1]
    ])


def random_shear(min, max, prng=DEFAULT_PROB):
    return shear(prng.uniform(min, max))


def scaling(factor):
    return np.array([
        [factor[0], 0, 0],
        [0, factor[1], 0],
        [0, 0, 1]
    ])


def random_scaling(min, max, prng=DEFAULT_PROB):
    return scaling(_random_vector(min, max, prng))


def random_flip(flip_x_chance, flip_y_chance, prng=DEFAULT_PROB):
    """ Construct a transformation randomly containing X/Y flips (or not).
    Args
        flip_x_chance: The chance that the result will contain a flip along the X axis.
        flip_y_chance: The chance that the result will contain a flip along the Y axis.
        prng:          The pseudo-random number generator to use.
    Returns
        a homogeneous 3 by 3 transformation matrix
    """
    flip_x = prng.uniform(0, 1) < flip_x_chance
    flip_y = prng.uniform(0, 1) < flip_y_chance
    # 1 - 2 * bool gives 1 for False and -1 for True.
    return scaling((1 - 2 * flip_x, 1 - 2 * flip_y))


def random_transform(min_rotation=0,
                     max_rotation=0,
                     min_translation=(0, 0),
                     max_translation=(0, 0),
                     min_shear=0,
                     max_shear=0,
                     min_scaling=(1, 1),
                     max_scaling=(1, 1),
                     flip_x_chance=0,
                     flip_y_chance=0,
                     rndn=DEFAULT_PROB):

    return np.linalg.multi_dot([
        random_rotation(min_rotation, max_rotation, rndn),
        random_translation(min_translation, max_translation, rndn),
        random_shear(min_shear, max_shear, rndn),
        random_scaling(min_scaling, max_scaling, rndn),
        random_flip(flip_x_chance, flip_y_chance, rndn)
    ])


def random_transform_generator(rndn=None, **kwargs):
    if rndn is None:
        rndn = np.random.RandomState()

    while True:
        yield random_transform(rndn=rndn, **kwargs)


def change_transform_origin(transform, center):
    """ Create a new transform representing the same transformation,
        only with the origin of the linear part changed.
    Args
        transform: the transformation matrix
        center: the new origin of the transformation
    Returns
        translate(center) * transform * translate(-center)
    """
    center = np.array(center)
    return np.linalg.multi_dot([translation(center), transform, translation(-center)])


if __name__ == "__main__":
    random_transform_generator()
    # random_transform()