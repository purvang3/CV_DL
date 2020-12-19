import keras
import initializers
import custom_layers as layers
from anchors import AnchorParameters
from feature_fusions import build_FPN, build_BiFPN, build_bottom_up_FPN, build_NoFPN, build_wBiFPN
import tensorflow as tf

MOMENTUM = 0.997
EPSILON = 1e-4

g_features = []


def assert_training_model(model):
    """ Assert that the model is a training model.
    """
    assert (all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(
            model.output_names)


def default_classification_model(
        num_classes,
        num_anchors,
        pyramid_feature_size=256,
        name='classification_submodel',
        width=256,
        depth=1,
        separable_conv=False,
        kernel_size=3
):
    options = {
        'kernel_size': kernel_size,
        'strides': 1,
        'padding': 'same',
    }

    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': keras.initializers.VarianceScaling(),
            'pointwise_initializer': keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        convs = [
            keras.layers.SeparableConv2D(filters=width, bias_initializer='zeros', name=f'{name}/class-{i}', **options)
            for i in range(depth)]

        head = keras.layers.SeparableConv2D(filters=num_classes * num_anchors,
                                            bias_initializer=initializers.PriorProbability(probability=0.01),
                                            name=f'{name}/class-predict', **options)
    else:
        kernel_initializer = {
            'kernel_initializer': keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }
        options.update(kernel_initializer)
        convs = [keras.layers.Conv2D(filters=width, bias_initializer='zeros', name=f'{name}/class-{i}', **options)
                 for i in range(depth)]
        head = keras.layers.Conv2D(filters=num_classes * num_anchors,
                                   bias_initializer=initializers.PriorProbability(probability=0.01),
                                   name='class-predict', **options)
    bns = [
        [keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/class-{i}-bn-{j}') for j
         in range(3, 8)] for i in range(depth)]

    relu = keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')
    reshape = keras.layers.Reshape((-1, num_classes))
    activation = keras.layers.Activation('sigmoid')
    level = 0

    outputs = inputs
    for i in range(depth):
        outputs = convs[i](outputs)
        outputs = bns[i][level](outputs)
        outputs = relu(outputs)
        level += 1
    outputs = head(outputs)
    outputs = reshape(outputs)
    outputs = activation(outputs)
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_values, num_anchors,
                             pyramid_feature_size=256,
                             width=256,
                             depth=1,
                             name='regression_submodel',
                             separable_conv=False,
                             kernel_size=3):
    if keras.backend.image_data_format() == 'channels_first':
        inputs = keras.layers.Input(shape=(pyramid_feature_size, None, None))
    else:
        inputs = keras.layers.Input(shape=(None, None, pyramid_feature_size))

    options = {
        'kernel_size': kernel_size,
        'strides': 1,
        'padding': 'same',
        'bias_initializer': 'zeros',
    }
    if separable_conv:
        kernel_initializer = {
            'depthwise_initializer': keras.initializers.VarianceScaling(),
            'pointwise_initializer': keras.initializers.VarianceScaling(),
        }
        options.update(kernel_initializer)
        convs = [keras.layers.SeparableConv2D(filters=width, name=f'{name}/box-{i}', **options) for i in
                 range(depth)]
        head = keras.layers.SeparableConv2D(filters=num_anchors * num_values,
                                            name=f'{name}/box-predict', **options)
    else:
        kernel_initializer = {
            'kernel_initializer': keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
        }
        options.update(kernel_initializer)
        convs = [keras.layers.Conv2D(filters=width, name=f'{name}/box-{i}', **options) for i in
                 range(depth)]
        head = keras.layers.Conv2D(filters=num_anchors * num_values, name=f'{name}/box-predict',
                                   **options)
    bns = [
        [keras.layers.BatchNormalization(momentum=MOMENTUM, epsilon=EPSILON, name=f'{name}/box-{i}-bn-{j}') for
         j in range(3, 8)]
        for i in range(depth)]
    relu = keras.layers.Lambda(lambda x: tf.nn.swish(x))
    reshape = keras.layers.Reshape((-1, num_values))
    level = 0

    outputs = inputs
    for i in range(depth):
        outputs = convs[i](outputs)
        outputs = bns[i][level](outputs)
        outputs = relu(outputs)
    outputs = head(outputs)
    outputs = reshape(outputs)
    level += 1
    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_submodels(num_classes, num_anchors, **kwargs):
    return [
        ('regression', default_regression_model(4, num_anchors, **kwargs)),
        ('classification', default_classification_model(num_classes, num_anchors, **kwargs))
    ]


def __build_model_pyramid(name, model, features):
    """ Applies a single submodel to each FPN level.

    Args
        name     : Name of the submodel.
        model    : The submodel to evaluate.
        features : The FPN features.

    Returns
        A tensor containing the response from the submodel on the FPN features.
    """
    return keras.layers.Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    """ Applies all submodels to each FPN level.

    Args
        models   : List of submodels to run on each pyramid level (by default only regression, classifcation).
        features : The FPN features.

    Returns
        A list of tensors, one for each submodel.
    """
    return [__build_model_pyramid(n, m, features) for n, m in models]


def retinanet(
        inputs,
        backbone_layers,
        num_classes,
        num_anchors=None,
        submodels=None,
        fpn_method="fpn",
        name=None,
        need_fpn_functionality=None,
        freeze_bn=None,
        fpn_depth=1,
        **kwargs
):
    global g_features
    features_size = kwargs.get("width")
    pyramid_feature_size = {"pyramid_feature_size": features_size}
    kwargs.update(pyramid_feature_size)
    noise_type = kwargs.get("noise_type")
    kwargs.pop("noise_type")

    if num_anchors is None:
        num_anchors = AnchorParameters.default.num_anchors()

    if submodels is None:
        submodels = default_submodels(num_classes, num_anchors, **kwargs)

    _, _, C3, C4, C5 = backbone_layers
    if not need_fpn_functionality or fpn_method == "nofpn":
        features = build_NoFPN(C3, C4, C5, feature_size=features_size)

    elif fpn_method == "fpn":
        features = build_FPN(C3, C4, C5, feature_size=features_size)

    elif fpn_method == "bottom_up_fpn":
        features = build_bottom_up_FPN(backbone_layers, feature_size=features_size)

    elif fpn_method == "top_down_fpn":
        raise ValueError("NOT IMPLEMENTED")

    elif fpn_method == "bifpn":
        features = backbone_layers
        for i in range(fpn_depth):
            features = build_BiFPN(features=features, num_channels=features_size, id=i, freeze_bn=freeze_bn)

    elif fpn_method == "wbifpn":
        # TODO logic for index 1
        features = backbone_layers
        for i in range(fpn_depth):
            features = build_wBiFPN(features=features, num_channels=features_size, id=i, freeze_bn=freeze_bn)

    g_features = [f for f in features]

    if noise_type:
        if noise_type not in NOISE_MAP_DICT.keys():
            raise ValueError(f"{noise_type} is not valid method. supported noise_types: {NOISE_MAP_DICT.keys()}")

        features = noisy_features(features, features_size, noise_type)

    pyramids = __build_pyramid(submodels, features)

    return keras.models.Model(inputs=inputs, outputs=pyramids, name=name)


NOISE_MAP_DICT = {
    "add": keras.layers.Add(),
    "average": keras.layers.Average(),
    "gaussian": keras.layers.GaussianNoise(stddev=1.0),
    "mul": keras.layers.Multiply()

}


def noisy_features(features, features_size, noise_type):
    t = []
    for layer in features:
        if noise_type == "gaussian":
            layer = NOISE_MAP_DICT[noise_type](layer)
            t.append(layer)
            continue
        l_ = keras.layers.Conv2D(filters=features_size, kernel_size=1)(layer)
        l_ = NOISE_MAP_DICT[noise_type]([l_, layer])
        l = keras.layers.Activation('sigmoid')(l_)
        t.append(l)
    features = t
    return features


def __build_anchors(anchor_parameters, features):
    anchors = [
        layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f) for i, f in enumerate(features)
    ]

    return keras.layers.Concatenate(axis=1, name='anchors')(anchors)


def retinanet_bbox(
        model=None,
        nms=True,
        class_specific_filter=True,
        name='retinanet-bbox',
        anchor_params=None,
        nms_threshold=0.5,
        score_threshold=0.05,
        max_detections=300,
        parallel_iterations=32,
        **kwargs
):
    global g_features
    # if no anchor parameters are passed, use default values
    if anchor_params is None:
        anchor_params = AnchorParameters.default

    # create RetinaNet model
    if model is None:
        raise ValueError("Model is not created yet.")

    # compute the anchors
    if not g_features:
        g_features = [model.get_layer(p_name).output for p_name in ['P3', 'P4', 'P5', 'P6', 'P7']]

    anchors = __build_anchors(anchor_params, g_features)

    # we expect the anchors, regression and classification values as first output
    regression = model.outputs[0]
    classification = model.outputs[1]

    # "other" can be any additional output from custom submodels, by default this will be []
    other = model.outputs[2:]

    # apply predicted regression to anchors
    boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
    boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    # filter detections (apply NMS / score threshold / select top-k)
    detections = layers.FilterDetections(
        nms=nms,
        class_specific_filter=class_specific_filter,
        name='filtered_detections',
        nms_threshold=nms_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
        parallel_iterations=parallel_iterations
    )([boxes, classification] + other)

    return keras.models.Model(inputs=model.inputs, outputs=detections, name=name)
