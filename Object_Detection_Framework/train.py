"""
Purvang Lapsiwala
"""

import os
import warnings
from argparse import Namespace

import keras
import keras.preprocessing.image
import losses
import numpy as np
import tensorflow as tf
import yaml
from anchors import make_shapes_callback, AnchorParameters
from backbones import ResNetBackbone, VGGBackbone, InceptionBackbone, EfficientNetBackbone, XceptionBackbone
from config import read_config_file, parse_anchor_parameters
from generator import CSVGenerator
from gpu import setup_gpu
from image import random_visual_effect_generator
from flex_utils import StepDecay, PolynomialDecay, FeatureVisualizer, RedirectModel
from retinanet import retinanet_bbox
from transform import random_transform_generator


def freeze(model):
    for layer in model.layers:
        layer.trainable = False
    return model


def backbone(backbone_name):
    return ResNetBackbone(backbone_name)


def load_model(filepath, backbone_name='resnet50'):
    import keras.models
    return keras.models.load_model(filepath, custom_objects=ResNetBackbone(backbone_name).custom_objects)


def model_with_weights(model, weights, skip_mismatch):
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def create_models(args, backbone_retinanet, num_classes, weights, multi_gpu=0,
                  freeze_backbone=False, lr=1e-5, config=None, backbone_name=None):
    """ Creates three models (model, training_model, prediction_model).

    Args
        backbone_retinanet : A function to call to create a retinanet model with a given backbone.
        num_classes        : The number of classes to train.
        weights            : The weights to load into the model.
        multi_gpu          : The number of GPUs to use for training.
        freeze_backbone    : If True, disables learning for the backbone.
        config             : Config parameters, None indicates the default configuration.

    Returns
        model            : The base model. This is also the model that is saved in snapshots.
        training_model   : The training model. If multi_gpu=0, this is identical to model.
        prediction_model : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
    """

    if backbone_name is None:
        raise

    modifier = freeze if freeze_backbone else None

    # load anchor parameters, or pass None (so that defaults will be used)
    anchor_params = None
    num_anchors = None
    # if config and 'anchor_parameters' in config:
    #     anchor_params = parse_anchor_parameters(config)
    #     num_anchors   = anchor_params.num_anchors()

    # if multi_gpu > 1:
    #     from keras.utils import multi_gpu_model
    #     with tf.device('/cpu:0'):
    #         model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
    #                                    weights=weights, skip_mismatch=True)
    #     training_model = multi_gpu_model(model, gpus=multi_gpu)
    # else:
    #     model = model_with_weights(backbone_retinanet(num_classes, num_anchors=num_anchors, modifier=modifier),
    #                                weights=weights, skip_mismatch=True)
    #     training_model = model

    kwargs = {
        "fpn_method": args.fpn_method,
        "width": args.width,
        "depth": args.depth,
        "num_anchors": num_anchors,
        "modifier": modifier,
        "need_fpn_functionality": args.need_fpn_functionality,
        "separable_conv": args.separable_conv,
        "freeze_bn": args.freeze_bn,
        "name": args.backbone,
        "fpn_depth": args.fpn_depth,
        "kernel_size": args.kernel_size,
        "noise_type": args.noise_type
    }

    if multi_gpu <= 1:
        # if "resnet" in args.backbone:
        #     model = model_with_weights(backbone_retinanet(num_classes, **kwargs), weights=weights, skip_mismatch=True)
        # else:
        #     model = backbone_retinanet(num_classes, **kwargs)
        model = backbone_retinanet(num_classes, **kwargs)
        training_model = model
    else:
        from keras.utils import multi_gpu_model
        with tf.device('/cpu:0'):
            model = backbone_retinanet(num_classes, **kwargs)
            training_model = multi_gpu_model(model, gpus=multi_gpu)

    # make prediction model
    prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)

    # compile model

    training_model.compile(
        loss={
            'regression': getattr(losses, args.regression_loss)(),
            'classification': getattr(losses, args.classification_loss)()
        },
        optimizer=keras.optimizers.adam(lr=lr, clipnorm=0.001)
    )

    return model, training_model, prediction_model


def create_generator(args, preprocess_image):
    common_args = {
        'batch_size': args.batch_size,
        'config': args.config,
        'image_min_side': args.image_min_side,
        'image_max_side': args.image_max_side,
        'no_resize': args.no_resize,
        'preprocess_image': preprocess_image,
        'anchor_generation_method': args.anchor_generation_method,
        'is_square_resize': args.is_square_resize
    }

    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05)
        )
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        **common_args
    )

    if args.val_annotations:
        validation_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            shuffle_groups=False,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def create_callbacks(model, training_model, prediction_model, validation_generator, args):
    """ Creates the callbacks to use during training.

    Args
        model: The base model.
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        os.makedirs(args.tensorboard_dir, exist_ok=True)

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )

        if args.tensorboard_dir:
            callbacks.append(tensorboard_callback)

    # if args.evaluation and validation_generator:
    #     if args.dataset_type == 'coco':
    #         from ..callbacks.coco import CocoEval
    #
    #         # use prediction model for evaluation
    #         evaluation = CocoEval(validation_generator, tensorboard=tensorboard_callback)
    #     else:
    #         evaluation = Evaluate(validation_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
    #     evaluation = RedirectModel(evaluation, prediction_model)
    #     callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(args.snapshot_path, exist_ok=True)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                '{backbone}_{dataset_type}_{{epoch:02d}}.h5'.format(backbone=args.backbone,
                                                                    dataset_type=args.dataset_type)
            ),
            verbose=1,
            save_best_only=True,
            monitor="loss",
            mode='min'
        )
        checkpoint = RedirectModel(checkpoint, model)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=args.reduce_lr_factor,
        patience=args.reduce_lr_patience,
        verbose=1,
        mode='auto',
        min_delta=0.0001,
        cooldown=0,
        min_lr=0
    ))

    try:
        if args.lr_schedule_method == "step":
            callbacks.append(keras.callbacks.LearningRateScheduler(
                schedule=StepDecay(init_lr=args.init_lr, factor=args.factor, dropEvery=args.dropEveryEpochs),
                verbose=0
            ))

        elif args.lr_schedule_method == "linear":
            callbacks.append(keras.callbacks.LearningRateScheduler(
                schedule=PolynomialDecay(maxEpochs=args.maxEpochs, initAlpha=args.initAlpha, power=args.power),
                verbose=0
            ))
    except Exception:
        warnings.warn("No Learning rate scheduler has been set.")

    callbacks.append(keras.callbacks.EarlyStopping(
        monitor='classification_loss',
        patience=5,
        mode='min',
        min_delta=0.01
    ))

    callbacks.append(FeatureVisualizer(model=model,
                                       image_path="1.jpeg",
                                       conv_features=args.conv_features,
                                       bn_features=args.bn_features,
                                       activation_features=args.activation_features,
                                       save_weights=args.save_weights,
                                       feature_freq_in_epochs=args.feature_freq_in_epochs,
                                       feature_saving_dir_path=args.feature_saving_dir_path,
                                       num_of_channels=args.num_of_channels
                                       ))

    return callbacks


def parse_config():
    supported_fpns = ["fpn", "bottom_up_fpn", "top_down_fpn", "bifpn", "wbifpn"]

    config = yaml.safe_load(open("./config.yaml"))
    ns = Namespace(**config)

    if ns.multi_gpu > 1 and ns.batch_size < ns.multi_gpu:
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(ns.batch_size,
                                                                                             ns.multi_gpu))

    if ns.multi_gpu > 1 and ns.snapshot:
        raise ValueError(
            "Multi GPU training ({}) and resuming from snapshots ({}) is not supported.".format(ns.multi_gpu,
                                                                                                ns.snapshot))

    if ns.multi_gpu > 1 and not ns.multi_gpu_force:
        raise ValueError(
            "Multi-GPU support is experimental, use at own risk! Run with --multi-gpu-force if you wish to continue.")

    if ns.fpn_method not in supported_fpns:
        raise ValueError(f"Backbone '{ns.fpn_method}' currently not supported. Supported backbones: {supported_fpns}")

    if ns.no_resize and ns.is_square_resize:
        raise ValueError(f"can not be used no_resize and is_square_resize together.")

    if not ns.need_fpn_functionality and ns.fpn_method:
        warnings.warn("**** with need_fpn_functionality set to False, no fpn method will be used.")

    if ns.anchor_generation_method == "multiple":
        if not (ns.sizes or ns.strides or ns.ratios or ns.scales):
            raise ValueError(f"multiple anchor generation needs all from [sizes, strides, ratios, scales]")

    if 'resnet' not in ns.backbone:
        warnings.warn(
            'Using experimental backbone {}. Only resnet50 has been properly tested.'.format(ns.backbone))

    return ns


def main():
    args = parse_config()

    supported_backbones = ['resnet', 'vgg', 'inception', 'xception', 'efficientnet']

    # create object that stores backbone information
    if "resnet" in args.backbone:
        backbone = ResNetBackbone(args.backbone)
    elif "vgg" in args.backbone:
        backbone = VGGBackbone(args.backbone)
    elif "inception" in args.backbone:
        backbone = InceptionBackbone(args.backbone)
    elif "xception" in args.backbone:
        backbone = XceptionBackbone(args.backbone)
    elif "efficientnet" in args.backbone:
        backbone = EfficientNetBackbone(args.backbone)
    else:
        raise ValueError(
            f"Backbone '{args.backbone}' currently not supported. Supported backbones: {supported_backbones}")

    # # make sure keras and tensorflow are the minimum required version
    # check_keras_version()
    # check_tf_version()

    # optionally choose specific GPU
    if args.gpu is not None:
        setup_gpu(args.gpu)

    if not "config" in dir(args):
        args.config = None

    # # optionally load config parameters
    elif args.config and args.anchor_generation_method == "multiple":
        args.config = read_config_file(args.config)

    # create the generators
    train_generator, validation_generator = create_generator(args, backbone.preprocess_image)

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        model = load_model(args.snapshot, backbone_name=args.backbone)
        training_model = model
        # anchor_params = None

        if args.config and 'anchor_parameters' in args.config:
            anchor_params = parse_anchor_parameters(args.config)
        else:
            anchor_params = get_anchor_config(args)
        prediction_model = retinanet_bbox(model=model, anchor_params=anchor_params)
    else:
        weights = args.weights
        # default to imagenet if nothing else is specified
        if weights is None and args.imagenet_weights:
            weights = ResNetBackbone.download_imagenet()

        print('Creating model, this may take a second...')
        model, training_model, prediction_model = create_models(
            args=args,
            backbone_retinanet=backbone.retinanet,
            num_classes=train_generator.num_classes(),
            weights=weights,
            multi_gpu=args.multi_gpu,
            freeze_backbone=args.freeze_backbone,
            lr=args.lr,
            config=args.config,
            backbone_name=args.backbone
        )

        # layer_dict = dict([(layer.name, layer) for layer in model.layers])

    # print model summary
    # print(model.summary())

    # this lets the generator compute backbone layer shapes using the actual backbone model
    if 'vgg' in args.backbone or 'densenet' in args.backbone:
        train_generator.compute_shapes = make_shapes_callback(model)
        if validation_generator:
            validation_generator.compute_shapes = train_generator.compute_shapes

    # create the callbacks
    callbacks = create_callbacks(
        model,
        training_model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None

    # start training
    return training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator,
        initial_epoch=args.initial_epoch
    )


def get_anchor_config(args):
    if args.anchor_generation_method == "multiple":
        if args.sizes and args.strides and args.ratios and args.scales:
            anchor_params = AnchorParameters(sizes=np.array(args.sizes, dtype=keras.backend.floatx()),
                                             strides=np.array(args.strides, dtype=keras.backend.floatx()),
                                             ratios=np.array(args.ratios, dtype=keras.backend.floatx()),
                                             scales=np.array(args.scales, dtype=keras.backend.floatx()))

            AnchorParameters.default.sizes, AnchorParameters.default.strides, AnchorParameters.default.ratios, \
            AnchorParameters.default.scales = anchor_params.sizes, anchor_params.strides, anchor_params.ratios, \
                                              anchor_params.scales
            return anchor_params
    return


if __name__ == '__main__':
    main()
