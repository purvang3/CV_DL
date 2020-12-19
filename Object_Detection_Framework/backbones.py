import keras
import keras_resnet
import keras_resnet.models
import retinanet
from efficientDet_backbone import EfficientNetB0_, EfficientNetB1_, EfficientNetB2_, \
    EfficientNetB3_, EfficientNetB4_, EfficientNetB5_, \
    EfficientNetB6_, EfficientNetB7_
from image import preprocess_image
from keras.utils import get_file


class BackBone(object):
    def __init__(self, backbone):
        import custom_layers as layers
        import losses
        import initializers
        self.custom_objects = {
            'PriorProbability': initializers.PriorProbability,

            'UpsampleLike': layers.UpsampleLike,
            'RegressBoxes': layers.RegressBoxes,
            'FilterDetections': layers.FilterDetections,
            'Anchors': layers.Anchors,
            'ClipBoxes': layers.ClipBoxes,
            # 'BatchNormalization': layers.BatchNormalization,

            '_smooth_l1': losses.smooth_l1(),
            '_focal': losses.focal(),
        }

        self.backbone = backbone
        self.validate()

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    # def download_imagenet(self):
    #     """ Downloads ImageNet weights and returns path to weights file.
    #     """
    #     raise NotImplementedError('download_imagenet method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')


class ResNetBackbone(BackBone):
    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone=backbone)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)

        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet101', 'resnet152']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return self.resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    @staticmethod
    def model_with_weights(model, weights, skip_mismatch):
        if weights is not None:
            model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
        return model

    def resnet_retinanet(self, num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
        """ Constructs a retinanet model using a resnet backbone.

        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

        Returns
            RetinaNet model with a ResNet backbone.
        """
        # choose default input
        if inputs is None:
            if keras.backend.image_data_format() == 'channels_first':
                inputs = keras.layers.Input(shape=(3, None, None))
            else:
                inputs = keras.layers.Input(shape=(None, None, 3))

        # create the resnet backbone
        if backbone == 'resnet50':
            model = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
        elif backbone == 'resnet101':
            model = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
        elif backbone == 'resnet152':
            model = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
        else:
            raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

        # invoke modifier if given
        if modifier:
            model = modifier(model)

        layer_outputs = model.outputs[1:]
        backbone_layers = [None, None, *layer_outputs]
        # create the full model
        # return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=backbone_layers, **kwargs)
        return self.model_with_weights(retinanet.retinanet(inputs=inputs, num_classes=num_classes,
                                                           backbone_layers=backbone_layers, **kwargs),
                                       weights=kwargs.get("weights"), skip_mismatch=True)


class VGGBackbone(BackBone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(VGGBackbone, self).__init__(backbone=backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return self.vgg_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg16':
            resource = keras.applications.vgg16.vgg16.WEIGHTS_PATH_NO_TOP
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif self.backbone == 'vgg19':
            resource = keras.applications.vgg19.vgg19.WEIGHTS_PATH_NO_TOP
            checksum = '253f8cb515780f3b799900260a226db6'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')

    @staticmethod
    def vgg_retinanet(num_classes, backbone='vgg16', inputs=None, modifier=None, *args, **kwargs):
        """ Constructs a retinanet model using a vgg backbone.

        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

        Returns
            RetinaNet model with a VGG backbone.
        """
        # choose default input
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 3))

        # create the vgg backbone
        if backbone == 'vgg16':
            vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False, weights='imagenet')
        elif backbone == 'vgg19':
            vgg = keras.applications.VGG19(input_tensor=inputs, include_top=False, weights='imagenet')
        else:
            raise ValueError("Backbone '{}' not recognized.".format(backbone))

        if modifier:
            vgg = modifier(vgg)

        layer_names = ["block3_pool", "block4_pool", "block5_pool"]
        layer_outputs = [vgg.get_layer(name).output for name in layer_names]
        layer_outputs = [None, None, *layer_outputs]
        return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)


class InceptionBackbone(BackBone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(InceptionBackbone, self).__init__(backbone=backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return self.inception_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inceptionV3']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')

    @staticmethod
    def inception_retinanet(num_classes, backbone='inceptionV3', inputs=None, modifier=None, **kwargs):
        """ Constructs a retinanet model using a vgg backbone.

        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('inceptionV3'))
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

        Returns
            RetinaNet model with a VGG backbone.
        """
        # choose default input
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 3))

        if backbone == 'inceptionV3':

            inception_model = keras.applications.InceptionV3(input_tensor=inputs,
                                                             include_top=False,
                                                             weights='imagenet'
                                                             )

        else:
            raise ValueError("Backbone '{}' not recognized.".format(backbone))

        if modifier:
            inception_model = modifier(inception_model)

        # layer_names = [["mixed0", "mixed1", "mixed2"],[ "mixed3", "mixed4", ("mixed5", "mixed6"),"mixed7"],
        # ["mixed8", ("mixed9_0", "mixed9_1")]
        layer_names = ["mixed2", "mixed6", "mixed7"]
        layer_outputs = [inception_model.get_layer(name).output for name in layer_names]
        layer_outputs = [None, None, *layer_outputs]

        return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)


class EfficientNetBackbone(BackBone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(EfficientNetBackbone, self).__init__(backbone=backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return self.efficientnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                             'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')

    @staticmethod
    def efficientnet_retinanet(num_classes, backbone='efficientnet-b0', inputs=None, modifier=None, **kwargs):
        """ Constructs a retinanet model using a vgg backbone.

        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
            'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7')).
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

        Returns
            RetinaNet model with a VGG backbone.
        """
        # choose default input
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 3))

        # create the vgg backbone
        if backbone == 'efficientnet-b0':
            efficientnet = EfficientNetB0_(input_tensor=inputs)

        elif backbone == 'efficientnet-b1':
            efficientnet = EfficientNetB1_(input_tensor=inputs)

        elif backbone == 'efficientnet-b2':
            efficientnet = EfficientNetB2_(input_tensor=inputs)

        elif backbone == 'efficientnet-b3':
            efficientnet = EfficientNetB3_(input_tensor=inputs)

        elif backbone == 'efficientnet-b4':
            efficientnet = EfficientNetB4_(input_tensor=inputs)

        elif backbone == 'efficientnet-b5':
            efficientnet = EfficientNetB5_(input_tensor=inputs)

        elif backbone == 'efficientnet-b6':
            efficientnet = EfficientNetB6_(input_tensor=inputs)

        elif backbone == 'efficientnet-b7':
            efficientnet = EfficientNetB7_(input_tensor=inputs)

        else:
            raise ValueError("Backbone '{}' not recognized.".format(backbone))

        if modifier:
            efficientnet = modifier(efficientnet)

        return retinanet.retinanet(inputs=inputs,
                                   num_classes=num_classes,
                                   backbone_layers=efficientnet,
                                   **kwargs)


class XceptionBackbone(BackBone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(XceptionBackbone, self).__init__(backbone=backbone)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return self.xception_retinanet(*args, backbone=self.backbone, **kwargs)

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['xception']

        if self.backbone not in allowed_backbones:
            raise ValueError(
                'Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')

    @staticmethod
    def xception_retinanet(num_classes, backbone='xception', inputs=None, modifier=None, **kwargs):
        """ Constructs a retinanet model using a vgg backbone.

        Args
            num_classes: Number of classes to predict.
            backbone: Which backbone to use (one of ('xception')).
            inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
            modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

        Returns
            RetinaNet model with a VGG backbone.
        """
        # choose default input
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 3))

        # create the vgg backbone
        if backbone == 'xception':
            xception = keras.applications.Xception(input_tensor=inputs,
                                                   include_top=False,
                                                   weights='imagenet'
                                                   )
        else:
            raise ValueError("Backbone '{}' not recognized.".format(backbone))

        if modifier:
            xception = modifier(xception)

        layer_names = ["block3_sepconv2_bn", "block4_sepconv2_bn", "block13_sepconv2_bn"]
        layer_outputs = [xception.get_layer(name).output for name in layer_names]
        layer_outputs = [None, None, *layer_outputs]

        return retinanet.retinanet(inputs=inputs,
                                   num_classes=num_classes,
                                   backbone_layers=layer_outputs,
                                   **kwargs)
