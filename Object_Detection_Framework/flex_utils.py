import numpy as np
import keras
from matplotlib import pyplot
import os


class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.

    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```

    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """

    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)


class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding
        # epoch
        lrs = [self(i) for i in epochs]
        # the learning rate schedule
        pyplot.style.use("ggplot")
        pyplot.figure()
        pyplot.plot(epochs, lrs)
        pyplot.title(title)
        pyplot.xlabel("Epoch #")
        pyplot.ylabel("Learning Rate")


class StepDecay(LearningRateDecay):
    def __init__(self, init_lr=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.init_lr = init_lr
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.init_lr * (self.factor ** exp)
        # return the learning rate
        return float(alpha)


class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        alpha = self.initAlpha * decay
        # return the new learning rate
        return float(alpha)


class FeatureVisualizer(keras.callbacks.Callback):

    def __init__(self,
                 model,
                 image_path,
                 conv_features=False,
                 bn_features=False,
                 activation_features=False,
                 save_weights=False,
                 feature_freq_in_epochs=1,
                 output_layers_only=False,
                 feature_saving_dir_path="./features",
                 num_of_channels=1
                 ):

        super(FeatureVisualizer, self).__init__()
        self.model = model
        self.save_weights = save_weights
        self.conv_features = conv_features
        self.bn_features = bn_features
        self.activation_features = activation_features
        self.feature_freq_in_epochs = feature_freq_in_epochs
        self.output_layers_only = output_layers_only
        self.feature_saving_dir_path = feature_saving_dir_path
        self.image_path = image_path
        self.num_of_channels = num_of_channels

    def on_epoch_end(self, epoch, logs=None):
        if not any([self.save_weights, self.conv_features, self.bn_features, self.activation_features]):
            return

        if epoch % self.feature_freq_in_epochs == 0:

            img = keras.preprocessing.image.load_img(self.image_path, target_size=(720, 1280))
            img = keras.preprocessing.image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = keras.applications.resnet.preprocess_input(img)

            # if self.output_layers_only:
            #     self.build_map_saving_op(img, model_layers[2:])
            # else:
            self.build_map_saving_op(img, self.model.layers)

            #         # self.best_weights = self.model.get_weights()
            #         # layer = features[-1]
            #         # print(layer)
            #         # print(dir(layer))
            #         # print(self.model.layers[1])
            #         filters, biases = layer_obj.get_weights()
            #         print(layer_obj.name, filters.shape, biases.shape, layer_obj.input.shape, layer_obj.output.shape)

    def build_map_saving_op(self, img, list_layes_objects):
        for layer in list_layes_objects:
            if "bn3d_branch2a" not in layer.name:
                continue
            try:
                model = keras.models.Model(inputs=self.model.inputs, outputs=layer.output)
                if self.conv_features and (not "bn" in layer.name and not "relu" in layer.name
                                           and not "padding" in layer.name):
                    feature_map = model.predict(img)
                    if feature_map.shape[1] < 30:
                        continue
                    feature_saving_dir_path = os.path.join(self.feature_saving_dir_path, "conv")
                    os.makedirs(feature_saving_dir_path, exist_ok=True)

                    for i in range(self.num_of_channels):
                        feature_map = np.squeeze(feature_map)

                        pyplot.imsave(
                            os.path.join(feature_saving_dir_path, f'{layer.name}_{i}.jpeg'),
                            feature_map[:, :, i], cmap='gray')

                if self.activation_features and "relu" in layer.name:
                    feature_map = model.predict(img)
                    if feature_map.shape[1] < 30:
                        continue
                    feature_saving_dir_path = os.path.join(self.feature_saving_dir_path, "activation")
                    os.makedirs(feature_saving_dir_path, exist_ok=True)

                    for i in range(self.num_of_channels):
                        feature_map = np.squeeze(feature_map)
                        pyplot.imsave(
                            os.path.join(feature_saving_dir_path, f'{layer.name}_{i}.jpeg'),
                            feature_map[:, :, i], cmap='gray')

                if self.bn_features and "bn" in layer.name:
                    feature_map = model.predict(img)
                    if feature_map.shape[1] < 30:
                        continue
                    feature_saving_dir_path = os.path.join(self.feature_saving_dir_path, "bn")
                    os.makedirs(feature_saving_dir_path, exist_ok=True)
                    for i in range(self.num_of_channels):
                        feature_map = np.squeeze(feature_map)
                        pyplot.imsave(
                            os.path.join(feature_saving_dir_path, f'{layer.name}_{i}.jpeg'),
                            feature_map[:, :, i], cmap='gray')

            except Exception:
                continue
        return





