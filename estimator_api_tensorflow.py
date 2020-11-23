"""
Purvang Lapsiwala

code provides overview of tensorflow Estimator class to train custom model.
# train_input_fun
# eval_input_fun
# estimator
# train_spec
# eval_spec
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

NUM_EPOCHS = 50
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

img_size = x_train.shape[1]
img_size_flat = x_train.shape[1] * x_train.shape[2]
img_shape = (x_train.shape[1], x_train.shape[2])
num_classes = 10
num_channels = 1

y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


images = x_test[0:9]
cls_true = y_test[0:9]
# plot_images(images=images, cls_true=cls_true)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_train)},
    y=np.array(y_train),
    batch_size=8,
    num_epochs=NUM_EPOCHS,
    shuffle=True)

test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": np.array(x_test)},
    y=np.array(y_test),
    batch_size=8,
    num_epochs=1,
    shuffle=False
)

predict_images = x_test[0:9]

predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": predict_images},
    num_epochs=1,
    shuffle=False)


# =========================================================================================
# below code section can be use, if client wants to use tf provided estimator.
# feature_x = tf.feature_column.numeric_column("x", shape=img_shape)
#
# feature_columns = [feature_x]
#
# num_hidden_units = [512, 256, 128]

# model = tf.estimator.DNNClassifier(feature_columns=feature_columns,
#                                    hidden_units=num_hidden_units,
#                                    activation_fn=tf.nn.relu,
#                                    n_classes=num_classes,
#                                    model_dir="./dataset/checkpoints",
#                                    batch_norm= True)
#
#
# # model.train(input_fn=train_input_fn, steps=1000)
# result = model.evaluate(input_fn=test_input_fn)
# =========================================================================================


def model_fn(features, labels, mode, params):
    x = features["x"]
    net = tf.reshape(x, [-1, img_size, img_size, num_channels])
    net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                           filters=16, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=36, kernel_size=5,
                           padding='same', activation=tf.nn.relu)
    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    net = tf.contrib.layers.flatten(net)

    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=10)

    logits = net
    y_pred = tf.nn.softmax(logits=logits)

    y_pred_cls = tf.argmax(y_pred, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=y_pred_cls)

    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                       logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        metrics = \
            {
                "accuracy": tf.metrics.accuracy(labels, y_pred_cls)
            }

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)

    return spec


params = {"learning_rate": 1e-4}

model = tf.estimator.Estimator(model_fn=model_fn,
                               params=params,
                               model_dir="./checkpoints_tutorial17-2/")

model.train(input_fn=train_input_fn, steps=2000)
