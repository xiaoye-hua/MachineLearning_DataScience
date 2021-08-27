# -*- coding: utf-8 -*-
# @File    : DenseNet.py
# @Author  : Hua Guo
# @Time    : 2019/12/25 下午9:56
# @Disc    :
import tensorflow as tf
from tensorflow.keras import (
    layers,
    Model
)
import matplotlib.pyplot as plt
from src.BaseClass.DLModel import DLModel


class ConvLayer(DLModel):
    def __init__(self, num_channel):
        super(ConvLayer, self).__init__()
        self.layers_lst.extend([layers.BatchNormalization()
                                   , layers.Activation("relu")
                                   , layers.Conv2D(filters=num_channel, kernel_size=3, padding='same', strides=1)
                                ])


class DenseBlock(DLModel):
    def __init__(self, num_conv, num_channel):
        super(DenseBlock, self).__init__()
        for _ in range(num_conv):
            self.layers_lst.append(ConvLayer(num_channel=num_channel))

    def call(self, inputs, training=None, mask=None):
        X = inputs
        # res = X.copy()
        for layer in self.layers_lst:
            Y = layer(X)
            X = tf.concat([X, Y], axis=-1)
        return X


class TransitionBlock(DLModel):
    def __init__(self, num_channel):
        super(TransitionBlock, self).__init__()
        self.layers_lst.extend(
            [layers.BatchNormalization()
            , layers.Activation('relu')
            , layers.Conv2D(filters=num_channel, kernel_size=1)
            , layers.MaxPool2D(pool_size=2, strides=2)]
        )


class DenseNet(DLModel):
    def __init__(self, label_num):
        super(DenseNet, self).__init__()
        channel_num = 64
        # first component
        self.layers_lst.extend(
            [
                layers.Conv2D(filters=channel_num, kernel_size=7, strides=2, padding='same')
                , layers.BatchNormalization()
                , layers.MaxPool2D(pool_size=3, strides=2, padding='same')
            ]
        )
        # second: dense and transition block
        growth_rate = 32
        num_conv_in_dense_block = [4, 4, 4, 4]
        for i, num_conv in enumerate(num_conv_in_dense_block):
            self.layers_lst.append(DenseBlock(num_channel=growth_rate, num_conv=num_conv))
            channel_num += num_conv*growth_rate
            if i != len(num_conv_in_dense_block)-1:
                channel_num = channel_num//2
                self.layers_lst.append(TransitionBlock(num_channel=channel_num))
        # third: final component
        self.layers_lst.extend(
            [
                layers.BatchNormalization()
                , layers.Activation('relu')
                , layers.GlobalAvgPool2D()
                , layers.Dense(units=label_num)
            ]
        )


if __name__ == "__main__":
    # config
    debug = True
    img_dimension = 28
    target_dimension = 224
    batch_size = 8
    epochs = 10
    label_num = 10
    if debug:
        train_num = 60
        test_num = 10
    else:
        train_num = -1
        test_num = -1

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:train_num, :, :]
    y_train = y_train[:train_num, ]
    x_test = x_test[:test_num, :, :]
    y_test = y_test[:test_num, ]


    x_train = (x_train.astype('float32')/225).reshape(x_train.shape[0], img_dimension, img_dimension, 1).astype('float32')
    x_test = (x_test.astype('float32')/225).reshape(x_test.shape[0], img_dimension, img_dimension, 1).astype('float32')

    x_train = tf.image.resize(x_train, [target_dimension, target_dimension])
    x_test = tf.image.resize(x_test, [target_dimension, target_dimension])

    model = DenseNet(label_num=10)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    sample_input = tf.ones(shape=[1, target_dimension, target_dimension, 1])
    print(sample_input.shape)
    # for layer in model.layers:
    #     print(layer)
    #     sample_input = layer(sample_input)
    #     print(sample_input.shape)
    # model(sample_input)
    # print(model.summary())
    history = model.fit(
        x=x_train
        , y=y_train
        , validation_data=(x_test, y_test)
        , epochs=epochs
        , batch_size=batch_size
    )
    print(model.summary())
    acc = history.history['accuracy']
    val_ac = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Train acc')
    plt.plot(epochs_range, val_ac, label='Val acc')
    plt.legend(loc='lower right')
    plt.title("Train & val accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Train loss')
    plt.plot(epochs_range, val_loss, label='Val loss')
    plt.legend(loc='upper right')
    plt.title("Train & val loss")
    plt.show()
