# -*- coding: utf-8 -*-
# @File    : ResNet.py
# @Author  : Hua Guo
# @Time    : 2019/12/25 下午9:56
# @Disc    :
import tensorflow as tf
from tensorflow.keras import (
    layers,
    Model
)
import matplotlib.pyplot as plt


class ConvBlock(Model):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.c1 = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            data_format="channels_last",
            strides=[1, 1],
            padding="same",
            # use_bias=False
        )
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation("relu")

    def call(self, inputs, training=None, mask=None):
        x = self.c1(inputs)
        for layer in [
            self.bn,
            self.activation
        ]:
            x = layer(x)
        return x


class ResidualBlock(Model):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv_block = ConvBlock(filters=filters)
        self.conv2 = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            data_format="channels_last",
            strides=[1, 1],
            padding="same"
        )
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation("relu")

    def call(self, inputs, training=None, mask=None):
        x = self.conv_block(inputs)
        for layer in [
            self.conv2,
            self.bn2
        ]:
            x = layer(x)
        return self.activation2(tf.add(inputs, x))


class ResNet(Model):
    def __init__(self, label_num=10, res_block_num=10, conv_filter_num=10):
        super(ResNet, self).__init__()
        self.con_block1 = ConvBlock(filters=conv_filter_num)
        self.res_block_lst = []
        for _ in range(res_block_num):
            self.res_block_lst.append(
                ResidualBlock(filters=conv_filter_num)
            )
        self.con_block2 = ConvBlock(filters=2)
        self.flat = layers.Flatten()
        self.dense = layers.Dense(label_num, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        x = self.con_block1(inputs)
        for layer in self.res_block_lst:
            x = layer(x)
        for layer in [
            self.con_block2,
            self.flat,
            self.dense
        ]:
            x = layer(x)
        return x


if __name__ == "__main__":
    # config
    debug = True
    img_dimension = 28
    batch_size = 128
    epochs = 10
    if debug:
        train_num = 600
        test_num = 100
    else:
        train_num = -1
        test_num = -1


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train[:train_num, :, :]
    y_train = y_train[:train_num, ]
    x_test = x_test[:test_num, :, :]
    y_test = y_test[:test_num, ]


    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

    print(x_train.shape)
    print(x_test.shape)

    model = ResNet(
        label_num=10,
        res_block_num=5,
        conv_filter_num=10
    )
    # model(tf.ones(shape=(1, 32, 32, 10)))
    # print(model.summary())
    model.compile(
        optimizer='Adam'
        , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           metrics = ['accuracy']
    )
    # print(model.summary())
    history = model.fit(
        x=x_train
        , y=y_train
        , batch_size=batch_size
        , epochs=epochs
        , validation_data=(x_test, y_test)
    )
    print(model.summary())
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
