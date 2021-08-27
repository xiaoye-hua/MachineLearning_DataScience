# -*- coding: utf-8 -*-
# @File    : ResNet18.py
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


class ResidualBlock(Model):
    def __init__(self, filter_num, conv_bypass=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters=filter_num, kernel_size=3, padding='same', strides=strides)
        self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation('relu')
        self.conv2 = layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=strides,
            padding="same"
        )
        self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation("relu")
        if conv_bypass:
            self.conv3 = layers.Conv2D(filters=filter_num, kernel_size=1, padding='valid', strides=strides)
        else:
            self.conv3 = None

    def call(self, inputs, training=None, mask=None):
        print(f'Residual block')
        print('='*20)
        x = self.activation1(self.bn1(self.conv1(inputs)))
        for layer in [
            self.conv2
            , self.bn2
        ]:
            print(x.shape)
            x = layer(x)
        if self.conv3:
            inputs = self.conv3(inputs)
        print(x.shape)
        print(inputs.shape)
        return self.activation2(tf.add(inputs, x))


class ResidualCompoent(DLModel):
    """
    seveal residual block in one residualcomponent
    """
    def __init__(self, filter_num, residul_block_num, first_component=False):
        super(ResidualCompoent, self).__init__()
        for idx in range(residul_block_num):
            if idx == 0 and first_component:
                self.layers_lst.append(ResidualBlock(filter_num=filter_num, strides=2, conv_bypass=True))
            else:
                self.layers_lst.append(ResidualBlock(filter_num=filter_num, strides=1))


class ResNet18(DLModel):
    def __init__(self, label_num):
        super(ResNet18, self).__init__()
        # first component
        self.layers_lst.append(layers.Conv2D(filters=7, kernel_size=3, padding='same', strides=2))
        self.layers_lst.append(layers.BatchNormalization())
        self.layers_lst.append(layers.Activation('relu'))
        # Second - fifth component
        self.layers_lst.append(ResidualCompoent(filter_num=64, residul_block_num=2, first_component=True))
        for filter_num in [128, 256, 512]:
            self.layers_lst.append(ResidualCompoent(filter_num=filter_num, residul_block_num=2))
        # Sixth component
        self.layers_lst.append(layers.GlobalAvgPool2D())
        self.layers_lst.append(layers.Dense(units=label_num))


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

    model = ResNet18(label_num=10)
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
