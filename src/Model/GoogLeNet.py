# -*- coding: utf-8 -*-
# @File    : GoogLeNet.py
# @Author  : Hua Guo
# @Time    : 2021/8/14 上午9:26
# @Disc    :
from typing import List
from src.BaseClass.DLModel import DLModel
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAvgPool2D, Dense


class Inception(DLModel):
    def __init__(self, filter_num1: int, filter_num2: List[int], filter_num3:List[int], filter_num4: int):
        super(Inception, self).__init__()
        self.path1 = [Conv2D(filters=filter_num1, kernel_size=1, activation="relu")]
        self.path2 = [
            Conv2D(filters=filter_num2[0], kernel_size=1, activation='relu')
            , Conv2D(filters=filter_num2[1], kernel_size=3, padding='same', activation='relu')
        ]
        self.path3 = [
            Conv2D(filters=filter_num3[0], kernel_size=1, activation='relu')
            , Conv2D(filters=filter_num3[1], kernel_size=5, padding='same', activation='relu')
        ]
        self.path4 = [
            MaxPool2D(pool_size=3, strides=1,padding='same')
            , Conv2D(filters=filter_num4, kernel_size=1, activation='relu', padding='same')
        ]

    def call(self, inputs, training=None, mask=None):
        x1, x2, x3, x4 = inputs, inputs, inputs, inputs
        result_vector = []
        idx = 0
        for vector, path in zip([x1, x2, x3, x4], [self.path1, self.path2, self.path3, self.path4]):
            for layer in path:
                if idx == 3:
                    print(layer)
                    print(f"Shape before {vector.shape}")
                vector = layer(vector)
                if idx == 3:
                    print(f'Shape after {vector.shape}')
            result_vector.append(vector)
            idx += 1
        # for layer in result_vector:
        #     # print(layer)
        #     print(layer.shape)
        return tf.concat(result_vector, -1)


class GoogLeNet(DLModel):
    def __init__(self, label_num):
        super(GoogLeNet, self).__init__()
        # first component
        self.layers_lst.extend(
            [
                Conv2D(filters=64, kernel_size=7, padding='same', activation='relu')
                , MaxPool2D(pool_size=3, strides=2, padding='same')
            ]
        )
        # second component
        self.layers_lst.extend(
            [
                Conv2D(filters=64, kernel_size=1, padding='same', activation='relu')
                , Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')
                , MaxPool2D(pool_size=3, strides=2, padding='same')
            ]
        )
        # third part
        self.layers_lst.extend(
            [
                Inception(64, (96, 128), (16, 32), 32)
                , Inception(128, (128, 192), (32, 96), 64)
                , MaxPool2D(pool_size=3, strides=2, padding='same')
            ]
        )
        # fourth part
        self.layers_lst.extend(
            [
                Inception(192, [96, 208], [16, 48], 64),
                Inception(160, [112, 224], [24, 64], 64),
                Inception(128, [128, 256], [24, 64], 64),
                Inception(112, [144, 288], [32, 64], 64),
                Inception(256, [160, 320], [32, 128], 128),
                MaxPool2D(pool_size=3, strides=2, padding='same')
            ]
        )
        # fifth part
        self.layers_lst.extend(
            [
                Inception(256, [160, 320],  [32, 128],  128)
                , Inception(384, [192, 384],  [48, 128],  128)
                , GlobalAvgPool2D()
                , Dense(units=label_num)
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

    model = GoogLeNet(label_num=10)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    sample_input = tf.ones(shape=[1, target_dimension, target_dimension, 1])
    print(sample_input.shape)
    for layer in model.layers:
        print(layer)
        sample_input = layer(sample_input)
        print(sample_input.shape)
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





