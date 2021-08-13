from abc import ABC

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense
import matplotlib.pyplot as plt


class AlexNet(Model, ABC):
    def __init__(self, label_num):
        super(AlexNet, self).__init__()
        conv1 = Conv2D(
            kernel_size=11
            , strides=1
            , filters=96
            , activation='relu'
        )
        conv2 = Conv2D(
            filters=256
            , kernel_size=5
            , padding='same'
            , activation='relu'
        )
        conv3 = Conv2D(
            filters=384
            , kernel_size=3
            , padding='same'
            , activation='relu'
        )
        conv4 = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        conv5 = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')
        max_pool1 = MaxPool2D(
            pool_size=3
            , strides=2
        )
        max_pool2 = MaxPool2D(
            pool_size=3
            , strides=2
        )
        max_pool3 = MaxPool2D(
            pool_size=3
            , strides=2
        )
        dense1 = Dense(
            units=4096
            , activation='relu'
        )
        dense2 = Dense(units=4096, activation='relu')
        dense3 = Dense(units=label_num)
        self.layers_lst = [conv1, max_pool1, conv2, max_pool2, conv3, conv4, conv5, max_pool3
                           , dense1, dense2, dense3]

    def call(self, inputs, training=None, mask=None):
        for layer in self.layers_lst:
            inputs = layer(inputs)
        return inputs


if __name__ == "__main__":
    # config
    debug = False
    img_dimension = 28
    batch_size = 128
    epochs = 10
    label_num = 10
    if debug:
        train_num = 600
        test_num = 100
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

    model = AlexNet(label_num=10)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
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

