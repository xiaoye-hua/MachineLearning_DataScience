import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout, GlobalAvgPool2D


class NiNBlock(Model):
    def __init__(self, filter_num, kernel_size, padding, strids):
        super(NiNBlock, self).__init__()
        self.layers_lst = []
        self.layers_lst.append(
            Conv2D(filters=filter_num, kernel_size=kernel_size, padding=padding, strides=strids, activation='relu')
        )
        for _ in range(2):
            self.layers_lst.append(Conv2D(filters=filter_num, kernel_size=1, activation='relu'))

    def call(self, inputs, training=None, mask=None):
        for layer in self.layers_lst:
            inputs = layer(inputs)
        return inputs


class NiN(Model):
    def __init__(self, label_num):
        super(NiN, self).__init__()
        self.layer_lst = []
        NiNBlock_info = [
            [96, 11, 4, 'valid'],
            [256, 5, 1, 'same'],
            [384, 3, 1, 'same']
            # , [10, 3, 1, 1]
        ]
        for (filter_num, kernel_size, strides, pading) in NiNBlock_info:
            self.layer_lst.append(NiNBlock(filter_num=filter_num, kernel_size=kernel_size, strids=strides, padding=pading))
            self.layer_lst.append(MaxPool2D(pool_size=3, strides=2))
        self.layer_lst.append(Dropout(0.5))
        self.layer_lst.append(NiNBlock(filter_num=label_num, kernel_size=3, strids=1, padding='same'))
        self.layer_lst.append(GlobalAvgPool2D())
        self.layer_lst.append(Flatten())

    def call(self, inputs, training=None, mask=None):
        for layer in self.layer_lst:
            inputs = layer(inputs)
        return inputs


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

    model = NiN(label_num=10)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    sample_input = tf.ones(shape=[1, target_dimension, target_dimension, 1])
    print(sample_input.shape)
    for layer in model.layers:
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
