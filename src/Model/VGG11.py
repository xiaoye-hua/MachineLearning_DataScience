import tensorflow as tf
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D


class VGGBlock(Model):
    def __init__(self, cnn_number, filter_num):
        super(VGGBlock, self).__init__()
        self.layer_lst = []
        for _ in range(cnn_number):
            self.layer_lst.append(
                Conv2D(filters=filter_num, kernel_size=3, padding='same', activation='relu')
            )
        self.layer_lst.append(MaxPool2D(pool_size=2, strides=2))

    def call(self, inputs, training=None, mask=None):
        for layer in self.layer_lst:
            inputs = layer(inputs)
        return inputs


class VGG11(Model):
    def __init__(self, label_num):
        super(VGG11, self).__init__()
        # (cnn_num, filter_num)
        vgg_block_info = [(1, 64), (1, 128), (2, 256),
                          (2, 512)
                          , (2, 512)
                          ]
        self.layer_lst = []
        for cnn_num, filter_num in vgg_block_info:
            self.layer_lst.append(
                VGGBlock(cnn_number=cnn_num, filter_num=filter_num)
            )
        self.layer_lst.append(Dense(units=4096, activation='relu'))
        self.layer_lst.append(Dropout(rate=0.5))
        self.layer_lst.append(Dense(units=4096, activation='relu'))
        self.layer_lst.append(Dropout(rate=0.5))
        self.layer_lst.append(Dense(units=label_num))

    def call(self, inputs, training=None, mask=None):
        for layer in self.layer_lst:
            inputs = layer(inputs)
        return inputs


if __name__ == "__main__":
    # config
    debug = True
    img_dimension = 28
    target_dimension = 224
    batch_size = 256
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

    x_train = tf.image.resize(x_train, [target_dimension, target_dimension])
    x_test = tf.image.resize(x_test, [target_dimension, target_dimension])

    model = VGG11(label_num=10)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    sample_input = tf.ones(shape=[1, target_dimension, target_dimension, 1])
    print(sample_input.shape)
    model(sample_input)
    print(model.summary())
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


