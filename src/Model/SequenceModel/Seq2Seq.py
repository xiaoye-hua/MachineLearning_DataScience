# -*- coding: utf-8 -*-
# @File    : Seq2Seq.py
# @Author  : Hua Guo
# @Time    : 2021/8/28 上午5:28
# @Disc    : sequence to sequence for machine translation problem
import tensorflow as tf
from tensorflow.keras import layers

from src.BaseClass.DLModel import DLModel
import src.utils.d2l as d2l


class Seq2SeqEncoder(DLModel):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.rnn = layers.RNN(
            layers.StackedRNNCells([layers.GRUCell(units=num_hiddens, dropout=dropout) for _ in range(num_layers)])
            , return_state=True
            , return_sequences=True
        )

    def call(self, inputs, training=None, mask=None):
        """
        Args:
            inputs:
            training:
            mask:

        Returns:
            ()
        """
        x = self.embed(inputs)
        output = self.rnn(x)
        return output[0], output[1:]


class Seq2SeqDecoder(DLModel):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_size)
        self.rnn = layers.RNN(
            layers.StackedRNNCells([layers.GRUCell(units=num_hiddens, dropout=dropout) for _ in range(num_layers)])
            , return_state=True
            , return_sequences=True
        )
        self.dense = layers.Dense(units=vocab_size)

    def call(self, inputs, hidden_state, training=None, mask=None):
        X = self.embed(inputs)
        state = tf.repeat(tf.expand_dims(hidden_state[-1], axis=1), repeats=X.shape[1], axis=1)
        x_state_concat = tf.concat([X, state], axis=-1)
        rnn_out = self.rnn(x_state_concat)
        output = self.dense(rnn_out[0])
        return output, rnn_out[1:]


class SeqSeq(DLModel):
    def __init__(self, source_vocab_size, target_vocab_size, embed_size, num_hiddens, num_layers, dropout=0):
        super(SeqSeq, self).__init__()
        self.encoder = Seq2SeqEncoder(vocab_size=source_vocab_size, embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
        self.decoder = Seq2SeqDecoder(vocab_size=target_vocab_size, embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)

    def call(self, encoder_X, decoder_X, training=None, mask=None):
        output, state = self.encoder(encoder_X)
        output, state = self.decoder(inputs=decoder_X, hidden_state=state)
        return output, state


class MaskSoftmaxCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, valid_len):
        # reduction='none' ensure the error will not be reduce_mean
        super(MaskSoftmaxCrossEntropy, self).__init__(reduction='none')
        self.valid_len = valid_len

    def call(self, y_ture, y_pred):
        """
        Args:
            pred: [batch_size, timestep, unit_num]
            label: [batch_size, timestep]

        Returns:
        """
        label = y_ture
        pred = y_pred
        weight = tf.ones_like(label, dtype=tf.float32)
        weight = sequence_mask(weight, self.valid_len)
        one_hot_label = tf.one_hot(label, depth=pred.shape[-1])
        noweighted_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction='none')(one_hot_label, pred)
        weight_loss = tf.reduce_mean(noweighted_loss*weight, axis=1)
        return weight_loss


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)

def train_seq2seq(net: SeqSeq, data_iter, lr, num_epochs, tgt_vocab, device):
    optimizor = tf.keras.optimizers.Adam(learning_rate=lr)
    metric = []
    for epoch in range(num_epochs):
        for data in data_iter:
            X, X_len, Y, Y_len = [x for x in data]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]),
                             shape=(-1, 1))
            dec_input = tf.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with tf.GradientTape() as tape:
                Y_hat, _ = net(encoder_X=X, decoder_X=dec_input)
                l = MaskSoftmaxCrossEntropy(Y_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = d2l.grad_clipping(gradients, 1)
            optimizor.apply_gradients(zip(gradients, net.trainable_variables))
            token_num = tf.reduce_sum(Y_len).numpy()
            error = tf.reduce_sum(l)
            # error for every token
            metric.append(error/token_num)
            # print(epoch)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch: {epoch +1}; Error: {metric[-1]}')


if __name__ == "__main__":
    pass

# #     # decoder
#     decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
#                              num_layers=2)
#     # state = decoder.init_state(encoder(X))
#     output, state = decoder(X, encoder(X)[1], training=False)
#     output.shape, len(state), state[0].shape
#
#     loss = MaskSoftmaxCrossEntropy(tf.constant([4, 2, 0]))
#     res = loss(y_true=tf.ones((3, 4), dtype=tf.int32), y_pred=tf.ones((3, 4, 10))).numpy()
#
#     embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
#     batch_size, num_steps = 64, 10
#     lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
#
#     train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
#     net = SeqSeq(source_vocab_size=len(src_vocab), target_vocab_size=len(tgt_vocab), embed_size=embed_size, num_hiddens=num_hiddens, num_layers=num_layers, dropout=dropout)
#     train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
#

