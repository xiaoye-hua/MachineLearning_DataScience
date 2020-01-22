# -*- coding: utf-8 -*-
"""0109_Trnsformers.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pzwfkoFNGhjdTnq0W17ovf2b78O3YsbN

Module

    requirments:
        # !pip install transformers==2.2.0
        # !pip install tensorflow==2.0.0
        # !pip install  tensorflow_datasets
"""

# !pip install transformers==2.2.0
# !pip install tensorflow==2.0.0
# !pip install  tensorflow_datasets

import os
import tensorflow as tf
import tensorflow_datasets
from transformers import (
    BertTokenizer,
    TFBertForSequenceClassification, 
    BertConfig,
    glue_convert_examples_to_features, 
    # BertForSequenceClassification,
    glue_processors
    )
from transformers import (
    DistilBertConfig,
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification, 
)
from transformers import (
    AlbertConfig,
    AlbertTokenizer,
    TFAlbertForSequenceClassification, 
)
print(f"tensorflow version: {tf.__version__}")
print(f"GPU availiable:  {tf.test.is_gpu_available()}")

"""# Config"""

model_setting_dict = {
    "albert" : [
                AlbertConfig,
                AlbertTokenizer,
                TFAlbertForSequenceClassification,
                # "../../pretrained_models/nlp/albert_base_v2_transformers"
                "../../pretrained_models/nlp/albert_base_v2_transformers"
    ],
    "bert": [
        BertConfig,
        BertTokenizer,
        TFBertForSequenceClassification,
        "../../pretrained_models/nlp/uncased_L-12_H-768_A-12/"
    ]
}
model_type = "albert"
model_index = "albert-base-v2"

# model_type = "bert"
# model_index = "bert-base-uncased"
# model_index = "bert-base-chinese"

# script parameters
BATCH_SIZE = 32
EVAL_BATCH_SIZE = BATCH_SIZE * 2
USE_XLA = False
USE_AMP = False
EPOCHS = 3

TASK = "sst-2"

if TASK == "sst-2":
    TFDS_TASK = "sst2"
elif TASK == "sts-b":
    TFDS_TASK = "stsb"
else: 
    TFDS_TASK = TASK

num_labels = len(glue_processors[TASK]().get_labels())
print(f"Num of lables: {num_labels}")

tf.config.optimizer.set_jit(USE_XLA)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})

##################################
#    load model

# 1. load remote
# 2. download & load
# 3. download checkpoint & load
##################################
# 1. load remote -> success
config = model_setting_dict[model_type][0].from_pretrained(model_index, num_labels=num_labels)
tokenizer = model_setting_dict[model_type][1].from_pretrained(model_index)
model = model_setting_dict[model_type][2].from_pretrained(model_index, config=config)

# 2. download & load  -> failed
# tokenizer = model_setting_dict[model_type][1].from_pretrained(model_setting_dict[model_type][3])
# model = model_setting_dict[model_type][2].from_pretrained(model_setting_dict[model_type][3]
#                                                           # , config=config
#                                                           )

# 3. download checkpoint & load -> failed
# config = BertConfig.from_json_file("../../pretrained_models/nlp/uncased_L-12_H-768_A-12/bert_config.json")
# tokenizer = BertTokenizer.from_pretrained("../../pretrained_models/nlp/uncased_L-12_H-768_A-12/")
# model = TFBertForSequenceClassification.from_pretrained(
# "../../pretrained_models/nlp/uncased_L-12_H-768_A-12/",
#     # model_setting_dict[model_type][3]
#     # from_tf=True,
#     config=config
# )

#############################
# load data
#############################

data, info = tensorflow_datasets.load(
    # "sst2",
    # data_dir="../../data/nlp/glue",
    # download=False,
    f'glue/{TFDS_TASK}',
    with_info=True,
    # data_dir="../../data/nlp"
)
train_examples = info.splits['train'].num_examples
# MNLI expects either validation_matched or validation_mismatched
valid_examples = info.splits['validation'].num_examples

# Prepare dataset for GLUE as a tf.data.Dataset instance
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, 128, TASK)

# MNLI expects either validation_matched or validation_mismatched
valid_dataset = glue_convert_examples_to_features(data['validation'], tokenizer, 128, TASK)
train_dataset = train_dataset.shuffle(128).batch(BATCH_SIZE).repeat(-1)
valid_dataset = valid_dataset.batch(EVAL_BATCH_SIZE)

# Prepare training: Compile tf.keras model with optimizer, loss and learning rate schedule
opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
if USE_AMP:
    # loss scaling is currently required when using mixed precision
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, 'dynamic')


if num_labels == 1:
    loss = tf.keras.losses.MeanSquaredError()
else:
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=opt, loss=loss, metrics=[metric])

# Train and evaluate using tf.keras.Model.fit()
train_steps = train_examples//BATCH_SIZE
valid_steps = valid_examples//EVAL_BATCH_SIZE

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
                    validation_data=valid_dataset,
    validation_steps=valid_steps)

# Save TF2 model
os.makedirs('./save/', exist_ok=True)
model.save_pretrained('./save/')


