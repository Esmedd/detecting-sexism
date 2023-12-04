import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sys
sys.path.append('models')

from official.nlp.data import classifier_data_lib
from official.nlp.modeling.models.bert_token_classifier import tokenization
from official.nlp import optimization
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# import transformers, evaluate, datasets


import pandas as pd


print("TF Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


train, test = pd.read_csv("/content/train_df_balanced.csv"), pd.read_csv("/content/test_df_balanced.csv")

train, test = train[["Unnamed: 0","text", "sexist_binary"]], test[["Unnamed: 0","text", "sexist_binary"]]

train, test = train.rename(columns = {'Unnamed: 0':'idx', 'sexist_binary':'label'}), test.rename(columns = {'Unnamed: 0':'idx', 'sexist_binary':'label'})
train['text'], test['text'] = train['text'].astype(str), test['text'].astype(str)



train, val = train_test_split(train, test_size=0.2, random_state=42)

from datasets import load_dataset
from evaluate import load


with tf.device('/cpu:0'):
    train_data = tf.data.Dataset.from_tensor_slices((train['text'].values, train['label'].values))
    valid_data = tf.data.Dataset.from_tensor_slices((val['text'].values, val['label'].values))
    # lets look at 3 samples from train set
    for text,label in train_data.take(3):
        print(text)
        print(label)


config = {'label_list' : [0, 1], # Label categories
          'max_seq_length' : 100, # maximum length of (token) input sequences
          'train_batch_size' : 32,
          'learning_rate': 2e-5,
          'epochs':50,
          'optimizer': 'adam',
          'dropout': 0.5,
          'train_samples': len(train_data),
          'valid_samples': len(valid_data)
         }


bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2',
                            trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy() # checks if the bert layer we are using is uncased or not
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)


def create_feature(text, label, label_list=config['label_list'], max_seq_length=config['max_seq_length'], tokenizer=tokenizer):
    """
    converts the datapoint into usable features for BERT using the classifier_data_lib


    Parameters:
    text: Input text string
    label: label associated with the text
    label_list: (list) all possible labels
    max_seq_length: (int) maximum sequence length set for bert
    tokenizer: the tokenizer object instantiated by the files in model assets


    Returns:
    feature.input_ids: The token ids for the input text string
    feature.input_masks: The padding mask generated
    feature.segment_ids: essentially here a vector of 0s since classification
    feature.label_id: the corresponding label id from lable_list [0, 1] here


    """


    # since we only have 1 sentence for classification purpose, textr_b is None
    example = classifier_data_lib.InputExample(guid = None,
                                            text_a = text.numpy(),
                                            text_b = None,
                                            label = label.numpy())
    # since only 1 example, the index=0
    feature = classifier_data_lib.convert_single_example(0, example, label_list,
                                    max_seq_length, tokenizer)


    return (feature.input_ids, feature.input_mask, feature.segment_ids, feature.label_id)



def create_feature_map(text, label):
    """
    A tensorflow function wrapper to apply the transformation on the dataset.
    Parameters:
    Text: the input text string.
    label: the classification ground truth label associated with the input string


    Returns:
    A tuple of a dictionary and a corresponding label_id with it. The dictionary
    contains the input_word_ids, input_mask, input_type_ids
    """


    input_ids, input_mask, segment_ids, label_id = tf.py_function(create_feature, inp=[text, label],
                                Tout=[tf.int32, tf.int32, tf.int32, tf.int32])
    max_seq_length = config['max_seq_length']


    # py_func doesn't set the shape of the returned tensors.
    input_ids.set_shape([max_seq_length])
    input_mask.set_shape([max_seq_length])
    segment_ids.set_shape([max_seq_length])
    label_id.set_shape([])


    x = {
        'input_word_ids': input_ids,
        'input_mask': input_mask,
        'input_type_ids': segment_ids
    }
    return (x, label_id)


# Now we will simply apply the transformation to our train and test datasets
with tf.device('/cpu:0'):
  # train
  train_data = (train_data.map(create_feature_map,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)


                          .shuffle(1000)
                          .batch(32, drop_remainder=True)
                          .prefetch(tf.data.experimental.AUTOTUNE))


  # valid
  valid_data = (valid_data.map(create_feature_map,
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
                          .batch(32, drop_remainder=True)
                          .prefetch(tf.data.experimental.AUTOTUNE))


train_data.element_spec

def create_model():

    input_word_ids = tf.keras.layers.Input(shape=(config['max_seq_length'],),
					    dtype=tf.int32,
                                           name="input_word_ids")


    input_mask = tf.keras.layers.Input(shape=(config['max_seq_length'],),
					dtype=tf.int32,
                                   	name="input_mask")


    input_type_ids = tf.keras.layers.Input(shape=(config['max_seq_length'],),
					    dtype=tf.int32,
                                    	    name="input_type_ids")




    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, input_type_ids])
    # for classification we only care about the pooled-output.
    # At this point we can play around with the classification head based on the
    # downstream tasks and its complexity


    drop = tf.keras.layers.Dropout(config['dropout'])(pooled_output)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(drop)


    # inputs coming from the function
    model = tf.keras.Model(
      inputs={
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
        'input_type_ids': input_type_ids},
      outputs=output)


    return model


model = create_model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.PrecisionAtRecall(0.5),
                       tf.keras.metrics.Precision(),
                       tf.keras.metrics.Recall()])
model.summary()


tf.keras.utils.plot_model(model=model, show_shapes=True, dpi=76, )

# Update CONFIG dict with the name of the model.
config['model_name'] = 'BERT_EN_UNCASED'
print('Training configuration: ', config)


# Initialize W&B run
run = wandb.init(project='Finetune-BERT-Text-Classification',
                 config=config,
                 group='BERT_EN_UNCASED',
                 job_type='train')


epochs = config['epochs']
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=1,
                    callbacks = [WandbCallback()])
run.finish()



# Train model
# setting low epochs as It starts to overfit with this limited data, please feel free to change
epochs = config['epochs']
history = model.fit(train_data,
                    validation_data=valid_data,
                    epochs=epochs,
                    verbose=1,
                    callbacks = [WandbCallback()])
run.finish()
