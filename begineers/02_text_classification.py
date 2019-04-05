#!/usr/bin/env python
# coding: utf-8

# -----------------------------------------------------------------------
# Copyright 2019 Damien Pretet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

# import numpy as np
import matplotlib.pyplot as plt


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


if __name__ == '__main__':

    print(tf.__version__)

    # Downloads the IMDB dataset to your machine (or uses a cached copy if you've already downloaded it):
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # Let's take a moment to understand the format of the data. The dataset comes preprocessed:
    # each example is an array of integers representing the words of the movie review.
    # Each label is an integer value of either 0 or 1, where 0 is a negative review,
    # and 1 is a positive review.

    print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

    # The text of reviews have been converted to integers, where each integer represents a
    # specific word in a dictionary. Here's what the first review looks like:

    print(train_data[0])

    # It may be useful to know how to convert integers back to text. Here, we'll create a
    # helper function to query a dictionary object that contains the integer to string mapping:

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    print(decode_review(train_data[0]))

    # The reviews—the arrays of integers—must be converted to tensors before fed into the neural network.

    # This conversion can be done a couple of ways:

    #    * Convert the arrays into vectors of 0s and 1s indicating word occurrence, similar to a
    #      one-hot encoding. For example, the sequence [3, 5] would become a 10,000-dimensional vector
    #      that is all zeros except for indices 3 and 5, which are ones. Then, make this the first
    #      layer in our network—a Dense layer—that can handle floating point vector data. This
    #      approach is memory intensive, though, requiring a num_words * num_reviews size matrix.

    #    * Alternatively, we can pad the arrays so they all have the same length, then create an
    #      integer tensor of shape max_length * num_reviews. We can use an embedding layer capable
    #      of handling this shape as the first layer in our network.

    # In this tutorial, we will use the second approach.
    # Since the movie reviews must be the same length, we will use the pad_sequences function to
    # standardize the lengths:

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)
    print(train_data[0])

    # Let's build the model

    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    # The layers are stacked sequentially to build the classifier:

    # * The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and
    #   looks up the embedding vector for each word-index. These vectors are learned as the model trains.
    #   The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
    #
    # * Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by
    #   averaging over the sequence dimension. This allows the model to handle input of variable length,
    #   in the simplest way possible.
    #
    # * This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
    #
    # * The last layer is densely connected with a single output node. Using the sigmoid activation function,
    #   this value is a float between 0 and 1, representing a probability, or confidence level.

    # A model needs a loss function and an optimizer for training. Since this is a binary classification
    # problem and the model outputs a probability (a single-unit layer with a sigmoid activation),
    # we'll use the binary_crossentropy loss function.

    # This isn't the only choice for a loss function, you could, for instance, choose mean_squared_error.
    # But, generally, binary_crossentropy is better for dealing with probabilities—it measures the "distance"
    # between probability distributions, or in our case, between the ground-truth distribution and the predictions.

    # Later, when we are exploring regression problems (say, to predict the price of a house), we will
    # see how to use another loss function called mean squared error.

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['acc'])

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=40,
                        batch_size=512,
                        validation_data=(x_val, y_val),
                        verbose=1)

    results = model.evaluate(test_data, test_labels)

    print(results)

    history_dict = history.history
    history_dict.keys()

    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'bo', label='Training loss')
    # b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.clf()   # clear figure

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
