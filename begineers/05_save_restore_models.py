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

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

if __name__ == '__main__':

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_labels = train_labels[:1000]
    test_labels = test_labels[:1000]

    train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
    test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

    # Define a model

    # Let's build a simple model we'll use to demonstrate saving and loading weights.

    # Returns a short sequential model
    def create_model():
        model = tf.keras.models.Sequential([
          keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
          keras.layers.Dropout(0.2),
          keras.layers.Dense(10, activation=tf.keras.activations.softmax)
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.sparse_categorical_crossentropy,
                      metrics=['accuracy'])

        return model

    # Create a basic model instance
    model = create_model()
    model.summary()

    # Save checkpoints during training
    # The primary use case is to automatically save checkpoints during and at the end of training.
    # This way you can use a trained model without having to retrain it, or pick-up training where
    # you left of—in case the training process was interrupted.
    #
    # tf.keras.callbacks.ModelCheckpoint is a callback that performs this task. The callback takes
    # a couple of arguments to configure checkpointing.
    #
    # Checkpoint callback usage
    #
    # Train the model and pass it the ModelCheckpoint callback:

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model = create_model()

    model.fit(train_images, train_labels, epochs=10,
              validation_data=(test_images, test_labels),
              callbacks=[cp_callback])  # pass callback to training

    # This may generate warnings related to saving the state of the optimizer.
    # These warnings (and similar warnings throughout this notebook)
    # are in place to discourage outdated usage, and can be ignored.

    model = create_model()

    loss, acc = model.evaluate(test_images, test_labels)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

    # Then load the weights from the checkpoint, and re-evaluate:

    model.load_weights(checkpoint_path)
    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # Checkpoint callback options

    # The callback provides several options to give the resulting checkpoints unique names,
    # and adjust the checkpointing frequency.

    # Train a new model, and save uniquely named checkpoints once every 5-epochs:

    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        # Save weights, every 5-epochs.
        period=5)

    model = create_model()
    model.save_weights(checkpoint_path.format(epoch=0))
    model.fit(train_images, train_labels,
              epochs=50, callbacks=[cp_callback],
              validation_data=(test_images, test_labels),
              verbose=0)

    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)

    # To test, reset the model and load the latest checkpoint:

    # model = create_model()
    # model.load_weights(latest)
    # loss, acc = model.evaluate(test_images, test_labels)
    # print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # What are these files?
    # The above code stores the weights to a collection of checkpoint-formatted files that contain
    # only the trained weights in a binary format. Checkpoints contain: * One or more shards that
    # contain your model's weights. * An index file that indicates which weights are stored in a
    # which shard.

    # If you are only training a model on a single machine, you'll have one shard
    # with the suffix: .data-00000-of-00001

    # Manually save weights
    # Above you saw how to load the weights into a model.

    # Manually saving the weights is just as simple, use the Model.save_weights method.

    # Save the weights
    model.save_weights('./checkpoints/my_checkpoint')

    # Restore the weights
    model = create_model()
    model.load_weights('./checkpoints/my_checkpoint')

    loss, acc = model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # Save the entire model
    # The entire model can be saved to a file that contains the weight values, the model's
    # configuration, and even the optimizer's configuration (depends on set up). This allows you to
    # checkpoint a model and resume training later—from the exact same state—without access to
    # the original code.

    # Saving a fully-functional model is very useful—you can load them in TensorFlow.js (HDF5,
    #         Saved Model) and then train and run them in web browsers, or convert them to run on
    # mobile devices using TensorFlow Lite (HDF5, Saved Model)

    # As an HDF5 file

    # Keras provides a basic save format using the HDF5 standard. For our purposes, the saved model
    # can be treated as a single binary blob.

    model = create_model()

    model.fit(train_images, train_labels, epochs=5)

    # Save entire model to a HDF5 file
    model.save('my_model.h5')

    # Now recreate the model from that file:

    # Recreate the exact same model, including weights and optimizer.
    new_model = keras.models.load_model('my_model.h5')
    new_model.summary()

    # Check its accuracy:

    loss, acc = new_model.evaluate(test_images, test_labels)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

    # This technique saves everything:

    #  - The weight values
    #  - The model's configuration(architecture)
    #  - The optimizer configuration

    # Keras saves models by inspecting the architecture. Currently, it is not able to save
    # TensorFlow optimizers (from tf.train). When using those you will need to re-compile the
    # model after loading, and you will lose the state of the optimizer.

    # It exsits also a possibility to save with 'save_model'
    # It is an experiemental feature which may evovle
