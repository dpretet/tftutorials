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

# import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


if __name__ == '__main__':

    # Use a well-known dataset and import it with pandas
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "https://archive.ics.uci.edu/ml/machine-learning-databases/\
                                        auto-mpg/auto-mpg.data")

    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    print(dataset.tail())

    # Detect missing values
    print(dataset.isna().sum())

    # And drop them
    dataset = dataset.dropna()

    # The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
    origin = dataset.pop('Origin')
    print(dataset.tail())

    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0

    print(dataset.tail())

    # Now split the dataset into a training set and a test set.
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    # plt.show()

    # look at the overall statistics:
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    print(train_stats)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # It is good practice to normalize features that use different scales and ranges.
    # Although the model might converge without feature normalization, it makes training more
    # difficult, and it makes the resulting model dependent on the choice of units used in the input.

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # Build the model

    # Let's build our model. Here, we'll use a Sequential model with two densely connected hidden
    # layers, and an output layer that returns a single, continuous value. The model building steps
    # are wrapped in a function, build_model, since we'll create a second model, later on.

    def build_model():

        model = keras.Sequential([
          layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
          layers.Dense(64, activation=tf.nn.relu),
          layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mean_squared_error',
                      optimizer=optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        return model

    model = build_model()

    model.summary()

    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    # Train the model

    # Train the model for 1000 epochs, and record the training and validation
    # accuracy in the history object.

    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs):

            if epoch % 100 == 0:
                print('.', end='')

    EPOCHS = 1000

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()

    plot_history(history)

    # This graph shows little improvement, or even degradation in the validation error after about
    # 100 epochs. Let's update the model.fit call to automatically stop training when the validation
    # score doesn't improve. We'll use an EarlyStopping callback that tests a training condition for
    # every epoch. If a set amount of epochs elapses without showing improvement, then automatically
    # stop the training.

    # You can learn more about this callback here.

    model = build_model()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                        validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

    plot_history(history)

    # The graph shows that on the validation set, the average error is usually around + /- 2 MPG.
    # Is this good? We'll leave that decision up to you.

    # Let's see how well the model generalizes by using the test set, which we did not use when
    # training the model. This tells us how well we can expect the model to predict when we use it
    # in the real world.

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

    print("")
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    # Make predictions

    # Finally, predict MPG values using data in the testing set:

    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    # It looks like our model predicts reasonably well. Let's take a look at the error distribution.

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()

    # It's not quite gaussian, but we might expect that because the number of samples is very small.

    # Conclusion
    #
    # This notebook introduced a few techniques to handle a regression problem.

    # Mean Squared Error (MSE) is a common loss function used for regression problems (different loss
    # functions are used for classification problems).
    # Similarly, evaluation metrics used for regression differ from classification. A common
    # regression metric is Mean Absolute Error (MAE).
    # When numeric input data features have values with different ranges, each feature should be
    # scaled independently to the same range.
    # If there is not much training data, one technique is to prefer a small network with few hidden
    # layers to avoid overfitting.
    # Early stopping is a useful technique to prevent overfitting.
