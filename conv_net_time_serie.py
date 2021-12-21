# convolutional neural net, time serie classification, stocks
# as expected, the model performed poorly (random walk ?), but we saw a
# possible way to treat a multivariate time serie classification problem
# improvement: labeling, wavelet transform

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from utils import clean, augment, split
from labeling import generate_labels

class Time_serie():

    def __init__(self, df, train_size=0.9, val_size=0.1, sequence_length=101):
        self.train_size = train_size
        self.val_size = val_size
        self.target = "signal"
        self.df = df
        self.sequence_length = sequence_length
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.y_train = None

    # filename to save model
    @property
    def model_filename(self) -> str:
        return "classif_multi.h5"

    # number of different labels
    @property
    def number_of_class(self)  -> int:
        return len(np.unique(self.y_train))

    # weight/count of each label in the training set
    @property
    def weights(self) -> dict:
        unique, counts = np.unique(self.y_train, return_counts=True)
        return dict(zip(unique, counts))

    # weight correction for loss computation
    @property
    def balance_weights(self) -> dict:
        balance_weights = {k: round(1 / v * 10000, 2) for k, v in self.weights.items()}
        print("weights for loss correction : ", balance_weights)
        return balance_weights

    # create train, val, test dataset
    def preprocess_data(self) -> None:
        # data augmentation
        df_augment = augment(self.df)
        # labeling
        df_signal = generate_labels(df_augment.copy(), self.sequence_length)
        # cleaning
        df_final, df_signal = clean(df_augment.copy(), df_signal.copy())
        # train dataframe
        train_df_scaled = self.build_train_df(df_final, df_signal)
        # validation dataframe
        val_df_scaled = self.build_val_df(df_final, df_signal)
        # test dataframe
        test_df_scaled = self.build_test_df(df_final, df_signal)
        # transfrom dataframes into sequence of length sequence_length
        self.x_train, self.y_train = self.make_dataset(train_df_scaled)
        self.x_val, self.y_val = self.make_dataset(val_df_scaled)
        self.x_test, self.y_test = self.make_dataset(test_df_scaled)

    def build_train_df(self, df, df_signal) -> pd.DataFrame:
        # split data
        train_df, _, _ = split(df, train_size=self.train_size, val_size=self.val_size)
        signal, _, _ = split(df_signal, train_size=self.train_size, val_size=self.val_size)
        # scaling minmax
        scaler = MinMaxScaler()
        train_df_scaled = scaler.fit_transform(train_df)
        # convert into pd.dataframe
        train_df_scaled = pd.DataFrame(train_df_scaled, index=train_df.index, columns=train_df.columns)
        # save scaler for val and test sets
        joblib.dump(scaler, 'scaler.save')
        # add signal
        train_df_scaled = train_df_scaled.join(signal)

        return train_df_scaled

    def build_val_df(self, df, df_signal) -> pd.DataFrame:
        # split data
        _, val_df, _ = split(df, train_size=self.train_size, val_size=self.val_size)
        _, signal, _ = split(df_signal, train_size=self.train_size, val_size=self.val_size)
        # load scaler
        scaler = joblib.load('scaler.save')
        # fit val data
        val_df_scaled = scaler.transform(val_df)
        # convert into pd.dataframe
        val_df_scaled = pd.DataFrame(val_df_scaled, index=val_df.index, columns=val_df.columns)
        # add signal
        val_df_scaled = val_df_scaled.join(signal)

        return val_df_scaled

    def build_test_df(self, df, df_signal) -> pd.DataFrame:
        # split data
        _, _, test_df = split(df, train_size=self.train_size, val_size=self.val_size)
        _, _, signal = split(df_signal, train_size=self.train_size, val_size=self.val_size)
        # load scaler
        scaler = joblib.load('scaler.save')
        # fit test data
        test_df_scaled = scaler.transform(test_df)
        # convert into pd.dataframe
        test_df_scaled = pd.DataFrame(test_df_scaled, index=test_df.index, columns=test_df.columns)
        # add signal
        test_df_scaled = test_df_scaled.join(signal)

        return test_df_scaled

    def make_dataset(self, df) -> np.array:
        df = df.copy()

        # prepare dataframe for TimeseriesGenerator object
        target = df[self.target]
        df.drop(columns=[self.target], inplace=True)

        # convert to numpy
        data = np.array(df, dtype=np.float32)
        target = np.array(target, dtype=np.float32)

        # create sequences, output shape -> (n_example, n_time_step, n_features)
        dataset = TimeseriesGenerator(
            data=data,
            targets=target,
            length=self.sequence_length,
            stride=1,
            shuffle=True,
            batch_size=len(data))

        X, y = dataset[0]

        return X, y

    def build_nn(self) -> None:

        # conv net : the intuition is that input (n_time_step,n_features) is like an image
        # but without chanels. 1D convolution should caputure the patterns if there are any

        input_shape = (self.x_train.shape[1], self.x_train.shape[2])
        inputs = Input(shape=input_shape)

        x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
        x = Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Flatten()(x)
        x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        x = Dense(54, activation="relu", kernel_initializer="he_normal")(x)

        # size of output equal to number of class
        outputs = Dense(units=self.number_of_class, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)

        model.summary()

        opt = keras.optimizers.Adam(learning_rate=0.00001)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        model.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])

        self.model = model

    def train(self, epoch=5, batch_size=128) -> None:

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=5,
                                                          mode='min')

        self.history = self.model.fit(x=self.x_train,
                                      y=self.y_train,
                                      epochs=epoch,
                                      validation_data=(self.x_val, self.y_val),
                                      callbacks=[early_stopping],
                                      class_weight=self.balance_weights,
                                      batch_size=batch_size)

        self.model.save(self.model_filename)

    def test(self) -> None:
        model = keras.models.load_model(self.model_filename)
        y_pred_prob = model.predict(self.x_test)
        y_pred = np.argmax(y_pred_prob, axis=1)

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(self.y_train))
        disp.plot(cmap=plt.cm.Blues)
        plt.show()


if __name__ == "__main__":
    project_name = "time_serie"
    pd.options.display.max_columns = 10
    df = pd.read_csv('data_5_min.csv')
    model = Time_serie(df=df.head(200000).copy(),
                       train_size=0.8,
                       val_size=0.1,
                       sequence_length=401, )

    model.preprocess_data()
    model.build_nn()
    model.train(epoch=10, batch_size=128)
    model.test()


