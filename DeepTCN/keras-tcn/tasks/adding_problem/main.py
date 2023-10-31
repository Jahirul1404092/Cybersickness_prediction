import numpy as np
from tensorflow.keras.callbacks import Callback
from utils import data_generator

from tcn import compiled_tcn, tcn_full_summary

x_train, y_train = data_generator(n=200, seq_length=6)
x_test, y_test = data_generator(n=400, seq_length=6)


class PrintSomeValues(Callback):

    def on_epoch_begin(self, epoch, logs={}):
        print('y_true, y_pred')
        print(np.hstack([y_test[:5], self.model.predict(x_test[:5])]))


def run_task():
    model = compiled_tcn(
        return_sequences=False,
        num_feat=x_train.shape[2],
        num_classes=0,
        nb_filters=24,
        kernel_size=8,
        dilations=[2 ** i for i in range(9)],
        nb_stacks=1,
        max_len=x_train.shape[1],
        use_skip_connections=False,
        use_weight_norm=True,
        regression=True,
        dropout_rate=0
    )

    tcn_full_summary(model)
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1,
              batch_size=256, callbacks=[PrintSomeValues()])
    history=model.predict(x_test)
    print(history.shape, y_test.shape)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score , confusion_matrix
    print(round(mean_squared_error(y_test[:,-1], history[:,-1]),3))


if __name__ == '__main__':
    run_task()
