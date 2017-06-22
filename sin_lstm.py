import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn



def sin_signal():
    '''
    generate a sin function
    the train set is ten periods in length
    the test set is one additional period
    the return variable is in pandas format for easy plotting
    '''
    X = np.arange(0, 2*np.pi*11, 0.5)
    y = np.sin(X)
    data = pd.DataFrame.from_dict({'X': X, 'y':y})
    train_data = data[data.X<=2*np.pi*10].copy()
    test_data = data[data.X>2*np.pi*10].copy()
    return train_data, test_data


class lstm_model():
    def __init__(self, size_x, size_y, num_units=32, num_layers=3, keep_prob=0.5):
        def single_unit():
            return rnn.DropoutWrapper(
                rnn.LSTMCell(num_units), output_keep_prob=keep_prob)

        self.graph = tf.Graph()
        with self.graph.as_default():
            '''input place holders'''
            self.X = tf.placeholder(tf.float32, [None, size_x], name='X')
            self.y = tf.placeholder(tf.float32, [None, size_y], name='y')

            '''network'''
            cell = rnn.MultiRNNCell([single_unit() for _ in range(num_layers)])
            X = tf.expand_dims(self.X, -1)
            val, state = tf.nn.dynamic_rnn(cell, X, time_major=True, dtype=tf.float32)
            val = tf.transpose(val, [1, 0, 2])
            last = tf.gather(val, int(val.get_shape()[0])-1)
            weights = tf.Variable(tf.truncated_normal([num_units, size_y], 0.0, 1.0), name='weights')
            bias = tf.Variable(tf.zeros(size_y), name='bias')
            predicted_y = tf.nn.xw_plus_b(last, weights, bias, name='predicted_y')

            '''optimizer'''
            optimizer = tf.train.AdamOptimizer(name='adam_optimizer')
            global_step = tf.Variable(0, trainable=False, name='global_step')
            self.loss = tf.reduce_mean(tf.squared_difference(predicted_y, self.y), name='mse_loss')
            self.train_op = optimizer.minimize(self.loss, global_step=global_step, name='training_op')

            '''initializer'''
            self.init_op = tf.global_variables_initializer()


class lstm_regressor():
    def __init__(self):
        if not os.path.isdir('./check_pts'):
            os.mkdir('./check_pts')


    @staticmethod
    def get_shape(dataframe):
        df_shape = dataframe.shape
        num_rows = df_shape[0]
        num_cols = 1 if len(df_shape)<2 else df_shape[1]
        return num_rows, num_cols


    def train(self, X_train, y_train, iterations):
        train_pts, size_x = lstm_regressor.get_shape(X_train)
        train_pts, size_y = lstm_regressor.get_shape(y_train)
        model = lstm_model(size_x=size_x, size_y=size_y, num_units=32, num_layers=1)

        with tf.Session(graph=model.graph) as sess:
            sess.run(model.init_op)
            saver = tf.train.Saver()
            feed_dict={
                model.X: X_train.values.reshape(-1, size_x),
                model.y: y_train.values.reshape(-1, size_y)
            }

            for step in range(iterations):
                _, loss = sess.run([model.train_op, model.loss], feed_dict=feed_dict)
                if step%100==0:
                    print('step={}, loss={}'.format(step, loss))
            saver.save(sess, './check_pts/lstm')


    def predict(self, X_test):
        test_pts, size_x = lstm_regressor.get_shape(X_test)
        X_np = X_test.values.reshape(-1, size_x)
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.import_meta_graph('./check_pts/lstm.meta')
                saver.restore(sess, './check_pts/lstm')
                X = graph.get_tensor_by_name('X:0')
                y_tf = graph.get_tensor_by_name('predicted_y:0')
                y_np = sess.run(y_tf, feed_dict={X: X_np})
                return y_np.reshape(test_pts)


def main():
    train_data, test_data = sin_signal()
    regressor = lstm_regressor()
    # regressor.train(train_data.X, train_data.y, iterations=100000)
    y_predicted = regressor.predict(test_data.X)
    test_data['y_predicted'] = y_predicted

    test_data[['y', 'y_predicted']].plot()

if __name__ == '__main__':
    main()
