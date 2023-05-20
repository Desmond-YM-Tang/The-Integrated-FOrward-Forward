import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import timeit

# Setting the random seeds for reproducability
tf.random.set_seed(42)


class FFDense(keras.layers.Layer):
    """
    A custom ForwardForward-enabled Dense layer.
    """
    def __init__(
        self,
        units,
        optimizer,
        loss_metric,
        num_epochs=50,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dense = keras.layers.Dense(
            units=units,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
        )
        self.relu = keras.layers.ReLU()
        self.optimizer = optimizer
        self.loss_metric = loss_metric
        self.threshold = 1.5
        self.num_epochs = num_epochs

    def call(self, x):
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True)
        x_norm = x_norm + 1e-4
        x_dir = x / x_norm
        res = self.dense(x_dir)
        return self.relu(res)

    # The Forward-Forward algorithm
    def forward_forward(self, x_pos, x_neg):
        for i in range(self.num_epochs):
            with tf.GradientTape() as tape:
                g_pos = tf.math.reduce_mean(tf.math.pow(self.call(x_pos), 2), 1)
                g_neg = tf.math.reduce_mean(tf.math.pow(self.call(x_neg), 2), 1)
                loss = tf.math.log(
                    1
                    + tf.math.exp(
                        tf.concat([-g_pos + self.threshold, g_neg - self.threshold], 0)
                    )
                )
                mean_loss = tf.cast(tf.math.reduce_mean(loss), tf.float32)
                self.loss_metric.update_state([mean_loss])
            gradients = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (
            tf.stop_gradient(self.call(x_pos)),
            tf.stop_gradient(self.call(x_neg)),
            self.loss_metric.result(),
        )


class FFNetwork(keras.Model):
    """
    A keras.Model that supports a `FFDense` network creation.
    """
    def __init__(
        self,
        dims,
        layer_optimizer=keras.optimizers.Adam(learning_rate=0.03),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_optimizer = layer_optimizer
        self.loss_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.loss_count = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.layer_list = [keras.Input(shape=(dims[0],))]
        for d in range(len(dims) - 1):
            self.layer_list += [
                FFDense(
                    dims[d + 1],
                    optimizer=self.layer_optimizer,
                    loss_metric=keras.metrics.Mean(),
                )
            ]

    # Encoding the first 10 pixels
    def overlay_y_on_x(self, data):
        X_sample, y_sample = data
        max_sample = tf.reduce_max(X_sample, axis=0, keepdims=True)
        max_sample = tf.cast(max_sample, dtype=tf.float64)
        X_zeros = tf.zeros([10], dtype=tf.float64)
        indices = tf.reshape(y_sample, [-1, 1])
        X_update = tf.tensor_scatter_nd_update(X_zeros, indices, max_sample)
        # Generate a range of indices to match the size of X_update
        indices = tf.reshape(tf.range(10), [-1, 1])
        X_sample = tf.tensor_scatter_nd_update(X_sample, indices, X_update)

        return X_sample, y_sample

    # A custom `predict_one_sample`
    def predict_one_sample(self, x):
        goodness_per_label = []
        x = tf.reshape(x, [tf.shape(x)[0] * tf.shape(x)[1]])
        for label in range(10):
            h, label = self.overlay_y_on_x(data=(x, label))
            h = tf.reshape(h, [-1, tf.shape(h)[0]])
            goodness = []
            for layer_idx in range(1, len(self.layer_list)):
                layer = self.layer_list[layer_idx]
                h = layer(h)
                goodness += [tf.math.reduce_mean(tf.math.pow(h, 2), 1)]
            goodness_per_label += [
                tf.expand_dims(tf.reduce_sum(goodness, keepdims=True), 1)
            ]
        goodness_per_label = tf.concat(goodness_per_label, 1)
        return tf.cast(tf.argmax(goodness_per_label, 1), tf.float64)

    def predict(self, data):
        x = data
        preds = list()
        preds = tf.map_fn(fn=self.predict_one_sample, elems=x)
        return np.asarray(preds, dtype=int)

    # The custom `train_step`
    def train_step(self, data):
        x, y = data
        x = tf.reshape(x, [-1, tf.shape(x)[1] * tf.shape(x)[2]])
        x_pos, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, y))
        random_y = tf.random.shuffle(y)
        x_neg, y = tf.map_fn(fn=self.overlay_y_on_x, elems=(x, random_y))
        h_pos, h_neg = x_pos, x_neg
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, FFDense):
                print(f"Training layer {idx+1} now : ")
                h_pos, h_neg, loss = layer.forward_forward(h_pos, h_neg)
                self.loss_var.assign_add(loss)
                self.loss_count.assign_add(1.0)
            else:
                print(f"Passing layer {idx+1} now : ")
                x = layer(x)
        mean_res = tf.math.divide(self.loss_var, self.loss_count)
        return {"FinalLoss": mean_res}


if __name__=='__main__':
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(float) / 255
    x_test = x_test.astype(float) / 255
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    # Noise
    train_data_mean = np.mean(x_train)
    NUM_Noises = 20000
    epsilon = 0.1
    # Type 1 Data with Gaussian Noises
    x_train[20000:40000, :, :] = x_train[20000:40000, :, :] + epsilon * np.random.randn(20000, np.shape(x_train)[1],
                                                                                        np.shape(x_train)[2])
    # Type 2
    unloaded_line = np.random.randint(low=1, high=28, size=(20000))
    for i in range(20000):
        x_train[40000 + i, 28 - unloaded_line[i]:28, :] = 0
    # Type 3 Completely Gaussian Noise
    noise_data = train_data_mean + epsilon * np.random.randn(NUM_Noises, np.shape(x_train)[1], np.shape(x_train)[2])
    noise_label = np.random.randint(low=0, high=10, size=[NUM_Noises])
    x_train = np.concatenate((x_train, noise_data), axis=0)
    y_train = np.concatenate((y_train, noise_label), axis=0)
    # Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_dataset = train_dataset.batch(60000+NUM_Noises)
    test_dataset = test_dataset.batch(10000)
    # Model
    model = FFNetwork(dims=[784, 100, 100, 100, 100])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.03),
        loss="mse",
        jit_compile=False,
        metrics=[keras.metrics.Mean()],
    )
    # Training
    epochs = 15
    start_time = timeit.default_timer()
    history = model.fit(train_dataset, epochs=epochs, batch_size=128)
    end_time = timeit.default_timer()
    #  Results
    batch_size = 100
    Results = 0
    for i in range(int(x_test.shape[0]/batch_size)):
        preds = model.predict(tf.convert_to_tensor(x_test[i*batch_size:(i+1)*batch_size, :, :]))
        preds = preds.reshape((preds.shape[0], preds.shape[1]))
        results = accuracy_score(preds, y_test[i*batch_size:(i+1)*batch_size])
        Results += results
        print('Testing Batch:', i, '/',int(x_test.shape[0]/batch_size))
    Results = Results/int(x_test.shape[0]/batch_size)
    print(f"Test Accuracy score : {Results * 100}%")
    # Loss
    plt.plot(range(len(history.history["FinalLoss"])), history.history["FinalLoss"])
    plt.title("Loss over training")
    plt.show()
    # Time
    elapsed_time = end_time - start_time
    print(f"Elapsed time for training: {elapsed_time} seconds:")