import numpy as np
import tensorflow as tf


class CustomForwardProp:
    def sigmoid(self, z: np.ndarray | int) -> int:
        return 1 / (1 + np.exp(-z))

    def dense(self, a_in: np.array, W: np.array, b: np.array) -> np.array:
        units = W.shape[1]
        a_out = np.zeros(units)
        for j in range(units):
            z = np.dot(W[:, j], a_in) + b[j]
            a_out[j] = self.sigmoid(z)
        return a_out

    def dense_vectorized(self, a_in: np.array, W: np.array, b: np.array) -> np.array:
        a_out = self.sigmoid(np.matmul(a_in, W) + b)
        return a_out

    def custom_sequential(self, x: np.array, W1: np.array, b1: np.array, W2: np.array, b2: np.array) -> int:
        layer1 = self.dense(x, W1, b1)
        layer2 = self.dense(layer1, W2, b2)
        return layer2

    def predict(self, x: np.array, W1: np.array, b1: np.array, W2: np.array, b2: np.array) -> np.array:
        m = x.shape[0]
        y_hat = np.zeros((m, 1))
        for i in range(m):
            prediction = self.custom_sequential(x[i], W1, b1, W2, b2)
            y_hat[i] = 1 if prediction >= 0.5 else 0
        return y_hat


if __name__ == '__main__':
    W1_init = np.array([[-8.93, 0.29, 12.9], [-0.1, -7.32, 10.81]])
    b1_init = np.array([-9.82, -9.28, 0.96])
    W2_init = np.array([[-31.18], [-27.59], [-32.56]])
    b2_init = np.array([15.41])

    X_test = np.array([
        [200, 13.9],
        [200, 17]]
    )

    layer_norm = tf.keras.layers.Normalization(axis=-1)
    layer_norm.adapt(X_test)
    X_test_norm = layer_norm(X_test)

    predictions = CustomForwardProp().predict(X_test_norm, W1_init, b1_init, W2_init, b2_init)
    print(predictions)
