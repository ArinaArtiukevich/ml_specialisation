import numpy as np


class LogisticRegression:
    def __init__(self, alpha: float, l: float = 0):
        self.alpha = alpha
        self.l = l
        self.w: np.array
        self.b: int

    def sigmoid(self, z: np.array):
        return 1 / (1 + np.exp(-z))

    def get_cost(self, X: np.array, y: np.array, w: np.array, b: float):
        m, n = X.shape
        cost = 0
        for i in range(m):
            z = np.dot(w, X[i]) + b
            cost -= np.log(self.sigmoid(z)) if y[i] == 1 else np.log(1 - self.sigmoid(z))
        return (1 / m) * cost

    def get_regularized_cost(self, X: np.array, y: np.array, w: np.array, b: float):
        m, n = X.shape
        cost = self.get_cost(X, y, w, b)
        cost += (self.l / (2 * m)) * np.sum([w_j ** 2 for w_j in w])
        return cost

    def get_gradient_descent_step(self, X: np.array, y: np.array, w: np.array, b: float):
        m, n = X.shape
        dj_dw = np.zeros_like(X[0])
        dj_db = 0
        for i in range(m):
            z = np.dot(X[i], w) + b
            err = self.sigmoid(z) - y[i]
            for j in range(n):
                dj_dw[j] += err * X[i][j]
            dj_db += err
        return dj_dw / m, dj_db / m

    def get_regularized_gradient_descent_step(self, X: np.array, y: np.array, w: np.array, b: float):
        m, n = X.shape
        dj_dw, dj_db = self.get_gradient_descent_step(X, y, w, b)
        for j in range(n):
            dj_dw[j] += (self.l / m) * w[j]
        return dj_dw, dj_db

    def compute_gradient_descent(self, X: np.array, y: np.array, w: np.array, b: float, eps: float):
        prev_cost = 0
        current_cost = self.get_cost(X, y, w, b) if self.l == 0 else self.get_regularized_cost(X, y, w, b)
        current_w, current_b = w.copy(), b
        while np.abs(prev_cost - current_cost) > eps:
            dj_dw, dj_db = self.get_gradient_descent_step(X, y, current_w, current_b) if self.l == 0 else self.get_regularized_gradient_descent_step(X, y, w, b)
            current_w = current_w - self.alpha * dj_dw
            current_b = current_b - self.alpha * dj_db
            prev_cost = current_cost
            current_cost = self.get_cost(X, y, w, b) if self.l == 0 else self.get_regularized_cost(X, y, w, b)
        self.w, self.b = current_w, current_b
        return self.w, self.b


    def predict(self, X):
        m, n = X.shape
        p = np.zeros(m)
        for i in range(m):
            z_wb = np.dot(X[i], self.w) + self.b
            f_wb = self.sigmoid(z_wb)
            p[i] = 1 if f_wb > 0.5 else 0
        return p


if __name__ == '__main__':
    X_train = np.array([
        [4.17022005e-01, 7.20324493e-01, 1.14374817e-04],
        [3.02332573e-01, 1.46755891e-01, 9.23385948e-02],
        [1.86260211e-01, 3.45560727e-01, 3.96767474e-01],
        [5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
        [2.04452250e-01, 8.78117436e-01, 2.73875932e-02]
    ])
    y_train = np.array([0, 1, 0, 1, 0])
    w_initial = np.array([0.67046751, 0.4173048, 0.55868983])
    b_initial = 0.5
    lambda_initial = 0.7

    alpha = 5.0e-7
    eps = 0.001

    m, n = X_train.shape
    X_test = np.random.randn(X_train.shape[0], X_train.shape[1])

    lr = LogisticRegression(alpha, lambda_initial)
    lr.compute_gradient_descent(X_train, y_train, w_initial, b_initial, eps)

    print(lr.predict(X_test))
