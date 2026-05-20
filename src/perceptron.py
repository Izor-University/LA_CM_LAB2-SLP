import numpy as np
from src.metrics import compute_bce_loss


class Perceptron:
    def __init__(self, input_size, loss_fn='bce', l2_lambda=0.0):
        """
        Однослойный перцептрон с поддержкой BCE, Hinge Loss, L2 и Momentum SGD.
        """
        self.input_size = input_size
        self.loss_fn = loss_fn.lower()
        self.l2_lambda = l2_lambda

        # Инициализация параметров
        self.w = np.random.randn(input_size, 1) * 0.01
        self.b = 0.0

    def sigmoid(self, z):
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    def forward(self, X):
        """
        Прямой проход. Возвращает вероятности для BCE или z для Hinge.
        """
        z = np.dot(X, self.w) + self.b
        if self.loss_fn == 'bce':
            return self.sigmoid(z)
        return z

    def predict(self, X):
        """
        Предсказание дискретных классов.
        """
        outputs = self.forward(X)
        if self.loss_fn == 'bce':
            return (outputs >= 0.5).astype(int)
        return np.where(outputs >= 0, 1, -1)

    def _compute_l2_loss(self):
        return 0.5 * self.l2_lambda * np.sum(self.w ** 2)

    def fit(self, X_train, y_train, X_val, y_val, epochs=100, lr=0.1, batch_size=32, momentum_beta=0.0):
        """
        Обучение перцептрона с мини-батчами.
        """
        y_train = y_train.reshape(-1, 1).copy()
        y_val = y_val.reshape(-1, 1).copy()

        if self.loss_fn == 'hinge':
            y_train[y_train == 0] = -1
            y_val[y_val == 0] = -1

        self.history = {'train_loss': [], 'val_loss': []}
        n_samples = X_train.shape[0]

        # Переменные для Momentum SGD
        v_w = np.zeros_like(self.w)
        v_b = 0.0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                m_batch = X_batch.shape[0]

                # Шаг 1. Прямой ход
                outputs = self.forward(X_batch)

                # Шаг 2. Расчет градиентов (Обратный ход)
                if self.loss_fn == 'bce':
                    dz = outputs - y_batch
                    dw = (1 / m_batch) * np.dot(X_batch.T, dz)
                    db = (1 / m_batch) * np.sum(dz)
                elif self.loss_fn == 'hinge':
                    condition = y_batch * outputs < 1
                    dw = np.zeros_like(self.w)
                    db = 0.0
                    if np.any(condition):
                        X_err = X_batch[condition.ravel()]
                        y_err = y_batch[condition.ravel()]
                        m_err = X_err.shape[0]
                        dw = - (1 / m_err) * np.dot(X_err.T, y_err)
                        db = - (1 / m_err) * np.sum(y_err)

                # Применение L2-регуляризации
                if self.l2_lambda > 0:
                    dw += self.l2_lambda * self.w

                # Шаг 3. Обновление весов с использованием Momentum
                v_w = momentum_beta * v_w + dw
                v_b = momentum_beta * v_b + db

                self.w -= lr * v_w
                self.b -= lr * v_b

            # Оценка функции потерь за эпоху
            train_outputs = self.forward(X_train)
            val_outputs = self.forward(X_val)

            if self.loss_fn == 'bce':
                t_loss = compute_bce_loss(y_train, train_outputs) + self._compute_l2_loss()
                v_loss = compute_bce_loss(y_val, val_outputs) + self._compute_l2_loss()
            else:
                t_loss = np.mean(np.maximum(0, 1 - y_train * train_outputs)) + self._compute_l2_loss()
                v_loss = np.mean(np.maximum(0, 1 - y_val * val_outputs)) + self._compute_l2_loss()

            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)

        return self.history