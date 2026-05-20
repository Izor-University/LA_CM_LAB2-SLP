import numpy as np
from sklearn.datasets import make_classification


def get_sklearn_data():
    """
    Базовая генерация данных по умолчанию.
    """
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )
    return X, y


class CustomDataGenerator:
    @staticmethod
    def generate_linear_data(n_samples=500, mean1=[1.5, 1.5], mean2=[-1.5, -1.5], cov=[[0.8, 0], [0, 0.8]],
                             noise_prob=0.0):
        """
        Линейно разделимые гауссовы облака.
        """
        n_class = n_samples // 2
        class_0 = np.random.multivariate_normal(mean1, cov, n_class)
        class_1 = np.random.multivariate_normal(mean2, cov, n_class)

        X = np.vstack((class_0, class_1))
        y = np.hstack((np.zeros(n_class), np.ones(n_class)))

        if noise_prob > 0:
            y = CustomDataGenerator._add_noise(y, noise_prob)
        return X, y

    @staticmethod
    def generate_xor_data(n_samples=500, noise_prob=0.0):
        """
        Нелинейные данные типа XOR.
        """
        X = np.random.uniform(-2, 2, (n_samples, 2))
        y = (X[:, 0] * X[:, 1] < 0).astype(int)

        if noise_prob > 0:
            y = CustomDataGenerator._add_noise(y, noise_prob)
        return X, y

    @staticmethod
    def generate_circle_data(n_samples=500, radius=1.2, noise_prob=0.0):
        """
        Нелинейные данные в форме кольца.
        """
        X = np.random.uniform(-2, 2, (n_samples, 2))
        dist_sq = X[:, 0] ** 2 + X[:, 1] ** 2
        y = (dist_sq < radius ** 2).astype(int)

        if noise_prob > 0:
            y = CustomDataGenerator._add_noise(y, noise_prob)
        return X, y

    @staticmethod
    def _add_noise(y, noise_prob):
        noise_mask = np.random.rand(len(y)) < noise_prob
        y_noisy = np.logical_xor(y, noise_mask).astype(int)
        return y_noisy