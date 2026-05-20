import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-8

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise ValueError("Сначала необходимо вызвать метод fit()")
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def train_test_split_stratified(X, y, test_size=0.3, random_state=None):
    """
    Разделяет данные со стратификацией классов.
    """
    if random_state is not None:
        np.random.seed(random_state)

    classes = np.unique(y)
    train_indices = []
    test_indices = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        n_test_cls = int(len(cls_indices) * test_size)
        test_indices.extend(cls_indices[:n_test_cls])
        train_indices.extend(cls_indices[n_test_cls:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]