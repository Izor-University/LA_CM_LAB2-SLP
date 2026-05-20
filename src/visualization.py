import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(train_loss, val_loss, title="Кривая обучения (Loss)"):
    epochs = range(1, len(train_loss) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linewidth=2, linestyle='--')
    plt.title(title)
    plt.xlabel('Эпохи')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()


def plot_decision_boundary(X, y, model, title="Разделяющая граница"):
    plt.figure(figsize=(8, 6))

    # Приводим метки к бинарному виду для визуализации (0 и 1)
    y_bin = np.where(y <= 0, 0, 1).ravel()

    plt.scatter(X[y_bin == 0][:, 0], X[y_bin == 0][:, 1], color='red', label='Класс 0 (-1)', alpha=0.6, edgecolors='k')
    plt.scatter(X[y_bin == 1][:, 0], X[y_bin == 1][:, 1], color='blue', label='Класс 1', alpha=0.6, edgecolors='k')

    w1, w2 = model.w[0, 0], model.w[1, 0]
    b = model.b

    if np.abs(w2) > 1e-5:
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x1_vals = np.array([x1_min, x1_max])
        x2_vals = -(w1 * x1_vals + b) / w2
        plt.plot(x1_vals, x2_vals, color='green', linewidth=2, label='w^T x + b = 0')
        plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
        plt.xlim(x1_min, x1_max)
    else:
        x1_val = -b / w1
        plt.axvline(x=x1_val, color='green', linewidth=2, label='w^T x + b = 0')

    # Фоновая закраска областей принятия решений
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z_bin = np.where(Z <= 0, 0, 1).reshape(xx.shape)

    plt.contourf(xx, yy, Z_bin, alpha=0.15, cmap=plt.cm.RdBu)
    plt.title(title)
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()


def plot_roc_curve(fpr, tpr, auc_val):
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc_val:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC-кривая')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()


def plot_misclassified(X, y_true, y_pred, title="Точки ошибок классификации"):
    y_true = np.where(y_true.ravel() <= 0, 0, 1)
    y_pred = np.where(y_pred.ravel() <= 0, 0, 1)

    correct = (y_true == y_pred)
    incorrect = ~correct

    plt.figure(figsize=(8, 6))
    plt.scatter(X[correct & (y_true == 0)][:, 0], X[correct & (y_true == 0)][:, 1],
                color='green', marker='o', alpha=0.5, label='Верно Класс 0', edgecolors='k')
    plt.scatter(X[correct & (y_true == 1)][:, 0], X[correct & (y_true == 1)][:, 1],
                color='blue', marker='o', alpha=0.5, label='Верно Класс 1', edgecolors='k')

    if np.any(incorrect):
        plt.scatter(X[incorrect][:, 0], X[incorrect][:, 1],
                    color='red', marker='X', s=130, label='Ошибки', edgecolors='black', linewidths=1.0)

    plt.title(f"{title} (Ошибок: {np.sum(incorrect)})")
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()