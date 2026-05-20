import numpy as np


def compute_bce_loss(y_true, y_pred):
    """
    Вычисляет среднее значение бинарной кросс-энтропии (BCE).
    """
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return loss


def accuracy_score(y_true, y_pred_classes):
    """
    Вычисляет долю правильных ответов (Accuracy).
    """
    return np.mean(y_true == y_pred_classes)


def confusion_matrix(y_true, y_pred):
    """
    Вычисляет элементы матрицы ошибок: TP, FP, TN, FN.
    Автоматически обрабатывает метки классов {-1, 1} и {0, 1}.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    y_true_bin = np.where(y_true <= 0, 0, 1)
    y_pred_bin = np.where(y_pred <= 0, 0, 1)

    TP = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
    FP = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
    TN = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
    FN = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

    return TP, FP, TN, FN


def precision_score(y_true, y_pred):
    TP, FP, _, _ = confusion_matrix(y_true, y_pred)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def recall_score(y_true, y_pred):
    TP, _, _, FN = confusion_matrix(y_true, y_pred)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def f1_score(y_true, y_pred):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0


def roc_curve_data(y_true, y_probs):
    """
    Генерирует точки FPR и TPR для построения ROC-кривой.
    """
    y_true = np.where(y_true.ravel() <= 0, 0, 1)
    y_probs = y_probs.ravel()

    desc_score_indices = np.argsort(y_probs)[::-1]
    y_probs = y_probs[desc_score_indices]
    y_true = y_true[desc_score_indices]

    tpr_list = [0.0]
    fpr_list = [0.0]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Выборка должна содержать оба класса для расчета ROC-кривой.")

    tp = 0
    fp = 0
    for i in range(len(y_probs)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    return np.array(fpr_list), np.array(tpr_list)


def roc_auc_score(fpr, tpr):
    """
    Вычисляет площадь под ROC-кривой (AUC) методом трапеций.
    Совместимо со всеми версиями NumPy (включая 2.0+).
    """
    # Сортируем FPR по возрастанию, чтобы dx всегда был положительным
    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    # Формула метода трапеций: сумма по i ( (y_i + y_{i-1}) / 2 * (x_i - x_{i-1}) )
    auc = np.sum((tpr_sorted[1:] + tpr_sorted[:-1]) / 2.0 * (fpr_sorted[1:] - fpr_sorted[:-1]))
    return auc