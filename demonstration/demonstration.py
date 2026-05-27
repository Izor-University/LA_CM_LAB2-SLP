import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Добавление текущей директории в пути поиска модулей
sys.path.append(os.path.abspath('.'))

# Импорт пользовательских модулей проекта
from src.data_generator import get_sklearn_data, CustomDataGenerator
from src.preprocessing import train_test_split_stratified, StandardScaler
from src.perceptron import Perceptron
from src.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve_data, roc_auc_score
)
from src.visualization import (
    plot_learning_curve, plot_decision_boundary,
    plot_roc_curve, plot_misclassified
)


def run_mandatory_part():
    print("\n" + "=" * 50)
    print("=== ЧАСТЬ 1: ОБЯЗАТЕЛЬНЫЕ ЗАДАНИЯ ===")
    print("=" * 50)

    # 1. Подготовка и масштабирование данных
    print("\n[Шаг 1] Генерация и подготовка данных...")
    X, y = get_sklearn_data()
    X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Размер обучающей выборки: {X_train_scaled.shape}")
    print(f"Размер тестовой выборки: {X_test_scaled.shape}")
    print(f"Распределение классов в трейне: {np.bincount(y_train)}")
    print(f"Распределение классов в тесте:  {np.bincount(y_test)}")

    # 2. Обучение базовой модели
    print("\n[Шаг 2] Обучение базовой модели (BCE, lr=0.1, batch_size=32)...")
    np.random.seed(42)
    model = Perceptron(input_size=2, loss_fn='bce')
    history = model.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=32)

    train_preds = model.predict(X_train_scaled)
    test_preds = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train.reshape(-1, 1), train_preds)
    test_acc = accuracy_score(y_test.reshape(-1, 1), test_preds)

    print(f"Точность на обучении: {train_acc:.4f}")
    print(f"Точность на тесте:    {test_acc:.4f}")

    # Отрисовка базовых графиков
    plot_learning_curve(history['train_loss'], history['val_loss'], title="Кривая потерь базовой модели (BCE)")
    plot_decision_boundary(X_test_scaled, y_test, model, title="Граница решений базовой модели на тесте")

    # 3. Эксперимент со скоростью обучения
    print("\n[Шаг 3] Эксперимент: Влияние скорости обучения (lr)...")
    learning_rates = [0.001, 0.01, 0.5, 1.0]
    plt.figure(figsize=(10, 5))
    print(f"{'Learning Rate':<15} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
    print("-" * 53)
    for lr in learning_rates:
        np.random.seed(42)
        model_exp = Perceptron(input_size=2)
        hist_exp = model_exp.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=lr, batch_size=32)
        tr_acc = accuracy_score(y_train.reshape(-1, 1), model_exp.predict(X_train_scaled))
        te_acc = accuracy_score(y_test.reshape(-1, 1), model_exp.predict(X_test_scaled))
        print(f"{lr:<15} | {tr_acc:<15.4f} | {te_acc:<15.4f}")
        plt.plot(hist_exp['train_loss'], label=f'lr = {lr}')
    plt.title("Влияние темпа обучения (lr) на сходимость")
    plt.xlabel("Эпохи")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # 4. Эксперимент с размером батча
    print("\n[Шаг 4] Эксперимент: Влияние размера батча...")
    batch_sizes = [1, 16, 64, 256]
    plt.figure(figsize=(10, 5))
    print(f"{'Batch Size':<15} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
    print("-" * 53)
    for bs in batch_sizes:
        np.random.seed(42)
        model_exp = Perceptron(input_size=2)
        hist_exp = model_exp.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=bs)
        tr_acc = accuracy_score(y_train.reshape(-1, 1), model_exp.predict(X_train_scaled))
        te_acc = accuracy_score(y_test.reshape(-1, 1), model_exp.predict(X_test_scaled))
        print(f"{bs:<15} | {tr_acc:<15.4f} | {te_acc:<15.4f}")
        plt.plot(hist_exp['train_loss'], label=f'batch_size = {bs}')
    plt.title("Влияние размера батча на сходимость")
    plt.xlabel("Эпохи")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

    # 5. Эксперимент с инициализацией весов
    print("\n[Шаг 5] Эксперимент: Влияние инициализации весов...")
    initializations = {
        "Zeros": lambda: (np.zeros((2, 1)), 0.0),
        "Small Random": lambda: (np.random.randn(2, 1) * 0.01, 0.0),
        "Large Random (N(0, 10))": lambda: (np.random.randn(2, 1) * 10.0, 0.0)
    }
    print(f"{'Init Method':<25} | {'Train Accuracy':<15} | {'Test Accuracy':<15}")
    print("-" * 61)
    for name, init_fn in initializations.items():
        np.random.seed(42)
        model_exp = Perceptron(input_size=2)
        model_exp.w, model_exp.b = init_fn()
        model_exp.fit(X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, lr=0.1, batch_size=32)
        tr_acc = accuracy_score(y_train.reshape(-1, 1), model_exp.predict(X_train_scaled))
        te_acc = accuracy_score(y_test.reshape(-1, 1), model_exp.predict(X_test_scaled))
        print(f"{name:<25} | {tr_acc:<15.4f} | {te_acc:<15.4f}")


def run_bonus_part():
    print("\n" + "=" * 50)
    print("=== ЧАСТЬ 2: ДОПОЛНИТЕЛЬНЫЕ ЗАДАНИЯ ===")
    print("=" * 50)

    gen = CustomDataGenerator()

    # 1. Тестирование нелинейной разделимости
    print("\n[Бонус 1] Тестирование нелинейных геометрических паттернов...")
    scenarios = {
        "Линейные Гауссовы Облака": gen.generate_linear_data(noise_prob=0.05),
        "XOR": gen.generate_xor_data(),
        "Окружность": gen.generate_circle_data()
    }
    for name, (X_raw, y_raw) in scenarios.items():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        model = Perceptron(input_size=2)
        model.fit(X_scaled, y_raw, X_scaled, y_raw, epochs=100, lr=0.1, batch_size=32)
        plot_decision_boundary(X_scaled, y_raw, model, title=f"Результаты на: {name}")

    # 2. Сравнение BCE и Hinge Loss
    print("\n[Бонус 2.А] Сравнение BCE и Hinge Loss на линейных данных...")
    X_lin, y_lin = gen.generate_linear_data(n_samples=500, noise_prob=0.0)
    X_train, X_test, y_train, y_test = train_test_split_stratified(X_lin, y_lin, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_bce = Perceptron(input_size=2, loss_fn='bce')
    hist_bce = model_bce.fit(X_train_s, y_train, X_test_s, y_test, epochs=80, lr=0.05, batch_size=32)

    model_hinge = Perceptron(input_size=2, loss_fn='hinge')
    hist_hinge = model_hinge.fit(X_train_s, y_train, X_test_s, y_test, epochs=80, lr=0.05, batch_size=32)

    plt.figure(figsize=(10, 4))
    plt.plot(hist_bce['train_loss'], label='BCE Train Loss', color='blue')
    plt.plot(hist_hinge['train_loss'], label='Hinge Train Loss', color='red')
    plt.title("Скорость сходимости: BCE vs Hinge Loss")
    plt.xlabel("Эпохи")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. L2-регуляризация
    print("\n[Бонус 2.Б] Оценка влияния L2-регуляризации...")
    lambdas = [0.0, 0.001, 0.05, 0.5]
    for lam in lambdas:
        model_l2 = Perceptron(input_size=2, loss_fn='bce', l2_lambda=lam)
        model_l2.fit(X_train_s, y_train, X_test_s, y_test, epochs=80, lr=0.1, batch_size=32)
        w_norm = np.linalg.norm(model_l2.w)
        print(f"Коэффициент lambda = {lam:<5} | Норма вектора весов (||w||_2): {w_norm:.6f}")

    # 4. Расчет расширенных метрик и анализ ошибок
    print("\n[Бонус 3] Расчет расширенных метрик и визуализация ошибок...")
    np.random.seed(42)
    X_noisy, y_noisy = gen.generate_linear_data(noise_prob=0.15)
    X_tr, X_te, y_tr, y_te = train_test_split_stratified(X_noisy, y_noisy, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model_eval = Perceptron(input_size=2, loss_fn='bce')
    model_eval.fit(X_tr_s, y_tr, X_te_s, y_te, epochs=100, lr=0.1, batch_size=32)

    preds = model_eval.predict(X_te_s)
    probs = model_eval.forward(X_te_s)

    print(f"Precision: {precision_score(y_te, preds):.4f}")
    print(f"Recall:    {recall_score(y_te, preds):.4f}")
    print(f"F1-Score:  {f1_score(y_te, preds):.4f}")

    fpr, tpr = roc_curve_data(y_te, probs)
    auc_val = roc_auc_score(fpr, tpr)
    plot_roc_curve(fpr, tpr, auc_val)
    plot_misclassified(X_te_s, y_te, preds)

    # 5. Сравнение Momentum SGD
    print("\n[Бонус 4] Оценка оптимизатора Momentum SGD...")
    betas = [0.0, 0.5, 0.9, 0.99]
    plt.figure(figsize=(10, 5))
    for b in betas:
        np.random.seed(42)
        model_mom = Perceptron(input_size=2)
        hist_mom = model_mom.fit(X_tr_s, y_tr, X_te_s, y_te, epochs=100, lr=0.01, batch_size=32, momentum_beta=b)
        lbl = "Обычный SGD (beta=0)" if b == 0.0 else f"Momentum SGD (beta={b})"
        plt.plot(hist_mom['train_loss'], label=lbl)
    plt.title("Влияние коэффициента Momentum на скорость сходимости")
    plt.xlabel("Эпохи")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. Кросс-валидация и Grid Search
    print("\n[Бонус 5] Поиск оптимальных гиперпараметров по сетке (Grid Search)...")

    def k_fold_cross_validation(X, y, k=5, epochs=50, lr=0.1, batch_size=32):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        folds = np.array_split(indices, k)
        scores = []
        for fold_idx in range(k):
            val_indices = folds[fold_idx]
            train_indices = np.hstack([folds[i] for i in range(k) if i != fold_idx])
            X_tr_fold, y_tr_fold = X[train_indices], y[train_indices]
            X_val_fold, y_val_fold = X[val_indices], y[val_indices]

            scaler_fold = StandardScaler()
            X_tr_scaled = scaler_fold.fit_transform(X_tr_fold)
            X_val_scaled = scaler_fold.transform(X_val_fold)

            model_fold = Perceptron(input_size=X.shape[1])
            model_fold.fit(X_tr_scaled, y_tr_fold, X_val_scaled, y_val_fold, epochs=epochs, lr=lr,
                           batch_size=batch_size)
            preds_fold = model_fold.predict(X_val_scaled)
            scores.append(accuracy_score(y_val_fold.reshape(-1, 1), preds_fold))
        return np.mean(scores), np.std(scores)

    lrs = [0.005, 0.05, 0.2]
    batch_sizes = [16, 32, 64]

    best_score = -1.0
    best_params = {}

    print(f"{'lr':<10} | {'batch_size':<10} | {'Mean Acc':<12} | {'Std Dev':<10}")
    print("-" * 51)

    for lr in lrs:
        for bs in batch_sizes:
            mean_acc, std_acc = k_fold_cross_validation(X_noisy, y_noisy, k=5, epochs=50, lr=lr, batch_size=bs)
            print(f"{lr:<10} | {bs:<10} | {mean_acc:<12.4f} | {std_acc:<10.4f}")
            if mean_acc > best_score:
                best_score = mean_acc
                best_params = {"lr": lr, "batch_size": bs}

    print("\n" + "=" * 50)
    print(f"Лучшие параметры по результатам поиска: {best_params}")
    print(f"Точность на кросс-валидации: {best_score:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    # Фиксация глобального генератора случайных чисел для воспроизводимости всего сценария
    np.random.seed(42)

    # Запуск обязательной части работы
    run_mandatory_part()

    # Запуск части с бонусными задачами
    run_bonus_part()