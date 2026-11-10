"""Projeto de ML: MLP binaria multicamadas com backpropagation em NumPy.

O modelo, o gradiente e o descenso do gradiente usam apenas NumPy.
Matplotlib e usado exclusivamente para gerar os graficos do relatorio.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def stratified_split(X: np.ndarray, y: np.ndarray, seed: int = 42):
    """Divide em 60/20/20 mantendo aproximadamente a proporcao de classes."""
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for cls in np.unique(y):
        idx = np.flatnonzero(y.ravel() == cls)
        rng.shuffle(idx)
        n_train = int(round(0.60 * len(idx)))
        n_val = int(round(0.20 * len(idx)))
        parts[0].extend(idx[:n_train])
        parts[1].extend(idx[n_train:n_train + n_val])
        parts[2].extend(idx[n_train + n_val:])
    result = []
    for idx in parts:
        idx = np.asarray(idx, dtype=int)
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))
    return tuple(result)


@dataclass
class Standardizer:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.scale_ = X.std(axis=0, keepdims=True)
        self.scale_[self.scale_ < EPS] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("O padronizador ainda nao foi ajustado.")
        return (X - self.mean_) / self.scale_


class MLPBinaryClassifier:
    """MLP 2 -> camadas ocultas -> 1, tanh/sigmoid e Xavier/Glorot."""

    def __init__(self, n_features: int, hidden_layers=(8, 4), seed: int = 42,
                 learning_rate: float = 0.08, l2: float = 1e-3):
        hidden_layers = tuple(int(n) for n in hidden_layers)
        if len(hidden_layers) < 2:
            raise ValueError("O enunciado exige pelo menos duas camadas ocultas.")
        if any(n <= 0 for n in hidden_layers):
            raise ValueError("Cada camada deve ter ao menos um neuronio.")
        self.hidden_layers = hidden_layers
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        rng = np.random.default_rng(seed)
        sizes = (int(n_features),) + hidden_layers + (1,)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights.append(rng.uniform(-limit, limit, (fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    def forward(self, X: np.ndarray):
        activations, z_values = [X], []
        a = X
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ W + b
            a = np.tanh(z)
            z_values.append(z)
            activations.append(a)
        z = a @ self.weights[-1] + self.biases[-1]
        p = sigmoid(z)
        z_values.append(z)
        activations.append(p)
        return p, (activations, z_values)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        p, _ = self.forward(X)
        reg = 0.5 * self.l2 * sum(float(np.sum(W ** 2)) for W in self.weights)
        return binary_cross_entropy(y, p) + reg

    def gradients(self, X: np.ndarray, y: np.ndarray):
        p, (activations, _) = self.forward(X)
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)
        delta = (p - y) / X.shape[0]
        for layer in range(len(self.weights) - 1, -1, -1):
            dW[layer] = activations[layer].T @ delta + self.l2 * self.weights[layer]
            db[layer] = delta.sum(axis=0, keepdims=True)
            if layer > 0:
                delta = (delta @ self.weights[layer].T) * (1.0 - activations[layer] ** 2)
        return dW, db

    def _step(self, grads):
        dW, db = grads
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 6000, patience: int = 800):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_loss, best_epoch, wait = np.inf, 0, 0
        best = self.get_parameters()
        for epoch in range(epochs):
            self._step(self.gradients(X, y))
            train_loss, val_loss = self.loss(X, y), self.loss(X_val, y_val)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(self.accuracy(X, y))
            history["val_acc"].append(self.accuracy(X_val, y_val))
            if val_loss < best_loss - 1e-8:
                best_loss, best_epoch, wait = val_loss, epoch, 0
                best = self.get_parameters()
            else:
                wait += 1
            if wait >= patience:
                break
        self.set_parameters(best)
        history["best_epoch"] = best_epoch
        history["epochs_run"] = len(history["train_loss"])
        return history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def get_parameters(self):
        return ([W.copy() for W in self.weights], [b.copy() for b in self.biases])

    def set_parameters(self, params):
        self.weights = [W.copy() for W in params[0]]
        self.biases = [b.copy() for b in params[1]]

    def parameter_count(self) -> int:
        return int(sum(a.size for a in self.weights + self.biases))


def gradient_check(model: MLPBinaryClassifier, X: np.ndarray, y: np.ndarray,
                   epsilon: float = 1e-5):
    """Compara todos os gradientes com diferencas centrais finitas."""
    analytic_w, analytic_b = model.gradients(X, y)
    analytic_parts, numeric_parts = [], []
    for params, grads in ((model.weights, analytic_w), (model.biases, analytic_b)):
        for param, grad in zip(params, grads):
            numeric = np.zeros_like(param)
            for idx in np.ndindex(param.shape):
                old = param[idx]
                param[idx] = old + epsilon
                plus = model.loss(X, y)
                param[idx] = old - epsilon
                minus = model.loss(X, y)
                param[idx] = old
                numeric[idx] = (plus - minus) / (2.0 * epsilon)
            analytic_parts.append(grad.ravel())
            numeric_parts.append(numeric.ravel())
    ga = np.concatenate(analytic_parts)
    gn = np.concatenate(numeric_parts)
    relative_error = float(np.linalg.norm(ga - gn) / (np.linalg.norm(ga) + np.linalg.norm(gn) + EPS))
    coordinate_error = float(np.max(np.abs(ga - gn) / (np.abs(ga) + np.abs(gn) + EPS)))
    return {"relative_error": relative_error, "max_coordinate_error": coordinate_error,
            "checked_parameters": int(ga.size)}


def confusion_metrics(y: np.ndarray, pred: np.ndarray):
    yt, yp = y.ravel(), pred.ravel()
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, EPS)
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "accuracy": (tp + tn) / len(yt), "precision": precision,
            "recall": recall, "f1": f1}


def fit_linear_logistic(X: np.ndarray, y: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                        learning_rate: float = 0.1, epochs: int = 5000):
    """Baseline linear treinada por descenso do gradiente."""
    w = np.zeros((X.shape[1], 1))
    b = 0.0
    best_loss, best = np.inf, None
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        e = p - y
        w -= learning_rate * (X.T @ e / len(y))
        b -= learning_rate * float(e.mean())
        val_loss = binary_cross_entropy(y_val, sigmoid(X_val @ w + b))
        if val_loss < best_loss:
            best_loss, best = val_loss, (w.copy(), b)
    return best


def load_dataset(path: str | Path):
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("classification2.txt deve ter tres colunas: x1, x2 e classe.")
    return data[:, :2].astype(float), data[:, 2:3].astype(int)


def train_backprop(data_path: str | Path, out_dir: str | Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_dataset(data_path)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = stratified_split(X, y, seed=42)
    scaler = Standardizer().fit(Xtr)
    Xtr_s, Xv_s, Xte_s = map(scaler.transform, (Xtr, Xv, Xte))

    probe = MLPBinaryClassifier(2, (5, 4), seed=1, learning_rate=0.05, l2=0.0)
    grad_check = gradient_check(probe, Xtr_s[:8], ytr[:8])
    configs = [
        {"layers": [4, 4], "lr": 0.03, "l2": 1e-3},
        {"layers": [4, 4], "lr": 0.10, "l2": 1e-3},
        {"layers": [8, 4], "lr": 0.03, "l2": 1e-3},
        {"layers": [8, 4], "lr": 0.10, "l2": 1e-3},
        {"layers": [8, 8, 4], "lr": 0.03, "l2": 1e-3},
        {"layers": [8, 8, 4], "lr": 0.10, "l2": 1e-3},
        {"layers": [12, 8], "lr": 0.10, "l2": 1e-3},
        {"layers": [12, 8, 4], "lr": 0.10, "l2": 1e-3},
    ]
    trials = []
    best = None
    for cfg in configs:
        model = MLPBinaryClassifier(2, tuple(cfg["layers"]), seed=101,
                                    learning_rate=cfg["lr"], l2=cfg["l2"])
        hist = model.fit(Xtr_s, ytr, Xv_s, yv, epochs=6000, patience=800)
        row = {**cfg, "hidden_layer_count": len(cfg["layers"]),
               "parameters": model.parameter_count(), "best_epoch": hist["best_epoch"],
               "epochs_run": hist["epochs_run"], "train_acc": model.accuracy(Xtr_s, ytr),
               "val_acc": model.accuracy(Xv_s, yv), "val_loss": model.loss(Xv_s, yv)}
        trials.append(row)
        key = (row["val_acc"], -row["val_loss"], -row["parameters"])
        if best is None or key > best[0]:
            best = (key, model, hist, cfg)

    _, model, history, cfg = best
    metrics = confusion_metrics(yte, model.predict(Xte_s))
    linear_w, linear_b = fit_linear_logistic(Xtr_s, ytr, Xv_s, yv)
    linear_pred = (sigmoid(Xte_s @ linear_w + linear_b) >= .5).astype(int)
    linear_metrics = confusion_metrics(yte, linear_pred)
    metrics.update({"train_accuracy": model.accuracy(Xtr_s, ytr),
                    "validation_accuracy": model.accuracy(Xv_s, yv),
                    "test_loss": model.loss(Xte_s, yte),
                    "gradient_check": grad_check,
                    "selected_config": cfg,
                    "selected_parameter_count": model.parameter_count(),
                    "best_epoch": history["best_epoch"],
                    "epochs_run": history["epochs_run"],
                    "split_sizes": {"train": len(ytr), "validation": len(yv), "test": len(yte)},
                    "class_counts": {"class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1))},
                    "linear_logistic_test_accuracy": linear_metrics["accuracy"]})

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history["train_loss"], label="treino")
    ax[0].plot(history["val_loss"], label="validacao")
    ax[0].axvline(history["best_epoch"], color="0.4", ls="--", lw=1)
    ax[0].set(xlabel="epoca", ylabel="entropia cruzada + L2", title="Funcao de custo")
    ax[1].plot(history["train_acc"], label="treino")
    ax[1].plot(history["val_acc"], label="validacao")
    ax[1].set(xlabel="epoca", ylabel="acuracia", ylim=(0.4, 1.02), title="Acuracia")
    for axis in ax:
        axis.grid(alpha=.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "backprop_curvas.png", dpi=180)
    plt.close(fig)

    x1 = np.linspace(X[:, 0].min() - .08, X[:, 0].max() + .08, 300)
    x2 = np.linspace(X[:, 1].min() - .08, X[:, 1].max() + .08, 300)
    xx, yy = np.meshgrid(x1, x2)
    grid = np.c_[xx.ravel(), yy.ravel()]
    pp = model.predict_proba(scaler.transform(grid)).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(6.1, 5.1))
    ax.contourf(xx, yy, pp, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=.32)
    ax.contour(xx, yy, pp, levels=[.5], colors="black", linewidths=2)
    ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c="#b2182b", marker="+", s=48, label="classe 1")
    ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], facecolors="none", edgecolors="#2166ac", s=32, label="classe 0")
    ax.scatter(Xte[:, 0], Xte[:, 1], facecolors="none", edgecolors="gold", s=85, lw=1.5, label="teste")
    ax.set(xlabel="x1", ylabel="x2", title="Fronteira nao linear aprendida")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(out_dir / "backprop_fronteira.png", dpi=180)
    plt.close(fig)

    payload = {"metrics": metrics, "trials": trials}
    (out_dir / "backprop_resultados.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_logistic_exercise(out_dir: str | Path, seed: int = 42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    m = 200
    xs = np.hstack((rng.normal(.9, .3, m // 2), rng.normal(-.4, .5, m // 2)))
    ys = np.hstack((np.ones(m // 2), np.zeros(m // 2)))

    def hypothesis(x, theta): return sigmoid(theta[0] + theta[1] * x)
    def cost(theta): return binary_cross_entropy(ys, hypothesis(xs, theta))
    def gradient(theta):
        error = hypothesis(xs, theta) - ys
        return np.array([error.mean(), np.mean(error * xs)])
    def accuracy(theta): return float(np.mean((hypothesis(xs, theta) >= .5) == ys))
    def boundary(theta): return float(-theta[0] / theta[1]) if abs(theta[1]) > EPS else np.nan

    rate_runs = {}
    for alpha in (0.01, 0.1, 1.0, 10.0):
        theta = np.zeros(2)
        costs, accs, bounds = [], [], []
        for _ in range(2000):
            theta -= alpha * gradient(theta)
            costs.append(cost(theta)); accs.append(accuracy(theta)); bounds.append(boundary(theta))
        rate_runs[str(alpha)] = {"theta": theta.tolist(), "cost": costs[-1],
                                 "accuracy": accs[-1], "boundary": bounds[-1],
                                 "costs": costs, "accuracies": accs, "boundaries": bounds}
    main = rate_runs["0.1"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    ax[0].plot(main["costs"], color="#2166ac"); ax[0].set(xlabel="epoca", ylabel="J(theta)", title="Custo")
    ax[1].plot(main["accuracies"], color="#1b7837"); ax[1].set(xlabel="epoca", ylabel="acuracia", ylim=(.5, 1.01), title="Acuracia")
    ax[2].plot(main["boundaries"], color="#b2182b"); ax[2].set(xlabel="epoca", ylabel="x*", title="Fronteira")
    for axis in ax: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out_dir / "logistica_curvas.png", dpi=180); plt.close(fig)

    theta = np.asarray(main["theta"])
    fig, ax = plt.subplots(figsize=(6.1, 3.6))
    ax.scatter(xs[ys == 0], ys[ys == 0], facecolors="none", edgecolors="#2166ac", label="classe 0")
    ax.scatter(xs[ys == 1], ys[ys == 1], c="#b2182b", marker="+", label="classe 1")
    ax.axvline(boundary(theta), color="black", ls="--", label=f"fronteira = {boundary(theta):.3f}")
    ax.set(xlabel="x", ylabel="classe", yticks=[0, 1], title="Regressao logistica unidimensional")
    ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "logistica_fronteira.png", dpi=180); plt.close(fig)

    summary = {k: {q: v[q] for q in ("theta", "cost", "accuracy", "boundary")}
               for k, v in rate_runs.items()}
    (out_dir / "logistica_resultados.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    results = base / "resultados"
    bp = train_backprop(base / "classification2.txt", results)
    lr = train_logistic_exercise(results)
    print(json.dumps({"backprop": bp, "logistic": lr}, indent=2))
