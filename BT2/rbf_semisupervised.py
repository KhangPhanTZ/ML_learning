"""
RBF Network + GA/PSO cho bài toán Semi-Supervised Learning (3 lớp).

Pipeline:
    1. DataProcessor   : đọc CSV, tách labeled / unlabeled.
    2. RBFLayer        : K-Means trên toàn bộ X -> tâm μ; ước lượng σ; biến đổi Gaussian.
    3. EvolutionaryOptimizer (PSO / GA) : tối ưu W (K x 3) bằng cross-entropy
       trên labeled_data — KHÔNG dùng pseudo-inverse.
    4. RBFNetwork      : orchestrator (fit / predict / predict_proba).

Toàn bộ tính toán dạng ma trận bằng NumPy, vector hoá triệt để.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# 1. DATA
# ---------------------------------------------------------------------------
class DataProcessor:
    """Đọc CSV và tách dữ liệu thành (labeled, unlabeled). Nhãn -1 = chưa gán."""

    def __init__(self, csv_path: str, n_classes: int = 3):
        self.csv_path = csv_path
        self.n_classes = n_classes
        self.df = pd.read_csv(csv_path)

        # Toàn bộ feature ma trận X (n, 2) và nhãn y (n,)
        self.X = self.df[["x1", "x2"]].to_numpy(dtype=float)
        self.y = self.df["label"].to_numpy(dtype=int)

        # Mask phân tách
        self.labeled_mask = self.y != -1
        self.unlabeled_mask = self.y == -1

        self.X_labeled = self.X[self.labeled_mask]
        self.y_labeled = self.y[self.labeled_mask]
        self.X_unlabeled = self.X[self.unlabeled_mask]

    def one_hot(self, y: np.ndarray) -> np.ndarray:
        # Y[i, c] = 1 nếu y[i] == c, ngược lại 0  ->  shape (n, n_classes)
        Y = np.zeros((len(y), self.n_classes), dtype=float)
        Y[np.arange(len(y)), y] = 1.0
        return Y

    def summary(self) -> dict:
        unique, counts = np.unique(self.y, return_counts=True)
        return {
            "n_total": len(self.y),
            "n_labeled": int(self.labeled_mask.sum()),
            "n_unlabeled": int(self.unlabeled_mask.sum()),
            "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
        }


# ---------------------------------------------------------------------------
# 2. RBF HIDDEN LAYER
# ---------------------------------------------------------------------------
class RBFLayer:
    """Lớp ẩn RBF: K-Means tìm μ, heuristic σ, biến đổi Gaussian."""

    def __init__(self, n_centers: int = 15, random_state: int = 42,
                 sigma_strategy: str = "dmax"):
        self.n_centers = n_centers
        self.random_state = random_state
        self.sigma_strategy = sigma_strategy  # "dmax" | "mean"
        self.centers: np.ndarray | None = None  # (K, d)
        self.sigma: float | np.ndarray | None = None

    # --- Stage 1: clustering trên TOÀN BỘ dữ liệu (cả -1) ----------------
    def fit(self, X_all: np.ndarray) -> "RBFLayer":
        km = KMeans(
            n_clusters=self.n_centers,
            n_init=10,
            random_state=self.random_state,
        )
        km.fit(X_all)
        self.centers = km.cluster_centers_  # μ_k, k = 1..K

        # Khoảng cách đôi giữa các tâm: pairwise[i, j] = ||μ_i - μ_j||
        diffs = self.centers[:, None, :] - self.centers[None, :, :]
        pairwise = np.sqrt(np.sum(diffs ** 2, axis=-1))

        if self.sigma_strategy == "dmax":
            # Heuristic chuẩn: σ = d_max / sqrt(2K) — đảm bảo các Gaussian phủ lẫn nhau
            d_max = pairwise.max()
            self.sigma = d_max / np.sqrt(2.0 * self.n_centers)
        else:
            # Trung bình khoảng cách tới P láng giềng gần nhất (P=2 ngoại trừ chính nó)
            np.fill_diagonal(pairwise, np.inf)
            self.sigma = pairwise.min(axis=1).mean()
        return self

    # --- Biến đổi Gaussian ---------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centers is None:
            raise RuntimeError("RBFLayer chưa được fit.")
        # Khoảng cách bình phương: dist_sq[i, k] = ||x_i - μ_k||²  -> (n, K)
        diffs = X[:, None, :] - self.centers[None, :, :]
        dist_sq = np.sum(diffs ** 2, axis=-1)
        # Gaussian: φ_k(x) = exp( -||x-μ_k||² / (2 σ²) )
        H = np.exp(-dist_sq / (2.0 * self.sigma ** 2))
        return H


# ---------------------------------------------------------------------------
# 3. EVOLUTIONARY OPTIMIZERS (PSO + GA)  — bắt buộc, KHÔNG dùng pseudo-inverse
# ---------------------------------------------------------------------------
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)  # ổn định số
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _cross_entropy_fitness(W_flat: np.ndarray, H: np.ndarray,
                           Y_onehot: np.ndarray, shape: tuple) -> float:
    """Loss = - (1/N) Σ_i Σ_c y_ic * log p_ic  với p = softmax(H W)."""
    W = W_flat.reshape(shape)              # (K, C)
    logits = H @ W                         # (N, C)
    probs = _softmax(logits)               # (N, C)
    eps = 1e-12
    ce = -np.mean(np.sum(Y_onehot * np.log(probs + eps), axis=1))
    return float(ce)


class PSOOptimizer:
    """Particle Swarm Optimization viết từ con số 0 (chỉ dùng NumPy)."""

    def __init__(self, n_particles: int = 40, n_iter: int = 300,
                 w: float = 0.72, c1: float = 1.49, c2: float = 1.49,
                 v_clip: float = 1.0, init_scale: float = 0.5, seed: int = 42):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w           # quán tính
        self.c1 = c1         # cognitive
        self.c2 = c2         # social
        self.v_clip = v_clip
        self.init_scale = init_scale
        self.rng = np.random.default_rng(seed)
        self.history: list[float] = []

    def optimize(self, H: np.ndarray, Y_onehot: np.ndarray,
                 n_classes: int = 3) -> tuple[np.ndarray, float]:
        K = H.shape[1]
        shape = (K, n_classes)
        dim = K * n_classes

        # Khởi tạo vị trí (W) và vận tốc cho mỗi particle
        positions = self.rng.normal(0.0, self.init_scale, (self.n_particles, dim))
        velocities = self.rng.normal(0.0, 0.1, (self.n_particles, dim))

        # pbest = vị trí tốt nhất của từng particle; gbest = tốt nhất toàn bầy
        scores = np.array([_cross_entropy_fitness(p, H, Y_onehot, shape)
                           for p in positions])
        pbest = positions.copy()
        pbest_scores = scores.copy()
        gbest_idx = int(np.argmin(pbest_scores))
        gbest = pbest[gbest_idx].copy()
        gbest_score = float(pbest_scores[gbest_idx])

        for _ in range(self.n_iter):
            # Số ngẫu nhiên r1, r2 (uniform 0..1) cho mỗi chiều của mỗi particle
            r1 = self.rng.random((self.n_particles, dim))
            r2 = self.rng.random((self.n_particles, dim))
            # Cập nhật vận tốc:  v ← w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)
            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest - positions)
                + self.c2 * r2 * (gbest - positions)
            )
            # Clip để tránh phân kỳ
            np.clip(velocities, -self.v_clip, self.v_clip, out=velocities)
            # Cập nhật vị trí:  x ← x + v
            positions = positions + velocities

            # Đánh giá fitness cho mọi particle
            scores = np.array([_cross_entropy_fitness(p, H, Y_onehot, shape)
                               for p in positions])
            # Cập nhật pbest
            improved = scores < pbest_scores
            pbest[improved] = positions[improved]
            pbest_scores[improved] = scores[improved]
            # Cập nhật gbest
            best_idx = int(np.argmin(pbest_scores))
            if pbest_scores[best_idx] < gbest_score:
                gbest = pbest[best_idx].copy()
                gbest_score = float(pbest_scores[best_idx])

            self.history.append(gbest_score)

        return gbest.reshape(shape), gbest_score


class GAOptimizer:
    """Genetic Algorithm thuần NumPy: tournament + arithmetic crossover + Gaussian mutation."""

    def __init__(self, pop_size: int = 60, n_iter: int = 300,
                 mutation_rate: float = 0.15, mutation_scale: float = 0.25,
                 elite_frac: float = 0.1, tournament_k: int = 3,
                 init_scale: float = 0.5, seed: int = 42):
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.elite_frac = elite_frac
        self.tournament_k = tournament_k
        self.init_scale = init_scale
        self.rng = np.random.default_rng(seed)
        self.history: list[float] = []

    def _tournament(self, pop: np.ndarray, scores: np.ndarray) -> np.ndarray:
        idx = self.rng.integers(0, len(pop), self.tournament_k)
        winner = idx[int(np.argmin(scores[idx]))]   # min vì fitness = loss
        return pop[winner]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Arithmetic / blended crossover: child = α*p1 + (1-α)*p2  với α ~ U(0,1) per gene
        alpha = self.rng.random(p1.shape)
        c1 = alpha * p1 + (1.0 - alpha) * p2
        c2 = alpha * p2 + (1.0 - alpha) * p1
        return c1, c2

    def _mutate(self, ind: np.ndarray) -> np.ndarray:
        mask = self.rng.random(ind.shape) < self.mutation_rate
        noise = self.rng.normal(0.0, self.mutation_scale, ind.shape)
        return ind + mask * noise   # đột biến cộng nhiễu Gaussian theo gene

    def optimize(self, H: np.ndarray, Y_onehot: np.ndarray,
                 n_classes: int = 3) -> tuple[np.ndarray, float]:
        K = H.shape[1]
        shape = (K, n_classes)
        dim = K * n_classes

        # Quần thể khởi tạo
        pop = self.rng.normal(0.0, self.init_scale, (self.pop_size, dim))
        scores = np.array([_cross_entropy_fitness(p, H, Y_onehot, shape) for p in pop])

        n_elite = max(1, int(self.elite_frac * self.pop_size))

        for _ in range(self.n_iter):
            # Sắp xếp tăng dần theo loss (cá thể tốt đứng trước)
            order = np.argsort(scores)
            pop, scores = pop[order], scores[order]

            # Tinh hoa (elitism): giữ nguyên n_elite cá thể tốt nhất
            new_pop = [pop[i].copy() for i in range(n_elite)]

            # Sinh phần còn lại bằng tournament + crossover + mutation
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, scores)
                p2 = self._tournament(pop, scores)
                c1, c2 = self._crossover(p1, p2)
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))

            pop = np.asarray(new_pop)
            scores = np.array([_cross_entropy_fitness(p, H, Y_onehot, shape) for p in pop])
            self.history.append(float(scores.min()))

        best_idx = int(np.argmin(scores))
        return pop[best_idx].reshape(shape), float(scores[best_idx])


# ---------------------------------------------------------------------------
# 4. RBF NETWORK ORCHESTRATOR
# ---------------------------------------------------------------------------
class RBFNetwork:
    """Mạng RBF hoàn chỉnh (Gaussian hidden + softmax output) tối ưu bằng PSO/GA."""

    def __init__(self, n_centers: int = 15, n_classes: int = 3,
                 optimizer: str = "pso", random_state: int = 42,
                 sigma_strategy: str = "dmax", **opt_kwargs):
        self.n_centers = n_centers
        self.n_classes = n_classes
        self.optimizer_name = optimizer.lower()
        self.rbf = RBFLayer(n_centers=n_centers, random_state=random_state,
                            sigma_strategy=sigma_strategy)

        if self.optimizer_name == "pso":
            self.optimizer = PSOOptimizer(seed=random_state, **opt_kwargs)
        elif self.optimizer_name == "ga":
            self.optimizer = GAOptimizer(seed=random_state, **opt_kwargs)
        else:
            raise ValueError("optimizer phải là 'pso' hoặc 'ga'.")

        self.W: np.ndarray | None = None
        self.best_loss: float | None = None

    # X_all: gồm cả labeled lẫn unlabeled (cho clustering)
    # X_labeled, y_labeled: chỉ dữ liệu đã có nhãn (cho fitness)
    def fit(self, X_all: np.ndarray, X_labeled: np.ndarray,
            y_labeled: np.ndarray) -> "RBFNetwork":
        self.rbf.fit(X_all)
        H_lab = self.rbf.transform(X_labeled)
        Y = np.zeros((len(y_labeled), self.n_classes))
        Y[np.arange(len(y_labeled)), y_labeled] = 1.0
        self.W, self.best_loss = self.optimizer.optimize(H_lab, Y, self.n_classes)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        H = self.rbf.transform(X)
        return _softmax(H @ self.W)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# 5. ENTRY POINT (chạy thẳng từ terminal)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    HERE = os.path.dirname(os.path.abspath(__file__))
    CSV = os.path.join(HERE, "semi_supervised_data.csv")

    dp = DataProcessor(CSV)
    print("Tổng quan dữ liệu:", dp.summary())

    net = RBFNetwork(n_centers=15, optimizer="pso", n_iter=300, n_particles=40)
    net.fit(dp.X, dp.X_labeled, dp.y_labeled)
    print(f"Best loss (cross-entropy) trên labeled = {net.best_loss:.4f}")

    # ===== (b) Giữ nguyên class gốc, chỉ gán nhãn cho điểm -1 ===============
    pred_unlab = net.predict(dp.X_unlabeled)
    y_filled = dp.y.copy()
    y_filled[dp.unlabeled_mask] = pred_unlab

    # ===== (a) Gán nhãn lại TOÀN BỘ dữ liệu sau clustering ==================
    y_relabel_all = net.predict(dp.X)

    # Lưu kết quả ra CSV
    out_b = dp.df.copy()
    out_b["label"] = y_filled
    out_b.to_csv(os.path.join(HERE, "result_keep_original.csv"), index=False)

    out_a = dp.df.copy()
    out_a["label"] = y_relabel_all
    out_a.to_csv(os.path.join(HERE, "result_relabel_all.csv"), index=False)

    print("Đã ghi result_keep_original.csv và result_relabel_all.csv")
