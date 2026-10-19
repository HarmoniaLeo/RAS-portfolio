"""Toy example of the non-robust scoring problem (paper Sec. 3.2, Example 1) and of ARM (Sec. 4.2).

Five price patterns x1..x5 with constant price movements appear on three assets
(A, B, C). A linear scoring policy z = phi^T e(x) is trained by deterministic
policy gradient on decision steps t1, t2 and tested on t3, t4.

    python scripts/toy_example.py --out outputs/toy

Prints the final scores, test allocations a and returns R, and saves the
score-versus-epoch curves used in the two toy-example figures (without ARM and
with ARM, where masked epochs are shaded).
"""
import argparse
import colorsys
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PATTERNS = ["x1", "x2", "x3", "x4", "x5"]
MOVEMENT = {"x1": 0.6, "x2": 0.9, "x4": 1.2, "x3": 1.5, "x5": 1.8}
STEPS = {"t1": ["x1", "x2", "x3"], "t2": ["x4", "x5", "x2"],   # training steps (assets A, B, C)
         "t3": ["x2", "x1", "x4"], "t4": ["x3", "x5", "x1"]}   # testing steps
TRAIN, TEST = ("t1", "t2"), ("t3", "t4")
D, ALPHA, EPOCHS, M = 8, 1.0, 50, 3
Y = 1e9  # masking constant


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()


def colors(n, sat=0.6, light=0.55):
    return [colorsys.hls_to_rgb(h, light, sat) for h in np.linspace(0, 1, n, endpoint=False)]


class ToyPolicy:
    def __init__(self, embeddings, phi):
        self.E, self.phi = embeddings, phi

    def scores(self, step):
        X = np.vstack([self.E[PATTERNS.index(p)] for p in STEPS[step]])
        return X.dot(self.phi), X

    def update(self, step, mask=None, update_embeddings=False):
        """One DPG step on the reward log R_t (no transaction cost)."""
        z, X = self.scores(step)
        if mask is not None:
            z = z.copy()
            z[mask] = -Y
        a = softmax(z)
        r = np.array([MOVEMENT[p] for p in STEPS[step]])
        r_bar = a.dot(r)
        dz = a * (r - r_bar) / (r_bar + 1.0)  # Proposition 1: dR/dz_i ∝ a_i (r_i - r_bar)
        self.phi += ALPHA * X.T.dot(dz)
        if update_embeddings:
            grad = np.zeros_like(self.E)
            for j, p in enumerate(STEPS[step]):
                grad[PATTERNS.index(p)] += self.phi * dz[j]
            self.E += ALPHA * grad


def train(policy, rng, arm):
    """Return the score history {(step, asset): [z per epoch]} and the masked epochs."""
    history, masked = {}, {}
    for epoch in range(1, EPOCHS + 1):
        for step in TRAIN:
            j_mask = rng.randint(M)  # ARM masks one random asset per step
            if arm:
                z, _ = policy.scores(step)
                for j in range(M):
                    history.setdefault((step, j), []).append(z[j])
                    masked.setdefault((step, j), []).append(j == j_mask)
            policy.update(step, mask=j_mask if arm else None, update_embeddings=arm)
        for step in (TEST if arm else TRAIN + TEST):
            z, _ = policy.scores(step)
            for j in range(M):
                history.setdefault((step, j), []).append(z[j])
                masked.setdefault((step, j), []).append(False)
    return history, masked


def report(title, policy):
    print(f"== {title}")
    for step in TRAIN + TEST:
        z, _ = policy.scores(step)
        line = ", ".join(f"{p}: z={zi:.3f}" for p, zi in zip(STEPS[step], z))
        if step in TEST:
            a = softmax(z)
            R = a.dot([MOVEMENT[p] for p in STEPS[step]])
            line += " | a = (" + ", ".join(f"{ai:.3f}" for ai in a) + f") | R = {R:.3f}"
        print(f"  {step}: {line}")


def plot(history, masked, path):
    palette = dict(zip(PATTERNS, colors(len(PATTERNS))))
    keys = sorted(history)
    values = np.concatenate([history[k] for k in keys])
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 1.3 * len(keys)), sharex=True)
    for ax, (step, j) in zip(axes, keys):
        pattern = STEPS[step][j]
        epochs = np.arange(1, len(history[(step, j)]) + 1)
        ax.plot(epochs, history[(step, j)], color=palette[pattern], linewidth=2)
        for e, is_masked in zip(epochs, masked[(step, j)]):
            if is_masked:
                ax.axvspan(e - 0.5, e + 0.5, color="lightgray", alpha=0.5)
        ax.set_ylim(values.min() - 0.05, values.max() + 0.05)
        ax.set_ylabel("z")
        ax.set_title(f"{step}, asset {'ABC'[j]}: {pattern}", fontsize=8, loc="left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/toy")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rng = np.random  # a single random stream (seed 42), as in the paper's notebook
    rng.seed(42)
    embeddings = 0.1 * rng.randn(len(PATTERNS), D)  # fixed pattern embeddings e(x)

    # Example 1: plain training, only phi is learned
    policy = ToyPolicy(embeddings.copy(), 0.1 * rng.randn(D))
    history, masked = train(policy, rng, arm=False)
    report("Without ARM (Fig. 'theory')", policy)
    plot({k: v for k, v in history.items() if k[0] in TRAIN}, masked, os.path.join(args.out, "toy_without_arm.pdf"))

    # ARM: one random asset is masked at every training step
    policy = ToyPolicy(embeddings, 0.1 * rng.randn(D))
    history, masked = train(policy, rng, arm=True)
    report("With ARM (Fig. 'theory-masking'), scores after the last update", policy)
    print("  last value of each score curve (as shown in the figure):")
    for (step, j), values in sorted(history.items()):
        print(f"    {step} asset {'ABC'[j]} {STEPS[step][j]}: z={values[-1]:.4f}")
    plot(history, masked, os.path.join(args.out, "toy_with_arm.pdf"))
    print(f"Figures written to {args.out}")


if __name__ == "__main__":
    main()
