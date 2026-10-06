"""
Train a student performance prediction model.

This script creates a practice academic dataset, trains a linear regression
model with pure Python (no extra ML libraries), and saves weights as JSON.

Run from the project folder:
    python model/train_model.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

FEATURE_NAMES = [
    "attendance",
    "study_hours",
    "previous_gpa",
    "assignment_score",
    "internal_marks",
    "sleep_hours",
    "extra_activities",
    "midterm_score",
]

# Used to scale each input so training stays stable.
FEATURE_SCALE = {
    "attendance": 100.0,
    "study_hours": 10.0,
    "previous_gpa": 10.0,
    "assignment_score": 100.0,
    "internal_marks": 100.0,
    "sleep_hours": 8.0,
    "extra_activities": 10.0,
    "midterm_score": 100.0,
}


def _true_score(row: dict, noise: float = 0.0) -> float:
    score = (
        0.22 * row["attendance"]
        + 12.0 * (row["study_hours"] / 10.0)
        + 16.0 * (row["previous_gpa"] / 10.0)
        + 0.14 * row["assignment_score"]
        + 0.16 * row["internal_marks"]
        + 5.0 * min(max(row["sleep_hours"] / 8.0, 0.4), 1.15)
        + 4.0 * min(row["extra_activities"] / 8.0, 1.1)
        + 0.11 * row["midterm_score"]
        + noise
    )
    if row["sleep_hours"] < 5:
        score -= 4
    if row["attendance"] < 60:
        score -= 6
    return min(max(score, 20.0), 99.0)


def generate_dataset(n_samples: int = 900, seed: int = 42) -> list[tuple[list[float], float]]:
    rng = random.Random(seed)
    data = []
    for _ in range(n_samples):
        row = {
            "attendance": rng.uniform(40, 100),
            "study_hours": rng.uniform(0.5, 10),
            "previous_gpa": rng.uniform(4.0, 10.0),
            "assignment_score": rng.uniform(30, 100),
            "internal_marks": rng.uniform(30, 100),
            "sleep_hours": rng.uniform(3.5, 9.5),
            "extra_activities": float(rng.randint(0, 10)),
            "midterm_score": rng.uniform(25, 100),
        }
        y = _true_score(row, noise=rng.gauss(0, 2.8))
        x = [row[name] / FEATURE_SCALE[name] for name in FEATURE_NAMES]
        data.append((x, y))
    return data


def train_linear_regression(data, epochs: int = 2500, lr: float = 0.08):
    weights = [0.0] * len(FEATURE_NAMES)
    bias = 50.0
    n = len(data)

    for _ in range(epochs):
        grad_w = [0.0] * len(FEATURE_NAMES)
        grad_b = 0.0
        for x, y in data:
            pred = bias + sum(w * v for w, v in zip(weights, x))
            error = pred - y
            for i, value in enumerate(x):
                grad_w[i] += error * value
            grad_b += error
        for i in range(len(weights)):
            weights[i] -= lr * (grad_w[i] / n)
        bias -= lr * (grad_b / n)
    return weights, bias


def evaluate(data, weights, bias) -> tuple[float, float]:
    errors = []
    ss_res = 0.0
    mean_y = sum(y for _, y in data) / len(data)
    ss_tot = 0.0
    for x, y in data:
        pred = bias + sum(w * v for w, v in zip(weights, x))
        errors.append(abs(pred - y))
        ss_res += (pred - y) ** 2
        ss_tot += (y - mean_y) ** 2
    mae = sum(errors) / len(errors)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return mae, r2


def predict_from_payload(payload: dict, values: dict) -> float:
    x = [values[name] / payload["scales"][name] for name in payload["features"]]
    pred = payload["bias"] + sum(w * v for w, v in zip(payload["weights"], x))
    return min(max(pred, 20.0), 99.0)


def feature_importance(payload: dict) -> list[tuple[str, float]]:
    weights = [abs(w) for w in payload["weights"]]
    total = sum(weights) or 1.0
    ranked = sorted(
        ((name, weight / total) for name, weight in zip(payload["features"], weights)),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked


def train_and_save():
    data = generate_dataset()
    rng = random.Random(7)
    rng.shuffle(data)
    split = int(len(data) * 0.8)
    train, test = data[:split], data[split:]
    weights, bias = train_linear_regression(train)
    mae, r2 = evaluate(test, weights, bias)

    output_dir = Path(__file__).resolve().parent
    output_path = output_dir / "student_model.json"
    payload = {
        "features": FEATURE_NAMES,
        "scales": FEATURE_SCALE,
        "weights": weights,
        "bias": bias,
        "metrics": {"mae": mae, "r2": r2},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Model trained successfully.")
    print(f"Mean Absolute Error: {mae:.2f} marks")
    print(f"R2 Score: {r2:.3f}")
    print(f"Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    train_and_save()
