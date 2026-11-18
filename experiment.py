"""Reproducible synthetic demand modelling. No real clients or production system."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine, text
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    seed: int = 18
    days: int = 240
    regression_steps: int = 180
    classification_steps: int = 150
    bootstrap_samples: int = 1500
    elasticity: float = -1.3


SCHEMA = pa.DataFrameSchema({
    "row_id": pa.Column(str, unique=True),
    "day": pa.Column(int, pa.Check.ge(0)),
    "route": pa.Column(str, pa.Check.isin([f"R{i}" for i in range(4)])),
    "segment": pa.Column(str, pa.Check.isin(["standard", "premium"])),
    "promo": pa.Column(int, pa.Check.isin([0, 1])),
    "market": pa.Column(float, pa.Check(lambda s: np.isfinite(s)), nullable=True),
    "price": pa.Column(float, [pa.Check.gt(0), pa.Check(lambda s: np.isfinite(s))]),
    "demand": pa.Column(float, [pa.Check.gt(0), pa.Check(lambda s: np.isfinite(s))]),
    "source": pa.Column(str, pa.Check.isin(["reference", "shifted"])),
}, strict=True)


def deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def make_data(config: Config, shifted: bool = False) -> pd.DataFrame:
    """Known partially linear log-demand DGP, observable confounding and MCAR gaps."""
    rng = np.random.default_rng(config.seed + (1000 if shifted else 0))
    days = np.repeat(np.arange(config.days), 8)
    route = np.tile(np.repeat(np.arange(4), 2), config.days)
    premium = np.tile(np.arange(2), config.days * 4)
    n = len(days)
    promo = rng.binomial(1, 0.15 if not shifted else 0.4, n)
    market = rng.normal(0 if not shifted else 0.8, 1, n)
    weekly = np.sin(2 * np.pi * days / 7)
    monthly = np.cos(2 * np.pi * days / 30)
    log_price = (math.log(100) + 0.4 * market + 0.15 * promo + 0.08 * route
                 + 0.1 * monthly + rng.normal(0, 0.22, n) + (0.25 if shifted else 0))
    mu = 4.1 + 0.35 * market + 0.12 * market**2 + 0.2 * promo + 0.25 * weekly
    mu += np.array([0.0, 0.1, -0.15, 0.2])[route] + 0.1 * premium
    log_demand = mu + config.elasticity * (log_price - math.log(100)) + rng.normal(0, 0.1, n)
    observed_market = market.copy()
    observed_market[rng.random(n) < 0.03] = np.nan
    source = "shifted" if shifted else "reference"
    frame = pd.DataFrame({"row_id": [f"{source}-{i}" for i in range(n)], "day": days,
                          "route": [f"R{i}" for i in route],
                          "segment": np.where(premium, "premium", "standard"), "promo": promo,
                          "market": observed_market, "price": np.exp(log_price),
                          "demand": np.exp(log_demand), "source": source})
    return SCHEMA.validate(frame)


def temporal_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    unique_days = np.sort(frame["day"].unique())
    if len(unique_days) < 15:
        raise ValueError("At least 15 distinct days are needed")
    a, b = unique_days[int(len(unique_days) * 0.6)], unique_days[int(len(unique_days) * 0.8)]
    return {"train": frame.loc[frame.day < a].copy(),
            "validation": frame.loc[(frame.day >= a) & (frame.day < b)].copy(),
            "test": frame.loc[frame.day >= b].copy()}


class Preprocessor:
    """Fit every learned statistic on the training split, never on validation/test."""
    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        market = frame["market"].fillna(self.market_median)
        base = pd.DataFrame({"market": market, "market_squared": market**2,
                             "log_price": np.log(frame["price"]), "promo": frame["promo"],
                             "weekly_sin": np.sin(2 * np.pi * frame["day"] / 7),
                             "monthly_cos": np.cos(2 * np.pi * frame["day"] / 30)}, index=frame.index)
        categories = pd.get_dummies(frame[["route", "segment"]], dtype=float)
        return pd.concat([base, categories], axis=1).astype(float)

    def fit(self, train: pd.DataFrame) -> "Preprocessor":
        self.market_median = float(train.market.median())
        if not np.isfinite(self.market_median):
            raise ValueError("Training market covariate has no finite observations")
        raw = self.features(train)
        self.columns = list(raw.columns)
        self.mean = raw.mean()
        self.scale = raw.std(ddof=0).replace(0, 1)
        self.fit_ids = tuple(train.row_id)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        raw = self.features(frame).reindex(columns=self.columns, fill_value=0)
        return ((raw - self.mean) / self.scale).to_numpy(dtype=np.float32)

    def description(self) -> dict:
        return {"market_median": self.market_median, "columns": self.columns,
                "mean": self.mean.to_dict(), "scale": self.scale.to_dict(), "fit_rows": len(self.fit_ids)}


def regression_metrics(y: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted) - np.asarray(y)
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    truth, predicted = np.asarray(y).astype(bool), np.asarray(probability) >= 0.5
    tp, tn = int(np.sum(truth & predicted)), int(np.sum(~truth & ~predicted))
    fp, fn = int(np.sum(~truth & predicted)), int(np.sum(truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"accuracy": (tp + tn) / len(truth), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "positive_rate": float(truth.mean())}


def positive_weight(labels: np.ndarray) -> float:
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    positives = float(np.sum(labels))
    if not 0 < positives < len(labels):
        raise ValueError("Training labels must contain both classes")
    return (len(labels) - positives) / positives


class MLP(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 32), nn.ReLU(), nn.Linear(32, 16),
                                    nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def train_mlp(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray,
              steps: int, seed: int, classification: bool = False, weight: float = 1.0) -> tuple[MLP, dict]:
    torch.manual_seed(seed)
    model = MLP(x.shape[1])
    train_x, train_y = torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
    val_x, val_y = torch.from_numpy(vx), torch.tensor(vy, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.015, weight_decay=0.001)
    best, best_loss, best_step = copy.deepcopy(model.state_dict()), float("inf"), 0
    for step in range(steps):
        model.train()
        prediction = model(train_x)
        loss = (F.binary_cross_entropy_with_logits(prediction, train_y, pos_weight=torch.tensor(weight))
                if classification else F.mse_loss(prediction, train_y))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            vp = model(val_x)
            val_loss = (F.binary_cross_entropy_with_logits(vp, val_y, pos_weight=torch.tensor(weight))
                        if classification else F.mse_loss(vp, val_y))
        if float(val_loss) < best_loss:
            best, best_loss, best_step = copy.deepcopy(model.state_dict()), float(val_loss), step + 1
    model.load_state_dict(best)
    model.eval()
    return model, {"selection": "minimum validation loss", "selected_step": best_step,
                   "validation_loss": best_loss, "training_steps": steps, "seed": seed, "positive_weight": weight}


def predict(model: MLP, x: np.ndarray, classifier: bool = False) -> np.ndarray:
    with torch.no_grad():
        value = model(torch.from_numpy(x))
        return (value.sigmoid() if classifier else value).numpy()


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float = 1.0) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x]).astype(float)
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ y)


def ridge_predict(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.maximum(0, np.column_stack([np.ones(len(x)), x]) @ coefficients)


def paired_rmse_interval(y: np.ndarray, first: np.ndarray, second: np.ndarray,
                         seed: int, count: int) -> dict:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(y), (count, len(y)))
    difference = (np.sqrt(np.mean((first[indices] - y[indices])**2, axis=1))
                  - np.sqrt(np.mean((second[indices] - y[indices])**2, axis=1)))
    return {"difference": regression_metrics(y, first)["rmse"] - regression_metrics(y, second)["rmse"],
            "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
            "scope": "paired held-out examples; excludes retraining and temporal-dependence uncertainty"}


def fit_drift(reference: np.ndarray) -> dict:
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("Drift reference must be nonempty and finite")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) == 1:
        edges = np.array([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    counts = np.histogram(reference, bins=edges)[0]
    return {"edges": edges, "frequency": counts / counts.sum(), "reference": np.sort(reference)}


def drift_metrics(reference: dict, current: np.ndarray) -> dict:
    if not len(current) or not np.isfinite(current).all():
        raise ValueError("Drift observations must be nonempty and finite")
    counts = np.histogram(current, bins=reference["edges"])[0]
    p = np.maximum(reference["frequency"], 1e-6)
    q = np.maximum(counts / counts.sum(), 1e-6)
    p, q = p / p.sum(), q / q.sum()
    psi = float(np.sum((q - p) * np.log(q / p)))
    grid = np.sort(np.concatenate([reference["reference"], current]))
    a = np.searchsorted(reference["reference"], grid, side="right") / len(reference["reference"])
    b = np.searchsorted(np.sort(current), grid, side="right") / len(current)
    return {"psi": psi, "ks_statistic": float(np.max(np.abs(a - b))),
            "heuristic_flag_psi_gt_0_2": psi > 0.2, "threshold_is_statistical_test": False}


def causal_design(frame: pd.DataFrame) -> np.ndarray:
    """Only observed confounders; no price, demand or missing-value learned statistics."""
    values = [np.ones(len(frame)), frame.market.to_numpy(), frame.market.to_numpy()**2,
              frame.promo.to_numpy(), np.sin(2 * np.pi * frame.day / 7),
              np.cos(2 * np.pi * frame.day / 30)]
    values.extend((frame.route == f"R{i}").astype(float).to_numpy() for i in range(1, 4))
    values.append((frame.segment == "premium").astype(float).to_numpy())
    return np.column_stack(values)


def partialling_out(frame: pd.DataFrame, seed: int, folds: int = 5) -> dict:
    complete = frame.dropna(subset=["market"]).reset_index(drop=True)
    if len(complete) < 30 or not 2 <= folds <= len(complete) // 2:
        raise ValueError("Insufficient complete cases or invalid fold count")
    x = causal_design(complete)
    treatment = np.log(complete.price.to_numpy()) - math.log(100)
    outcome = np.log(complete.demand.to_numpy())
    residual_t, residual_y = np.empty(len(x)), np.empty(len(x))
    partition = np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)
    audit = []
    for heldout in partition:
        fit = np.setdiff1d(np.arange(len(x)), heldout)
        g = np.linalg.lstsq(x[fit], treatment[fit], rcond=None)[0]
        m = np.linalg.lstsq(x[fit], outcome[fit], rcond=None)[0]
        residual_t[heldout] = treatment[heldout] - x[heldout] @ g
        residual_y[heldout] = outcome[heldout] - x[heldout] @ m
        audit.append({"fit_ids": complete.row_id.iloc[fit].tolist(),
                      "heldout_ids": complete.row_id.iloc[heldout].tolist()})
    moment = float(np.mean(residual_t**2))
    if moment <= 1e-10:
        raise ValueError("No residual treatment variation: elasticity is not identified")
    estimate = float(np.mean(residual_t * residual_y) / moment)
    influence = residual_t * (residual_y - estimate * residual_t) / moment
    se = float(np.std(influence, ddof=1) / math.sqrt(len(x)))
    naive = float(np.linalg.lstsq(np.column_stack([np.ones(len(x)), treatment]), outcome, rcond=None)[0][1])
    return {"estimate": estimate, "naive_unadjusted": naive, "standard_error": se,
            "ci95": [estimate - 1.96 * se, estimate + 1.96 * se], "complete_case_rows": len(x),
            "folds": folds, "fold_audit": audit,
            "assumptions": "Known synthetic partially linear DGP, observable confounding, MCAR gaps and overlap; no real causal claim."}


def run(config: Config, output: Path, track_mlflow: bool = False) -> dict:
    deterministic(config.seed)
    data = make_data(config)
    splits = temporal_split(data)
    shifted = make_data(config, shifted=True)
    shifted = shifted.loc[shifted.day >= splits["test"].day.min()].copy()
    train, validation = splits["train"], splits["validation"]
    prep = Preprocessor().fit(train)
    x, vx = prep.transform(train), prep.transform(validation)
    threshold = float(train.demand.quantile(0.9))
    labels = (train.demand.to_numpy() > threshold).astype(float)
    val_labels = (validation.demand.to_numpy() > threshold).astype(float)
    weight = positive_weight(labels)
    models, histories, target_scales = {}, {}, {}
    for name, logged in (("mlp_log_target", True), ("mlp_raw_target", False)):
        y = np.log(train.demand.to_numpy()) if logged else train.demand.to_numpy()
        vy = np.log(validation.demand.to_numpy()) if logged else validation.demand.to_numpy()
        mean, scale = float(y.mean()), float(y.std())
        model, history = train_mlp(x, (y - mean) / scale, vx, (vy - mean) / scale,
                                   config.regression_steps, config.seed + 10)
        models[name], histories[name], target_scales[name] = model, history, [mean, scale]
    for name, balance in (("weighted_classifier", weight), ("unweighted_classifier", 1.0)):
        models[name], histories[name] = train_mlp(x, labels, vx, val_labels, config.classification_steps,
                                                  config.seed + 20, classification=True, weight=balance)
    ridge = ridge_fit(x, train.demand.to_numpy())
    reference = fit_drift(np.log(train.price.to_numpy()))
    report = {"config": asdict(config), "runtime": {"python": platform.python_version(), "device": "cpu",
               "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "torch", "pandera", "plotly", "SQLAlchemy")}},
              "splits": {}, "preprocessing": prep.description(), "models": histories,
              "training_artifacts": {"validation_selected_states": len(models), "model_checkpoint_files_written": 0},
              "data_quality": {"training_missing_market": int(train.market.isna().sum()),
                               "training_demand_skewness": float(train.demand.skew()),
                               "target_comparison": "raw versus logarithmic target; no universal superiority claim"},
              "rare_event": {"threshold_train_quantile": 0.9, "demand_threshold": threshold,
                             "training_positive_rate": float(labels.mean()), "positive_weight": weight,
                             "sigmoid_score_threshold": 0.5, "scores_calibrated": False}, "evaluation": {},
              "drift": {}, "causal": partialling_out(train, config.seed + 30),
              "limits": ["Synthetic data and local execution, not real revenue-management deployment.",
                         "Single training seed. Example bootstrap does not cover retraining or time dependence.",
                         "PSI threshold is a fixed heuristic; KS is a statistic without a p-value.",
                         "Causal estimates depend on a known simulator and strong assumptions.",
                         "No Argo, Kubernetes, PostgreSQL, GitLab CI, model registry or production service tested."]}
    report["causal"]["known_simulator_elasticity"] = config.elasticity
    report["causal"]["absolute_error"] = abs(report["causal"]["estimate"] - config.elasticity)
    outputs = []
    for name, part in {**splits, "shifted_test": shifted}.items():
        report["splits"][name] = {"rows": len(part), "day_min": int(part.day.min()), "day_max": int(part.day.max()),
                                  "sha256": hashlib.sha256(part.to_csv(index=False).encode()).hexdigest()}
    for name, part in (("test", splits["test"]), ("shifted_test", shifted)):
        tx, y = prep.transform(part), part.demand.to_numpy()
        predictions = {"median_baseline": np.full(len(part), float(train.demand.median())),
                       "ridge_baseline": ridge_predict(ridge, tx)}
        for key in ("mlp_log_target", "mlp_raw_target"):
            mean, scale = target_scales[key]
            predicted = predict(models[key], tx) * scale + mean
            predictions[key] = np.exp(np.clip(predicted, -10, 15)) if key == "mlp_log_target" else np.maximum(0, predicted)
        truth = (y > threshold).astype(int)
        probabilities = {key: predict(models[key], tx, True) for key in ("weighted_classifier", "unweighted_classifier")}
        report["evaluation"][name] = {"regression": {key: regression_metrics(y, value) for key, value in predictions.items()},
                                      "classification": {key: classification_metrics(truth, value) for key, value in probabilities.items()},
                                      "paired_rmse_log_minus_ridge": paired_rmse_interval(y, predictions["mlp_log_target"],
                                          predictions["ridge_baseline"], config.seed + 40, config.bootstrap_samples)}
        report["drift"][name] = drift_metrics(reference, np.log(part.price.to_numpy()))
        rows = part[["row_id", "day", "route", "segment", "price", "demand", "source"]].copy()
        rows["evaluation_split"], rows["high_demand_label"] = name, truth
        for key, value in {**predictions, **probabilities}.items():
            rows[key] = np.asarray(value, dtype=np.float64)
        outputs.append(rows)
    output.mkdir(parents=True, exist_ok=True)
    predictions_frame = pd.concat(outputs, ignore_index=True)
    predictions_frame.to_csv(output / "predictions.csv", index=False)
    engine = create_engine("sqlite:///" + (output / "results.sqlite").resolve().as_posix())
    try:
        predictions_frame.to_sql("predictions", engine, if_exists="replace", index=False)
        with engine.connect() as connection:
            stored = int(connection.scalar(text("SELECT COUNT(*) FROM predictions")))
        assert stored == len(predictions_frame)
    finally:
        engine.dispose()
    report["storage"] = {"engine": "SQLAlchemy / local SQLite", "prediction_rows": stored}
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Demande observée et estimée", "RMSE sur le jeu de test"))
    clean_rows = predictions_frame.loc[predictions_frame.evaluation_split == "test"]
    fig.add_trace(go.Scatter(x=clean_rows.demand, y=clean_rows.mlp_log_target, mode="markers", name="Perceptron, cible logarithmique"), row=1, col=1)
    scores = report["evaluation"]["test"]["regression"]
    fig.add_trace(go.Bar(x=list(scores), y=[v["rmse"] for v in scores.values()], name="RMSE"), row=1, col=2)
    fig.update_layout(title="Étude pédagogique : demande synthétique", template="plotly_white", showlegend=False)
    fig.write_html(output / "dashboard.html", include_plotlyjs="cdn", full_html=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if track_mlflow:
        import mlflow
        uri = "sqlite:///" + (output / "tracking.sqlite").resolve().as_posix()
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("synthetic-demand-portfolio")
        with mlflow.start_run(run_name="reproducible-local-run") as active:
            mlflow.log_params(asdict(config))
            mlflow.log_metrics({f"test_{name}_rmse": v["rmse"] for name, v in scores.items()})
            mlflow.log_metrics({"causal_elasticity": report["causal"]["estimate"],
                               "test_precision_weighted": report["evaluation"]["test"]["classification"]["weighted_classifier"]["precision"]})
            mlflow.log_artifact(str(output / "report.json"), artifact_path="evaluation")
            report["tracking"] = {"backend": "local MLflow SQLite", "run_id": active.info.run_id,
                                  "package": importlib.metadata.version("mlflow-skinny"), "registry_used": False}
        client = mlflow.MlflowClient()
        stored_run = client.get_run(report["tracking"]["run_id"])
        assert stored_run.info.status == "FINISHED"
        assert stored_run.data.metrics["test_mlp_log_target_rmse"] == scores["mlp_log_target"]["rmse"]
        artifacts = client.list_artifacts(stored_run.info.run_id, "evaluation")
        assert any(item.path == "evaluation/report.json" for item in artifacts)
        report["tracking"]["readback_verified"] = True
        report["tracking"]["status"] = stored_run.info.status
        report["tracking"]["stored_metric_count"] = len(stored_run.data.metrics)
        report["tracking"]["stored_report_artifact_count"] = sum(item.path == "evaluation/report.json" for item in artifacts)
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    report = run(Config(), args.output, track_mlflow=args.mlflow)
    print(json.dumps({"test": report["evaluation"]["test"], "drift": report["drift"],
                      "causal_estimate": report["causal"]["estimate"],
                      "causal_truth": report["config"]["elasticity"], "prediction_rows": report["storage"]["prediction_rows"]}, indent=2))


if __name__ == "__main__":
    main()
