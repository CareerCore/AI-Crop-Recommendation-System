from __future__ import annotations

import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # render straight to files; no GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import readme_builder
from inference import CROP_FEATURES, add_crop_features, predict_bug, predict_crop

warnings.filterwarnings("ignore")

# ============================== CONFIG ========================================
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "ml" / "data" / "raw"
ART_DIR = ROOT / "ml" / "artifacts"
REP_DIR = ROOT / "ml" / "reports"

SEED = 42
N_ITER = 12  # random-search draws per model
TOP_K = 3  # models combined in the voting ensemble
MIN_CLASS = 30  # classes with fewer rows are dropped (too few to learn or evaluate)
MODEL_VERSION = "2.0.0"
KEEP_ARCHIVES = 3  # previous model versions kept per target
QUICK = os.environ.get("ML_QUICK") == "1"
ONLY = {k.strip() for k in os.environ.get("ML_ONLY", "").split(",") if k.strip()}

SEV_MOZ = {
    "blocker": "S1-S2",
    "critical": "S1-S2",
    "major": "S1-S2",
    "normal": "S3",
    "minor": "S4",
    "trivial": "S4",
    "s1": "S1-S2",
    "s2": "S1-S2",
    "s3": "S3",
    "s4": "S4",
}


SPECS = [
    {
        "key": "crop_recommendation",
        "title": "Crop Recommendation Engine",
        "kind": "tabular",
        "source": "Public crop recommendation dataset (Kaggle, 2,200 field records)",
        "description": "Recommends the best-suited crop (22 classes) from soil nutrients, climate and rainfall.",
        "match": CROP_FEATURES + ["label"],
        "id_col": None,
        "time_col": None,
        "targets": {"crop": {"col": "label", "lower": True}},
    },
]

BG, GREEN = "#f8fdf9", "#1b5e20"
BASE_C = "#90a4ae"
FEAT_COLORS = [
    "#2e7d32",
    "#1565c0",
    "#6a1b9a",
    "#c62828",
    "#00838f",
    "#e65100",
    "#558b2f",
]
# ==============================================================================


# ------------------------------- discovery & loading -------------------------
def discover() -> tuple[dict, list]:
    """Scan ml/data/raw for CSVs and assign each to a dataset profile."""
    found, skipped = {}, []
    for p in sorted(RAW_DIR.rglob("*.csv")):
        try:
            cols = set(pd.read_csv(p, nrows=0).columns)
        except Exception as e:  # unreadable / empty file  # noqa: BLE001
            skipped.append((p, f"unreadable ({e})"))
            continue
        spec = next((s for s in SPECS if set(s["match"]) <= cols), None)
        if spec is None:
            skipped.append((p, "no matching dataset profile"))
        elif not spec.get("enabled", True):
            skipped.append((p, spec["disabled_reason"]))
        elif ONLY and spec["key"] not in ONLY:
            skipped.append((p, "not selected (ML_ONLY)"))
        else:
            found.setdefault(spec["key"], []).append(p)
    return found, skipped


def load_frame(spec: dict, paths: list[Path]) -> tuple[pd.DataFrame, int]:
    need = [
        spec.get("id_col"),
        spec.get("time_col"),
        *spec.get("text_cols", []),
        *spec.get("cat_cols", []),
        *[c for v in spec.get("extra_cat", {}).values() for c in v],
        *[t["col"] for t in spec["targets"].values()],
    ]
    if spec["kind"] == "tabular":
        need += CROP_FEATURES
    need = list(dict.fromkeys(c for c in need if c))
    frames = []
    for p in paths:
        hdr = set(pd.read_csv(p, nrows=0).columns)
        frames.append(
            pd.read_csv(p, usecols=[c for c in need if c in hdr], low_memory=False)
        )
    df = pd.concat(frames, ignore_index=True)
    if spec.get("id_col") and spec["id_col"] in df:
        df = df.drop_duplicates(spec["id_col"])
    df = df.drop_duplicates() if spec["kind"] == "tabular" else df

    if spec["kind"] == "text":
        tcols = spec["text_cols"]
        blank = (
            df[tcols]
            .fillna("")
            .astype(str)
            .apply(lambda s: s.str.strip() == "")
            .all(axis=1)
        )
        df = df[~blank].copy()
        if spec.get("time_col"):
            df[spec["time_col"]] = pd.to_datetime(
                df[spec["time_col"]], utc=True, errors="coerce"
            )
            df = df.dropna(subset=[spec["time_col"]]).sort_values(spec["time_col"])
    n_total = len(df)
    if spec.get("sample_rows") and n_total > spec["sample_rows"]:
        df = df.sample(spec["sample_rows"], random_state=SEED)
        if spec.get("time_col"):
            df = df.sort_values(spec["time_col"])
    df = df.reset_index(drop=True)

    if spec["kind"] == "text":
        df["text"] = (
            df[spec["text_cols"]].fillna("").astype(str).agg(" . ".join, axis=1)
        )
        for c in [
            *spec.get("cat_cols", []),
            *[c for v in spec.get("extra_cat", {}).values() for c in v],
        ]:
            if c in df:
                df[c] = df[c].fillna("NA").astype(str).str.strip()
    return df, n_total


def make_target(s: pd.Series, spec: dict) -> pd.Series:
    s = s.astype(object).map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    if spec.get("lower"):
        s = s.map(lambda v: v.lower() if isinstance(v, str) else v)
    if spec.get("map"):
        s = s.map(spec["map"])
    if spec.get("keep"):
        s = s.where(s.isin(spec["keep"]))
    return s


def split_indices(n: int, y: np.ndarray, spec: dict):
    if spec["kind"] == "tabular":
        tr, te = train_test_split(
            np.arange(n), test_size=0.20, stratify=y, random_state=SEED
        )
        return tr, None, te, "stratified random 80/20"
    if spec.get("time_col"):
        i, j = int(n * 0.70), int(n * 0.85)
        return np.arange(i), np.arange(i, j), np.arange(j, n), "chronological 70/15/15"
    tr, rest = train_test_split(
        np.arange(n), test_size=0.30, stratify=y, random_state=SEED
    )
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    return tr, va, te, "stratified random 70/15/15"


# ------------------------------- models --------------------------------------
def baseline_crop_features(X: pd.DataFrame) -> pd.DataFrame:
    """Features of the original crop notebook, kept as the reference baseline."""
    X = X[CROP_FEATURES].astype(float).copy()
    X["NPK_mean"] = (X["N"] + X["P"] + X["K"]) / 3
    X["THI"] = X["temperature"] * X["humidity"] / 100
    X["ph_category"] = np.select([X["ph"] < 5.5, X["ph"] <= 7.5], [0, 1], 2)
    X["rainfall_level"] = np.select(
        [X["rainfall"] <= 50, X["rainfall"] <= 100, X["rainfall"] <= 200], [0, 1, 2], 3
    )
    return X


def char_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=150_000,
    )


def text_prep(cat_cols: list[str], char: bool) -> ColumnTransformer:
    parts = [
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000
            ),
            "text",
        )
    ]
    if char:
        parts.append(
            ("tfidf_char", "drop", "text")
        )  # switched on by the search when it helps
    if cat_cols:
        parts.append(
            (
                "oh",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5),
                cat_cols,
            )
        )
    return ColumnTransformer(parts)


def text_candidates(cat_cols: list[str], char: bool) -> dict:
    def pipe(clf):
        return Pipeline([("prep", text_prep(cat_cols, char)), ("clf", clf)])

    tf = {
        "prep__tfidf__ngram_range": [(1, 1), (1, 2)],
        "prep__tfidf__min_df": [1, 2, 3],
    }
    if char:
        tf["prep__tfidf_char"] = ["drop", char_vectorizer()]
    return {
        "LogReg": (
            pipe(LogisticRegression(max_iter=300, class_weight="balanced")),
            {**tf, "clf__C": [0.3, 1, 3, 10, 30]},
        ),
        "LinearSVC": (
            pipe(CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)),
            {**tf, "clf__estimator__C": [0.1, 0.3, 1, 3]},
        ),
        "ComplementNB": (
            pipe(ComplementNB()),
            {**tf, "clf__alpha": [0.03, 0.1, 0.3, 1.0]},
        ),
    }


def tabular_candidates() -> dict:
    trees = {
        "n_estimators": [300, 500, 800],
        "max_depth": [None, 12, 20],
        "min_samples_split": [2, 4],
        "max_features": ["sqrt", "log2"],
    }
    return {
        "RandomForest": (RandomForestClassifier(random_state=SEED, n_jobs=1), trees),
        "ExtraTrees": (ExtraTreesClassifier(random_state=SEED, n_jobs=1), trees),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=SEED),
            {
                "learning_rate": [0.05, 0.1, 0.2],
                "max_iter": [150, 300],
                "max_depth": [None, 6, 10],
                "l2_regularization": [0.0, 0.1, 1.0],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=SEED, eval_metric="mlogloss", tree_method="hist", n_jobs=1
            ),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
        ),
        "SVM-RBF": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("svc", SVC(probability=True, random_state=SEED)),
                ]
            ),
            {"svc__C": [1, 3, 10, 30, 100], "svc__gamma": ["scale", 0.03, 0.1, 0.3]},
        ),
    }


def tune(name, estimator, space, X, y, cv, n_iter) -> dict:
    t0 = time.time()
    s = RandomizedSearchCV(
        estimator,
        space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        random_state=SEED,
        refit=True,
    ).fit(X, y)
    i = s.best_index_
    print(
        f"    {name:<22} cv macro-F1 {s.best_score_:.4f} +/- {s.cv_results_['std_test_score'][i]:.4f}"
        f"   ({time.time() - t0:.0f}s)"
    )
    return {
        "name": name,
        "model": s.best_estimator_,
        "cv_f1": float(s.best_score_),
        "params": s.best_params_,
    }


def run_baseline(spec, X_tr, y_tr, X_te, y_te, classes, cat_cols):
    """Reference model the delivered one is compared with."""
    if spec["kind"] == "tabular":
        pipe = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("rf", RandomForestClassifier(random_state=SEED)),
            ]
        )
        grid = {
            "rf__n_estimators": [50] if QUICK else [50, 100, 200],
            "rf__max_depth": [None] if QUICK else [None, 10, 20],
            "rf__min_samples_split": [2] if QUICK else [2, 5],
        }
        model = (
            GridSearchCV(pipe, grid, cv=5, scoring="accuracy", n_jobs=-1)
            .fit(X_tr, y_tr)
            .best_estimator_
        )
        name = "MinMax + Random Forest (grid search)"
    else:
        model = Pipeline(
            [
                ("prep", text_prep(cat_cols, False)),
                ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
            ]
        ).fit(X_tr, y_tr)
        name = "TF-IDF + Logistic Regression (untuned)"
    return name, model


# ------------------------------- metrics -------------------------------------
def wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - m) / d), float((c + m) / d)


def f1m(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


def score(model, X, y, n_classes) -> tuple[dict, np.ndarray]:
    p = model.predict(X)
    acc = float(accuracy_score(y, p))
    pr, rc, f1, _ = precision_recall_fscore_support(
        y, p, average="macro", zero_division=0
    )
    lo, hi = wilson(acc, len(y))
    out = {
        "accuracy": acc,
        "macro_precision": float(pr),
        "macro_recall": float(rc),
        "macro_f1": float(f1),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "errors": int((p != y).sum()),
        "n": len(y),
        "acc_ci_low": lo,
        "acc_ci_high": hi,
    }
    try:
        out["log_loss"] = float(
            log_loss(y, model.predict_proba(X), labels=np.arange(n_classes))
        )
    except Exception:  # noqa: BLE001
        out["log_loss"] = None
    return out, p


# ------------------------------- plotting ------------------------------------
def set_theme() -> None:
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": "#c8e6c9",
            "grid.color": "#e8f5e9",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlecolor": GREEN,
            "axes.labelcolor": "#37474f",
            "xtick.color": "#37474f",
            "ytick.color": "#37474f",
            "legend.framealpha": 0.92,
            "figure.dpi": 110,
        }
    )


def save_fig(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path.relative_to(ROOT).as_posix()


def fig_class_distribution(labels: pd.Series, title: str, path: Path) -> str:
    counts = labels.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.34 * len(counts) + 1.5)))
    ax.barh(
        counts.index.astype(str),
        counts.values,
        color=GREEN,
        alpha=0.85,
        edgecolor="white",
    )
    for i, v in enumerate(counts.values):
        ax.text(
            v + counts.max() * 0.01,
            i,
            f"{v:,}  ({v / counts.sum():.1%})",
            va="center",
            fontsize=9,
            color="#37474f",
        )
    ax.set_xlim(0, counts.max() * 1.22)
    ax.set_title(title)
    ax.set_xlabel("Records")
    return save_fig(fig, path)


def fig_selection(
    cands, sel_label, chosen, base_name, base_m, final_m, path: Path
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    order = sorted(cands, key=lambda c: c[1])
    lo = max(0.0, min(c[1] for c in order) - 0.08)
    for k, c in enumerate(order):
        win = c[0] == chosen
        ax.hlines(k, lo, c[1], color="#a5d6a7", lw=3)
        ax.plot(
            c[1],
            k,
            "o",
            color=GREEN if win else "#66bb6a",
            markersize=15 if win else 10,
        )
        ax.text(
            c[1] + 0.012,
            k,
            f"{c[1]:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold" if win else None,
        )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c[0] for c in order])
    ax.set_xlim(lo, min(1.03, max(c[1] for c in order) + 0.07))
    ax.set_title(f"Candidate models ({sel_label})")
    ax.set_xlabel(f"macro-F1   (axis starts at {lo:.2f}; selected model enlarged)")

    ax = axes[1]
    keys = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("weighted_f1", "Weighted-F1"),
    ]
    bv, fv = [base_m[k] for k, _ in keys], [final_m[k] for k, _ in keys]
    x, w = np.arange(3), 0.36
    ax.bar(x - w / 2, bv, w, color=BASE_C, label=base_name, edgecolor="white")
    ax.bar(x + w / 2, fv, w, color=GREEN, label="Delivered model", edgecolor="white")
    for xs, vs in ((x - w / 2, bv), (x + w / 2, fv)):
        for xi, v in zip(xs, vs):
            ax.text(
                xi,
                v + 0.012,
                f"{v:.3f}",
                ha="center",
                fontsize=9,
                fontweight="bold",
                color="#37474f",
            )
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in keys])
    ax.set_title("Test set: reference vs delivered")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return save_fig(fig, path)


def fig_confusion(y, p, classes, title, path: Path):
    n = len(classes)
    cm = confusion_matrix(y, p, labels=np.arange(n))
    if n <= 8:
        pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annot = np.array(
            [[f"{pct[i, j]:.0%}\n({cm[i, j]})" for j in range(n)] for i in range(n)]
        )
        data, size, vmax = pct, (8, 6.5), 1.0
    else:
        annot, data, size, vmax = (
            np.where(cm > 0, cm.astype(str), ""),
            cm,
            (15, 12.5),
            None,
        )
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(
        data,
        annot=annot,
        fmt="",
        cmap="Greens",
        vmin=0,
        vmax=vmax,
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45)
    return save_fig(fig, path), cm


def fig_per_class(report, classes, path: Path) -> str:
    d = pd.DataFrame(
        [
            (
                c,
                report[c]["precision"],
                report[c]["recall"],
                report[c]["f1-score"],
                int(report[c]["support"]),
            )
            for c in classes
        ],
        columns=["c", "p", "r", "f", "n"],
    ).sort_values("f")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(d) + 1.6)))
    y, w = np.arange(len(d)), 0.26
    ax.barh(y - w, d.p, w, label="Precision", color="#1565c0")
    ax.barh(y, d.r, w, label="Recall", color="#2e7d32")
    ax.barh(y + w, d.f, w, label="F1", color="#6a1b9a")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  (n={n})" for c, n in zip(d.c, d.n)])
    ax.set_xlim(0, 1.05)
    ax.set_title("Per-class precision, recall and F1 (test set)")
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    return save_fig(fig, path)


def fig_drift(d: pd.DataFrame, time_col: str, path: Path) -> str:
    ct = pd.crosstab(d[time_col].dt.year, d["y"], normalize="index").mul(100)
    greens = sns.color_palette("YlGn", n_colors=len(ct.columns) + 2)[2:]
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(
        kind="bar", stacked=True, ax=ax, color=greens, edgecolor="white", width=0.75
    )
    ax.set_title("Label mix by year (why evaluation is chronological)")
    ax.set_xlabel("Year reported")
    ax.set_ylabel("% of labelled reports")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left")
    return save_fig(fig, path)


def fig_importance(imp: pd.DataFrame, path: Path) -> str:
    d = imp.sort_values("importance")
    top = set(d.nlargest(3, "importance")["feature"])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(
        d["feature"],
        d["importance"],
        xerr=d["std"],
        height=0.62,
        edgecolor="white",
        ecolor="#78909c",
        color=[GREEN if f in top else "#a5d6a7" for f in d["feature"]],
    )
    ax.set_title("What drives the recommendation (permutation importance, test set)")
    ax.set_xlabel("drop in macro-F1 when the feature is shuffled")
    return save_fig(fig, path)


def crop_eda(df: pd.DataFrame, fig_dir: Path) -> dict:
    out = {}
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.ravel()
    for ax, col, color in zip(axes, CROP_FEATURES, FEAT_COLORS):
        sns.histplot(
            df[col],
            kde=True,
            bins=30,
            color=color,
            alpha=0.35,
            edgecolor="white",
            ax=ax,
        )
        ax.axvline(
            df[col].mean(),
            color=color,
            ls="--",
            lw=1.8,
            label=f"mean {df[col].mean():.1f}",
        )
        ax.axvline(
            df[col].median(),
            color="#37474f",
            ls=":",
            lw=1.8,
            label=f"median {df[col].median():.1f}",
        )
        ax.axvspan(
            df[col].quantile(0.25), df[col].quantile(0.75), color=color, alpha=0.08
        )
        ax.set_title(col)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle(
        "Feature distributions (dashed = mean, dotted = median, shaded = IQR)",
        fontsize=16,
        fontweight="bold",
        color=GREEN,
        y=1.02,
    )
    fig.tight_layout()
    out["distributions"] = save_fig(fig, fig_dir / "eda_distributions.png")

    corr = df[CROP_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.heatmap(
        corr,
        mask=np.triu(np.ones_like(corr, dtype=bool)),
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        cmap=sns.diverging_palette(145, 10, s=80, l=50, as_cmap=True),
        square=True,
        linewidths=2,
        linecolor="white",
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Feature correlation")
    out["correlation"] = save_fig(fig, fig_dir / "eda_correlation.png")

    means = df.groupby("label")[CROP_FEATURES].mean()
    z = (means - means.mean()) / means.std()
    fig, ax = plt.subplots(figsize=(11, 10))
    sns.heatmap(
        z,
        annot=means.round(0).astype(int),
        fmt="d",
        cmap="RdYlGn",
        center=0,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.7, "label": "z-score across crops"},
        ax=ax,
    )
    ax.set_title(
        "Crop profiles: mean value per feature (colour = how distinctive for that crop)"
    )
    ax.set_ylabel("")
    out["profiles"] = save_fig(fig, fig_dir / "eda_profiles.png")
    return out


def fig_overview(modules: list[dict], path: Path) -> str | None:
    rows = [
        (
            f"{m['title']} / {t['target']}",
            t["baseline"]["macro_f1"],
            t["test"]["macro_f1"],
        )
        for m in modules
        for t in m["targets"]
    ]
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(12, max(3.2, 0.9 * len(rows) + 1.4)))
    y, h = np.arange(len(rows)), 0.36
    ax.barh(
        y - h / 2,
        [r[1] for r in rows],
        h,
        color=BASE_C,
        label="Reference baseline",
        edgecolor="white",
    )
    ax.barh(
        y + h / 2,
        [r[2] for r in rows],
        h,
        color=GREEN,
        label="Delivered model",
        edgecolor="white",
    )
    for i, r in enumerate(rows):
        ax.text(
            r[1] + 0.006,
            i - h / 2,
            f"{r[1]:.3f}",
            va="center",
            fontsize=9,
            color="#455a64",
        )
        ax.text(
            r[2] + 0.006,
            i + h / 2,
            f"{r[2]:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=GREEN,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(
        "Macro-F1 on the held-out test set (higher is better; every class counts equally)"
    )
    ax.set_title("Results overview")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    return save_fig(fig, path)


# ------------------------------- persistence ---------------------------------
def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save_bundle(out_dir: Path, bundle: dict, metrics: dict) -> None:
    """Write the new model; the previous one is archived, not deleted."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cur, cur_m = out_dir / "model.joblib", out_dir / "metrics.json"
    if cur.exists():
        arch = (
            out_dir
            / "archive"
            / datetime.fromtimestamp(cur.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        )
        arch.mkdir(parents=True, exist_ok=True)
        shutil.move(str(cur), arch / "model.joblib")
        if cur_m.exists():
            shutil.move(str(cur_m), arch / "metrics.json")
        old = sorted((out_dir / "archive").iterdir())
        for stale in old[:-KEEP_ARCHIVES]:
            shutil.rmtree(stale, ignore_errors=True)
    joblib.dump(bundle, cur, compress=3)
    cur_m.write_text(json.dumps(metrics, indent=2, default=_js), encoding="utf-8")


# ------------------------------- one target ----------------------------------
def run_target(
    spec: dict, df: pd.DataFrame, tname: str, tspec: dict, n_total: int
) -> dict | None:
    t_start = time.time()
    print(f"\n  -> target: {tname}")
    d = df.copy()
    d["y"] = make_target(d[tspec["col"]], tspec)
    d = d.dropna(subset=["y"])
    vc = d["y"].value_counts()
    d = d[d["y"].isin(vc[vc >= MIN_CLASS].index)].reset_index(drop=True)
    if d["y"].nunique() < 2 or len(d) < 200:
        print("    skipped: not enough labelled rows")
        return None

    cat_cols = [
        c
        for c in spec.get("cat_cols", []) + spec.get("extra_cat", {}).get(tname, [])
        if c in d
    ]
    is_tab = spec["kind"] == "tabular"
    X = add_crop_features(d[CROP_FEATURES]) if is_tab else d[["text"] + cat_cols]
    X_ref = baseline_crop_features(d[CROP_FEATURES]) if is_tab else X
    le = LabelEncoder()
    y = le.fit_transform(d["y"])
    classes = list(le.classes_)
    tr, va, te, split_name = split_indices(len(d), y, spec)
    print(
        f"    rows {len(d):,} | classes {len(classes)} | split {split_name}: "
        f"train {len(tr):,} / val {0 if va is None else len(va):,} / test {len(te):,}"
    )

    fig_dir = REP_DIR / "figures"
    figs = {
        "class_distribution": fig_class_distribution(
            d["y"],
            f"{spec['title']}: {tname} label distribution ({len(d):,} records)",
            fig_dir / "class_distribution.png",
        )
    }
    if spec.get("time_col"):
        figs["drift"] = fig_drift(d, spec["time_col"], fig_dir / "label_drift.png")
    if is_tab:
        figs.update(
            {
                f"eda_{k}": v
                for k, v in crop_eda(d.assign(label=d["y"]), fig_dir).items()
            }
        )

    Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
    Xva, yva = (X.iloc[va], y[va]) if va is not None else (None, None)

    # reference baseline
    base_name, base_model = run_baseline(
        spec, X_ref.iloc[tr], ytr, X_ref.iloc[te], yte, classes, cat_cols
    )
    base_m, _ = score(base_model, X_ref.iloc[te], yte, len(classes))
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    maj_f1 = f1m(yte, maj.predict(Xte))
    print(f"    {base_name}: test macro-F1 {base_m['macro_f1']:.4f}")

    # tune candidates on the training split only
    big = len(tr) > 50_000
    folds = 3 if (QUICK or big) else 5
    n_iter = 2 if QUICK else (6 if big else N_ITER)
    cv = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    cands = (
        tabular_candidates()
        if is_tab
        else text_candidates(cat_cols, spec.get("char_ngrams", False))
    )
    print(f"    tuning {len(cands)} models ({folds}-fold CV, {n_iter} draws each)")
    tuned = sorted(
        (tune(n, e, s, Xtr, ytr, cv, n_iter) for n, (e, s) in cands.items()),
        key=lambda r: r["cv_f1"],
        reverse=True,
    )

    # choose the winner on held-out data (validation split if there is one, otherwise cross-validation)
    ens = VotingClassifier(
        [(r["name"], clone(r["model"])) for r in tuned[:TOP_K]], voting="soft"
    )
    ens_name = "Ensemble(" + "+".join(r["name"] for r in tuned[:TOP_K]) + ")"
    if Xva is not None:
        sel_label = "validation macro-F1"
        ens.fit(Xtr, ytr)
        for r in tuned:
            r["sel"] = f1m(yva, r["model"].predict(Xva))
        ens_sel = f1m(yva, ens.predict(Xva))
    else:
        sel_label = "cross-validated macro-F1"
        for r in tuned:
            r["sel"] = r["cv_f1"]
        ens_sel = float(
            cross_val_score(ens, Xtr, ytr, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
        )
        ens.fit(Xtr, ytr)
    best = max(tuned, key=lambda r: r["sel"])
    if ens_sel > best["sel"] + 0.001:  # the simpler single model wins ties
        final_name, final_model = ens_name, ens
    else:
        final_name, final_model = best["name"], best["model"]
    sel_rows = [(r["name"], r["sel"]) for r in tuned] + [(ens_name, ens_sel)]
    print(f"    selected: {final_name} ({sel_label} {dict(sel_rows)[final_name]:.4f})")

    # one-time test evaluation
    final_m, pred = score(final_model, Xte, yte, len(classes))
    report = classification_report(
        yte,
        pred,
        labels=np.arange(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    figs["selection"] = fig_selection(
        sel_rows,
        sel_label,
        final_name,
        base_name,
        base_m,
        final_m,
        fig_dir / "model_selection.png",
    )
    figs["confusion"], cm = fig_confusion(
        yte,
        pred,
        classes,
        f"{spec['title']}: {tname} (test set)",
        fig_dir / "confusion_matrix.png",
    )
    figs["per_class"] = fig_per_class(
        report, classes, fig_dir / "per_class_metrics.png"
    )
    if is_tab:
        imp = permutation_importance(
            final_model,
            Xte,
            yte,
            scoring="f1_macro",
            n_repeats=10,
            random_state=SEED,
            n_jobs=-1,
        )
        figs["importance"] = fig_importance(
            pd.DataFrame(
                {
                    "feature": Xte.columns,
                    "importance": imp.importances_mean,
                    "std": imp.importances_std,
                }
            ),
            fig_dir / "feature_importance.png",
        )
    print(
        f"    test: accuracy {final_m['accuracy']:.4f} | macro-F1 {final_m['macro_f1']:.4f} | "
        f"mistakes {final_m['errors']}/{final_m['n']}"
    )

    # deployable model: refit on every labelled row, so it also learns from the newest data
    deploy = clone(final_model).fit(X, y)
    text_cols = spec.get("text_cols", [])
    bundle = {
        "model": deploy,
        "classes": classes,
        "kind": spec["kind"],
        "dataset": spec["key"],
        "target": tname,
        "text_cols": text_cols,
        "cat_cols": cat_cols,
        "features": CROP_FEATURES if is_tab else None,
        "version": MODEL_VERSION,
        "model_name": final_name,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "trained_rows": len(d),
        "test_metrics": {k: final_m[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
    }
    save_bundle(
        ART_DIR,
        bundle,
        {
            "dataset": spec["key"],
            "target": tname,
            "model_name": final_name,
            "version": MODEL_VERSION,
            "classes": classes,
            "split": split_name,
            "selection": {"metric": sel_label, "candidates": dict(sel_rows)},
            "params": {r["name"]: r["params"] for r in tuned},
            "test": final_m,
            "baseline": dict(name=base_name, **base_m),
            "trained_rows": len(d),
        },
    )

    # real example for the README
    src = d.iloc[te[0]]
    if is_tab:
        ex_in = {c: float(src[c]) for c in CROP_FEATURES}
        ex_out = predict_crop(bundle, **ex_in)[0]
    else:
        ex_in = {c: str(src[c]) for c in text_cols + cat_cols}
        ex_out = predict_bug(bundle, **ex_in)
    example = {"input": ex_in, "output": ex_out, "true": str(src["y"])}

    # automatic validation notes (shown in the README)
    notes = []
    if not is_tab:
        uniq = d["text"].nunique()
        if uniq / len(d) < 0.05:
            notes.append(
                f"The source reports are highly repetitive ({uniq:,} distinct texts across {len(d):,} records), "
                "so scores partly reflect pattern memorisation. Re-validate on live data before go-live."
            )
    if spec.get("time_col"):
        notes.append(
            "Evaluation is chronological: models learn from older reports and are tested on newer ones."
        )
    shares = d["y"].value_counts(normalize=True)
    if shares.min() < 0.05:
        notes.append(
            f"Minority class '{shares.idxmin()}' is only {shares.min():.1%} of the data; classes are weighted and "
            "per-class recall is reported."
        )
    drift = (
        pd.Series(le.inverse_transform(ytr))
        .value_counts(normalize=True)
        .sub(
            pd.Series(le.inverse_transform(yte)).value_counts(normalize=True),
            fill_value=0,
        )
        .abs()
        .max()
    )
    if drift > 0.10:
        notes.append(
            f"The label mix shifts by up to {drift * 100:.0f} percentage points between training and test data."
        )
    if len(d) < n_total:
        notes.append(
            f"Trained on {len(d):,} labelled records ({n_total:,} rows in the source file before sampling and label filtering)."
        )
    if (
        final_m["acc_ci_low"] <= base_m["acc_ci_high"]
        and base_m["acc_ci_low"] <= final_m["acc_ci_high"]
    ):
        notes.append(
            "The accuracy gain over the reference is within sampling noise on this test set (95% intervals overlap); "
            "the main gains are validation rigour and a reproducible model package."
        )

    top_conf = sorted(
        (
            (int(cm[i, j]), classes[i], classes[j])
            for i in range(len(classes))
            for j in range(len(classes))
            if i != j and cm[i, j] > 0
        ),
        reverse=True,
    )[:5]
    per_class = [
        {
            "label": c,
            "precision": report[c]["precision"],
            "recall": report[c]["recall"],
            "f1": report[c]["f1-score"],
            "support": int(report[c]["support"]),
        }
        for c in classes
    ]
    entry = {
        "target": tname,
        "classes": classes,
        "n_classes": len(classes),
        "rows": len(d),
        "rows_source": n_total,
        "split": split_name,
        "n_train": len(tr),
        "n_val": 0 if va is None else len(va),
        "n_test": len(te),
        "selected_model": final_name,
        "selection_metric": sel_label,
        "selection_score": dict(sel_rows)[final_name],
        "test": final_m,
        "baseline": dict(name=base_name, **base_m),
        "majority_macro_f1": maj_f1,
        "per_class": per_class,
        "top_confusions": top_conf,
        "figures": figs,
        "example": example,
        "notes": notes,
        "version": MODEL_VERSION,
        "artifact": (ART_DIR / "model.joblib")
        .relative_to(ROOT)
        .as_posix(),
        "date_range": [
            str(d[spec["time_col"]].min().date()),
            str(d[spec["time_col"]].max().date()),
        ]
        if spec.get("time_col")
        else None,
        "seconds": round(time.time() - t_start),
    }
    results = [
        "# " + f"{spec['title']} / {tname}",
        "",
        f"Selected model: **{final_name}**  |  split: {split_name}  |  rows: {len(d):,}",
        "",
        "| metric | reference | delivered |",
        "|---|---|---|",
    ]
    for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        results.append(f"| {k} | {base_m[k]:.4f} | {final_m[k]:.4f} |")
    results += [
        "",
        "## Per class",
        "",
        "| class | precision | recall | F1 | support |",
        "|---|---|---|---|---|",
    ]
    results += [
        f"| {p['label']} | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {p['support']} |"
        for p in per_class
    ]
    results += ["", "## Most frequent confusions (true -> predicted)", ""]
    results += [f"- {a} -> {b}: {n}" for n, a, b in top_conf] or ["- none"]
    (REP_DIR ).mkdir(parents=True, exist_ok=True)
    (REP_DIR / "results.md").write_text(
        "\n".join(results), encoding="utf-8"
    )
    print(f"    done in {time.time() - t_start:.0f}s")
    return entry


# ------------------------------- main ----------------------------------------
def main() -> None:
    t0 = time.time()
    set_theme()
    print(f"Scanning {RAW_DIR} ...")
    found, skipped = discover()
    for key, paths in found.items():
        print(f"  {key}: " + ", ".join(p.name for p in paths))
    for p, why in skipped:
        print(f"  {p.name}: {why}")
    if not found:
        sys.exit("No usable CSV files found in ml/data/raw.")

    prev = {}
    if (
        ONLY and (REP_DIR / "index.json").exists()
    ):  # partial run: keep results of the other modules
        try:
            prev = {
                m["key"]: m
                for m in json.loads(
                    (REP_DIR / "index.json").read_text(encoding="utf-8")
                )["modules"]
            }
        except Exception:  # noqa: BLE001
            prev = {}
    fresh = {}
    for spec in SPECS:
        if spec["key"] not in found:
            continue
        print(f"\n=== {spec['title']} ===")
        df, n_total = load_frame(spec, found[spec["key"]])
        print(f"  loaded {len(df):,} rows from {len(found[spec['key']])} file(s)")
        targets = [
            e
            for tn, ts in spec["targets"].items()
            if (e := run_target(spec, df, tn, ts, n_total))
        ]
        if targets:
            fresh[spec["key"]] = {
                "key": spec["key"],
                "title": spec["title"],
                "description": spec["description"],
                "source": spec["source"],
                "kind": spec["kind"],
                "files": [p.name for p in found[spec["key"]]],
                "targets": targets,
            }
    modules = [
        fresh.get(s["key"]) or prev.get(s["key"])
        for s in SPECS
        if s["key"] in fresh or s["key"] in prev
    ]
    index = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "modules": modules,
        "skipped": [{"file": p.name, "reason": r} for p, r in skipped if "ML_ONLY" not in r],
    }
    REP_DIR.mkdir(parents=True, exist_ok=True)
    index["overview_figure"] = fig_overview(index["modules"], REP_DIR / "overview.png")
    (REP_DIR / "index.json").write_text(
        json.dumps(index, indent=2, default=_js), encoding="utf-8"
    )
    readme_builder.build(index, ROOT)
    print(
        f"\nAll done in {time.time() - t0:.0f}s. README.md refreshed; models in ml/artifacts, figures in ml/reports."
    )


if __name__ == "__main__":
    main()
