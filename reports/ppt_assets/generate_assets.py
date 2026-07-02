"""Generate charts and UI screenshots for the capstone Iteration 1 PPT."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "reports"
ASSET_DIR = REPORT_DIR / "ppt_assets"
sys.path.insert(0, str(BASE_DIR))

from app.streamlit_app import (  # noqa: E402
    MODEL_COLORS,
    build_three_day_series,
    compute_metrics,
    latest_issue_date,
    load_predictions,
    plot_all_models_forecast,
    plot_single_model_forecast,
)

ASSET_DIR.mkdir(parents=True, exist_ok=True)

BRIGHT = {
    "bg": "#F8FAFF",
    "primary": "#2563EB",
    "secondary": "#F97316",
    "accent": "#10B981",
    "purple": "#8B5CF6",
    "pink": "#EC4899",
    "text": "#1E293B",
    "muted": "#64748B",
}

MODELS = {
    "LightGBM": REPORT_DIR / "lgbm_predictions.csv",
    "XGBoost": REPORT_DIR / "xgboost_predictions.csv",
    "Prophet": REPORT_DIR / "prophet_predictions.csv",
    "SARIMAX": REPORT_DIR / "sarimax_predictions.csv",
}


def save_plotly(fig: go.Figure, name: str, width: int = 1200, height: int = 520) -> Path:
    out = ASSET_DIR / name
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#FAFBFF",
        font=dict(family="Arial", size=14, color=BRIGHT["text"]),
        title_font=dict(size=20, color=BRIGHT["primary"]),
    )
    try:
        fig.write_image(str(out), width=width, height=height, scale=2)
    except Exception:
        fig.write_html(str(out.with_suffix(".html")))
        # fallback matplotlib bar if kaleido missing
        print(f"Warning: saved HTML fallback for {name}")
    return out


def chart_model_comparison() -> Path:
    rows = []
    for model, path in MODELS.items():
        if not path.exists():
            continue
        m = compute_metrics(load_predictions(str(path)))
        rows.append(
            {
                "model": model,
                "avg_rmse": m["avg_rmse"],
                "avg_mae": m["avg_mae"],
                "avg_mape": m["avg_mape"],
                "day1_rmse": m["day1_rmse"],
                "day2_rmse": m["day2_rmse"],
                "day3_rmse": m["day3_rmse"],
            }
        )
    df = pd.DataFrame(rows).sort_values("avg_rmse")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Average RMSE (MW) — lower is better", "RMSE by forecast horizon"),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
    )
    colors = [MODEL_COLORS.get(m, BRIGHT["primary"]) for m in df["model"]]
    fig.add_trace(
        go.Bar(
            x=df["model"],
            y=df["avg_rmse"],
            marker_color=colors,
            text=[f"{v:,.0f}" for v in df["avg_rmse"]],
            textposition="outside",
            name="Avg RMSE",
        ),
        row=1,
        col=1,
    )
    for horizon, col in [(1, "day1_rmse"), (2, "day2_rmse"), (3, "day3_rmse")]:
        fig.add_trace(
            go.Bar(
                x=df["model"],
                y=df[col],
                name=f"Day {horizon}",
                marker_color=[MODEL_COLORS.get(m, "#999") for m in df["model"]],
                opacity=0.55 + horizon * 0.12,
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        title="Model comparison on held-out test set (3-day ahead)",
        barmode="group",
        height=520,
        showlegend=True,
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_yaxes(title_text="RMSE (MW)")
    return save_plotly(fig, "model_comparison.png", height=560)


def chart_feature_importance() -> Path:
    fi = pd.read_csv(REPORT_DIR / "xgboost_feature_importance.csv")
    top = fi.head(12).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=top["contribution_pct"],
            y=top["feature"],
            orientation="h",
            marker=dict(
                color=top["contribution_pct"],
                colorscale=[[0, "#93C5FD"], [0.5, "#3B82F6"], [1, "#1D4ED8"]],
            ),
            text=[f"{v:.2f}%" for v in top["contribution_pct"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="XGBoost top feature importance (gain, averaged day1–3)",
        xaxis_title="Contribution (%)",
        height=560,
        margin=dict(l=160),
    )
    return save_plotly(fig, "xgboost_feature_importance.png", height=560)


EDA_CHECKS = [
    {
        "label": "Time patterns",
        "features": ["day_of_week", "season", "month", "month_sin", "month_cos", "week_of_year"],
        "eda": "Season, month, DOW & hour drive demand",
    },
    {
        "label": "V-shape temperature",
        "features": ["apparent_temperature", "temperature_2m", "temp_max", "temp_min", "hdd", "cdd"],
        "eda": "Demand lowest ~10–18°C",
    },
    {
        "label": "Snowfall signal",
        "features": ["snowfall", "snow_depth"],
        "eda": "Higher demand on snow days",
    },
    {
        "label": "Holiday shifts",
        "features": ["is_holiday"],
        "eda": "Measurable holiday deviations",
    },
]

STATUS_STYLE = {
    "Aligned": {"color": "#10B981", "code": 3},
    "Partial": {"color": "#F97316", "code": 2},
    "Gap": {"color": "#EF4444", "code": 1},
    "Future": {"color": "#8B5CF6", "code": 0},
}

MODEL_FI_FILES = {
    "XGBoost": "xgboost_feature_importance.csv",
    "LightGBM": "lgbm_feature_importance.csv",
    "Prophet": "prophet_feature_importance.csv",
    "SARIMAX": "sarimax_feature_importance.csv",
}


def _rank_lookup(fi: pd.DataFrame) -> dict[str, int]:
    return dict(zip(fi["feature"], fi["combined_rank"].astype(int)))


def _pct_lookup(fi: pd.DataFrame) -> dict[str, float]:
    return dict(zip(fi["feature"], fi["contribution_pct"].astype(float)))


def _assess_eda_row(ranks: dict[str, int], features: list[str]) -> tuple[str, str]:
    hits = [(f, ranks[f]) for f in features if f in ranks]
    if not hits:
        return "Gap", "not in model"
    best_feat, best_rank = min(hits, key=lambda x: x[1])
    if best_rank <= 8:
        return "Aligned", f"{best_feat} #{best_rank}"
    if best_rank <= 20:
        return "Partial", f"{best_feat} #{best_rank}"
    return "Gap", f"{best_feat} #{best_rank}"


def eda_alignment_for_model(model: str) -> list[tuple[str, str, str, str]]:
    path = REPORT_DIR / MODEL_FI_FILES[model]
    fi = pd.read_csv(path)
    ranks = _rank_lookup(fi)
    rows = []
    for check in EDA_CHECKS:
        status, detail = _assess_eda_row(ranks, check["features"])
        rows.append((check["label"], check["eda"], detail, status))
    rows.append(("|temp − 14°C| feature", "Explicit V-shape", "not tested", "Future"))
    rows.append(("Soft snowfall flag", "EDA snowfall signal", "not tested", "Future"))
    return rows


def _draw_alignment_panel(ax, model: str, rows: list, title_color: str) -> None:
    ax.set_facecolor("#FAFBFF")
    status_colors = {k: v["color"] for k, v in STATUS_STYLE.items()}
    y = np.arange(len(rows))
    for i, (label, _eda, detail, status) in enumerate(rows):
        ax.barh(i, 1, color=status_colors.get(status, "#CBD5E1"), alpha=0.18, height=0.72)
        ax.text(0.02, i, label, va="center", fontsize=9.5, fontweight="bold", color=BRIGHT["text"])
        ax.text(0.46, i, detail, va="center", fontsize=8.8, color=BRIGHT["muted"])
        ax.text(0.98, i, status, va="center", ha="right", fontsize=8.8, fontweight="bold", color=status_colors.get(status, "#64748B"))
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(model, fontsize=13, fontweight="bold", color=title_color, pad=8, loc="left")


def chart_eda_alignment_all() -> list[Path]:
    outputs: list[Path] = []
    model_rows = {m: eda_alignment_for_model(m) for m in MODEL_FI_FILES}

    # Combined 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="white")
    model_colors = {
        "XGBoost": BRIGHT["secondary"],
        "LightGBM": BRIGHT["primary"],
        "Prophet": BRIGHT["purple"],
        "SARIMAX": BRIGHT["accent"],
    }
    for ax, model in zip(axes.flat, MODEL_FI_FILES):
        _draw_alignment_panel(ax, model, model_rows[model], model_colors[model])
    fig.suptitle("EDA alignment across models", fontsize=17, fontweight="bold", color=BRIGHT["primary"], y=0.98)
    patches = [mpatches.Patch(color=v["color"], alpha=0.65, label=k) for k, v in STATUS_STYLE.items()]
    fig.legend(handles=patches, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.01), fontsize=10)
    out_all = ASSET_DIR / "eda_alignment_all_models.png"
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out_all, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(out_all)

    # Matrix heatmap summary
    findings = [r[0] for r in model_rows["XGBoost"]]
    models = list(MODEL_FI_FILES.keys())
    codes = np.zeros((len(findings), len(models)))
    detail_matrix = [[""] * len(models) for _ in findings]
    for j, model in enumerate(models):
        for i, row in enumerate(model_rows[model]):
            codes[i, j] = STATUS_STYLE[row[3]]["code"]
            detail_matrix[i][j] = row[2]

    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor="white")
    cmap = plt.matplotlib.colors.ListedColormap(["#EDE9FE", "#FCA5A5", "#FDBA74", "#6EE7B7"])
    im = ax.imshow(codes, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(findings)))
    ax.set_yticklabels(findings, fontsize=10)
    for i in range(len(findings)):
        for j in range(len(models)):
            ax.text(j, i, detail_matrix[i][j], ha="center", va="center", fontsize=7.5, color=BRIGHT["text"])
    ax.set_title("EDA finding × model importance rank", fontsize=15, fontweight="bold", color=BRIGHT["primary"], pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    out_matrix = ASSET_DIR / "eda_alignment_matrix.png"
    fig.tight_layout()
    fig.savefig(out_matrix, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    outputs.append(out_matrix)

    # Per-model panels for appendix
    for model in MODEL_FI_FILES:
        fig, ax = plt.subplots(figsize=(11, 4.8), facecolor="white")
        _draw_alignment_panel(ax, model, model_rows[model], model_colors[model])
        fig.suptitle(f"EDA alignment — {model}", fontsize=14, fontweight="bold", color=BRIGHT["primary"])
        out = ASSET_DIR / f"eda_alignment_{model.lower()}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs.append(out)

    return outputs


def chart_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")
    ax.set_facecolor(BRIGHT["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.18,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, fontweight="bold", color=BRIGHT["text"])

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color=BRIGHT["muted"], lw=2))

    box(0.3, 4.5, 1.8, 1.0, "EIA Grid\nMonitor", BRIGHT["primary"])
    box(0.3, 3.0, 1.8, 1.0, "Open-Meteo\nAPI", BRIGHT["secondary"])
    box(0.3, 1.5, 1.8, 1.0, "Holidays /\nGDP / Pop", BRIGHT["purple"])
    box(2.6, 2.5, 2.0, 1.6, "preprocessing.py\nMerge · Engineer · Encode", BRIGHT["accent"])
    box(5.2, 2.5, 1.8, 1.6, "DVC + DagsHub\nFrozen train/test CSV", BRIGHT["pink"])
    box(7.4, 4.2, 2.2, 0.9, "XGBoost / LightGBM", BRIGHT["primary"])
    box(7.4, 3.0, 2.2, 0.9, "Prophet / SARIMAX", BRIGHT["secondary"])
    box(7.4, 1.8, 2.2, 0.9, "MLflow on DagsHub", BRIGHT["purple"])
    box(7.4, 0.5, 2.2, 0.9, "Streamlit UI", BRIGHT["accent"])

    arrow(2.1, 5.0, 2.6, 3.5)
    arrow(2.1, 3.5, 2.6, 3.2)
    arrow(2.1, 2.0, 2.6, 2.8)
    arrow(4.6, 3.3, 5.2, 3.3)
    arrow(7.0, 3.5, 7.4, 4.6)
    arrow(7.0, 3.3, 7.4, 3.4)
    arrow(7.0, 3.1, 7.4, 2.2)
    arrow(8.5, 4.2, 8.5, 1.4)

    ax.set_title("End-to-end pipeline architecture", fontsize=16, color=BRIGHT["primary"], pad=12)
    out = ASSET_DIR / "architecture.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def screenshot_ui_views() -> list[Path]:
    paths = []
    catalog_rows = []
    predictions = {}
    for model, p in MODELS.items():
        if not p.exists():
            continue
        df = load_predictions(str(p))
        predictions[model] = df
        catalog_rows.append({"model": model, **compute_metrics(df)})
    catalog = pd.DataFrame(catalog_rows).sort_values("avg_rmse")
    best = catalog.iloc[0]["model"]
    issue = latest_issue_date(predictions[best])
    series = build_three_day_series(predictions[best], issue)

    fig1 = plot_single_model_forecast(series, best)
    fig1.update_layout(title=f"UI: {best} — Actual vs Predicted (issue {issue.strftime('%Y-%m-%d')})")
    paths.append(save_plotly(fig1, "ui_single_model.png"))

    fig2 = plot_all_models_forecast(predictions, issue)
    fig2.update_layout(title=f"UI: All models vs actual (issue {issue.strftime('%Y-%m-%d')})")
    paths.append(save_plotly(fig2, "ui_all_models.png"))

    # metrics table image
    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor="white")
    ax.axis("off")
    table_df = catalog[["model", "avg_mae", "avg_rmse", "avg_mape"]].copy()
    table_df.columns = ["Model", "Avg MAE (MW)", "Avg RMSE (MW)", "Avg MAPE (%)"]
    table_data = []
    for _, row in table_df.iterrows():
        table_data.append(
            [
                row["Model"],
                f"{row['Avg MAE (MW)']:,.0f}",
                f"{row['Avg RMSE (MW)']:,.0f}",
                f"{row['Avg MAPE (%)']:.2f}",
            ]
        )
    table = ax.table(
        cellText=table_data,
        colLabels=list(table_df.columns),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.6)
    best_model = catalog.iloc[0]["model"]
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(BRIGHT["primary"])
            cell.set_text_props(color="white", fontweight="bold")
        elif r > 0 and table_data[r - 1][0] == best_model:
            cell.set_facecolor("#CCFBF1")
    ax.set_title("UI: Compare all models (full test set)", fontsize=14, color=BRIGHT["primary"], pad=10)
    out = ASSET_DIR / "ui_metrics_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    paths.append(out)
    return paths


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    outputs = [
        chart_model_comparison(),
        chart_feature_importance(),
        *chart_eda_alignment_all(),
        chart_architecture(),
        *screenshot_ui_views(),
    ]
    print("Generated assets:")
    for p in outputs:
        print(" ", p)


if __name__ == "__main__":
    main()
