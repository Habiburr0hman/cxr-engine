# src/plots.py
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def _format_pie_num(pct: float, allvals: list[int] | np.ndarray) -> str:
    """Helper to display absolute count and percentage on pie slices."""
    total = sum(allvals)
    val = round(pct * total / 100.0)
    return f"{val:,}\n({pct:.1f}%)"


def plot_split_dist_pie(
    split_counts: pd.Series | dict[str, int],
    output_path: Path,
    title: str = "Dataset Split Allocation",
    colors: list[str] | None = None,
) -> Path:
    """Renders a single pie chart showing overall train/val/test image distribution."""
    if isinstance(split_counts, dict):
        split_counts = pd.Series(split_counts)

    labels = [str(k).capitalize() for k in split_counts.index]
    data = split_counts.values
    slice_colors = colors or ["#2b5c8f", "#d95f02", "#7570b3"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"aspect": "equal"}, dpi=150)
    wedges, _texts, _autotexts = ax.pie(
        x=data,
        autopct=lambda pct: _format_pie_num(pct, data),
        colors=slice_colors,
        textprops={"color": "white", "weight": "bold", "fontsize": 9},
        pctdistance=0.6,
        startangle=140,
    )

    ax.legend(
        wedges,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(labels),
        fontsize=9,
        frameon=False,
    )
    ax.set_title(title, fontweight="bold", fontsize=12, pad=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_dist_bar(
    split_class_df: pd.DataFrame,
    output_path: Path,
    title: str = "Class Distribution Across Splits",
    palette: tuple[str, str] = ("#00d4ff", "#ff00d4"),
) -> Path:
    """Renders a grouped bar chart with sample counts and percentages above each bar.

    Args:
        split_class_df: DataFrame indexed by split ('train', 'val', 'test') with
                        class columns (e.g. ['Normal', 'Pneumonia']).
        output_path: Destination path for saving PNG.
        title: Plot title.
        palette: Colors for classes.
    """
    splits = [str(s).capitalize() for s in split_class_df.index]
    classes = list(split_class_df.columns)
    n_splits = len(splits)
    n_classes = len(classes)

    x = np.arange(n_splits)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for i, class_name in enumerate(classes):
        counts = split_class_df[class_name].values
        offset = (i - (n_classes - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            counts,
            width=bar_width,
            label=str(class_name).capitalize(),
            color=palette[i % len(palette)],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
        )

        # Annotate count + within-split percentage on top of each bar
        split_totals = split_class_df.sum(axis=1).values
        for bar, count, total in zip(bars, counts, split_totals, strict=False):
            height = bar.get_height()
            pct = (count / total * 100) if total > 0 else 0
            ax.annotate(
                f"{count:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold", fontsize=10)
    ax.set_ylabel("Image Count", fontweight="bold", fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=15)
    ax.legend(title="Class", loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Add headroom for text labels
    y_max = split_class_df.values.max()
    ax.set_ylim(0, y_max * 1.22)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_dist_pie(
    df_dict: dict[str, pd.Series],
    output_path: Path,
    suptitle: str = "Class Balance per Split",
    colors: list[str] | None = None,
) -> Path:
    """Renders 3 side-by-side pie charts (Train, Val, Test)."""
    fig, axes = plt.subplots(
        1,
        len(df_dict),
        figsize=(4 * len(df_dict), 4.2),
        subplot_kw={"aspect": "equal"},
        dpi=150,
    )
    if len(df_dict) == 1:
        axes = [axes]

    fig.suptitle(suptitle, fontweight="bold", fontsize=13, y=1.02)
    slice_colors = colors or ["#00d4ff", "#ff00d4"]

    for i, (split_name, series) in enumerate(df_dict.items()):
        ax = axes[i]
        data = series.values
        labels = [str(lbl).capitalize() for lbl in series.index]

        wedges, _texts, _autotexts = ax.pie(
            x=data,
            autopct=lambda pct, d=data: _format_pie_num(pct, d),
            textprops={"color": "black", "fontsize": 9},
            pctdistance=0.55,
            colors=slice_colors,
            startangle=90,
        )
        ax.set_title(split_name.capitalize(), fontweight="bold", fontsize=11)
        ax.legend(
            wedges,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=len(labels),
            fontsize=8,
            frameon=False,
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_learning_rate(
    lr_arr: list[float] | np.ndarray,
    output_path: Path,
    title: str = "Learning Rate Schedule",
    figsize: tuple[float, float] = (7.0, 4.375),
) -> Path:
    """Plots the learning rate progression across training epochs."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Strictly enforce integer ticks on the epoch axis
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    epochs = range(1, len(lr_arr) + 1)
    ax.plot(epochs, lr_arr, ".-", label="Learning Rate", color="#d62728")

    max_value = max(lr_arr) if len(lr_arr) > 0 else 1e-3
    n_epochs = len(lr_arr)

    ax.legend(frameon=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Learning Rate", fontweight="bold")
    ax.set_xlim([-0.02 * n_epochs, 1.02 * n_epochs])
    ax.set_ylim([-0.02 * max_value, 1.05 * max_value])
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_training_log(
    train_arr: list[float] | np.ndarray,
    val_arr: list[float] | np.ndarray | None,
    title: str,
    output_path: Path,
    metric_label: str = "Value",
    figsize: tuple[float, float] = (7.0, 4.375),
) -> Path:
    """Plots a single training vs. validation metric curve across epochs."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    # Strictly enforce integer ticks on the epoch axis
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    epochs = range(1, len(train_arr) + 1)
    ax.plot(epochs, train_arr, ".-", label="Train", color="#1f77b4")

    all_vals = list(train_arr)
    if val_arr is not None and len(val_arr) > 0:
        val_epochs = range(1, len(val_arr) + 1)
        ax.plot(val_epochs, val_arr, ".-", label="Validation", color="#ff7f0e")
        all_vals.extend(val_arr)

    ax.legend(frameon=True)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(metric_label, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)

    max_value = max(all_vals) if all_vals else 1.0
    n_epochs = len(train_arr)
    ax.set_xlim([-0.02 * n_epochs, 1.02 * n_epochs])
    ax.set_ylim([-0.02 * max_value, 1.02 * max_value])

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dual_training_log(
    history_dict: dict[str, list[float]],
    metric1_name: str,
    metric1_label: str,
    metric2_name: str,
    metric2_label: str,
    output_path: Path,
    suptitle: str = "Binary Model Training Log",
) -> Path:
    """Plots a side-by-side training log for loss and a target metric (e.g. F1)."""
    train_m1 = history_dict.get(metric1_name, [])
    val_m1 = history_dict.get(f"val_{metric1_name}", [])
    train_m2 = history_dict.get(metric2_name, [])
    val_m2 = history_dict.get(f"val_{metric2_name}", [])

    epochs = range(1, len(train_m1) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Panel 1: Metric 1 (typically Loss)
    axes[0].plot(epochs, train_m1, label="Train", color="#1f77b4")
    if val_m1:
        axes[0].plot(epochs, val_m1, label="Validation", color="#ff7f0e")
    axes[0].set_title(metric1_label, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Panel 2: Metric 2 (typically F1 or AUC)
    axes[1].plot(epochs, train_m2, label="Train", color="#1f77b4")
    if val_m2:
        axes[1].plot(epochs, val_m2, label="Validation", color="#ff7f0e")
    axes[1].set_title(metric2_label, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    fig.suptitle(suptitle, fontweight="bold", fontsize=14)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    subtitle: str = "",
    cmap: str = "cool",
    figsize: tuple[float, float] = (4.0, 3.8),
) -> Path:
    """Renders confusion matrix with optional scientific notation on colorbar."""
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    p = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        annot_kws={"fontweight": "bold", "fontsize": 10},
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"pad": 0.08},
        ax=ax,
    )

    # Scientific notation formatter on colorbar
    cbar = p.collections[0].colorbar
    if cbar is not None:
        cbar.ax.yaxis.get_offset_text().set_horizontalalignment("left")
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        cbar.ax.yaxis.set_major_formatter(formatter)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xlabel("Predicted Class", fontweight="bold", fontsize=9)
    ax.set_ylabel("Actual Class", fontweight="bold", fontsize=9)

    if subtitle:
        fig.suptitle(title, fontweight="bold", fontsize=11, y=1.02)
        ax.set_title(subtitle, fontweight="bold", fontsize=9)
    else:
        ax.set_title(title, fontweight="bold", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    thresholds: np.ndarray,
    output_path: Path,
    title: str = "Receiver Operating Characteristic (ROC)",
    subtitle: str = "",
    default_threshold: float = 0.50,
    optimal_threshold: float | None = None,
) -> Path:
    """Renders ROC curve highlighting default and Youden's J optimal threshold."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.plot(
        fpr,
        tpr,
        color="#1f77b4",
        lw=2,
        label=f"Model (AUC = {roc_auc:.2f})",
    )
    ax.plot(
        [0, 1],
        [0, 1],
        color="red",
        lw=1.5,
        linestyle="--",
        label="Random (AUC = 0.50)",
    )

    clean_thresholds = np.where(np.isinf(thresholds), 999999, thresholds)

    # 1. Default threshold point
    idx_def = int(np.argmin(np.abs(clean_thresholds - default_threshold)))
    ax.scatter(
        fpr[idx_def],
        tpr[idx_def],
        color="black",
        s=45,
        zorder=5,
        label=f"Default Thresh ({default_threshold:.2f})",
    )

    # 2. Optimal threshold point (Youden's J)
    if optimal_threshold is not None:
        idx_opt = int(np.argmin(np.abs(clean_thresholds - optimal_threshold)))
        ax.scatter(
            fpr[idx_opt],
            tpr[idx_opt],
            color="green",
            s=45,
            zorder=6,
            label=f"Youden J Thresh ({optimal_threshold:.2f})",
        )

    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlabel("FPR = 1 - Specificity", fontsize=9)
    ax.set_ylabel("TPR = Sensitivity", fontsize=9)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)

    if subtitle:
        fig.suptitle(title, fontweight="bold", fontsize=11, y=0.98)
        ax.set_title(subtitle, fontweight="bold", fontsize=9)
    else:
        ax.set_title(title, fontweight="bold", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pr_curve(
    recalls: np.ndarray,
    precisions: np.ndarray,
    pr_auc: float,
    thresholds: np.ndarray,
    output_path: Path,
    baseline_prevalence: float | None = None,
    title: str = "Precision-Recall Curve",
    subtitle: str = "",
    default_threshold: float = 0.50,
    optimal_threshold: float | None = None,
) -> Path:
    """Renders PR curve highlighting baseline prevalence and optimal operating points."""
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    ax.plot(
        recalls,
        precisions,
        color="#2ca02c",
        lw=2,
        label=f"Model (AUC = {pr_auc:.2f})",
    )

    if baseline_prevalence is not None:
        ax.axhline(
            y=baseline_prevalence,
            color="red",
            lw=1.5,
            linestyle="--",
            label=f"Baseline ({baseline_prevalence:.2f})",
        )

    # Match thresholds to (recall, precision) pairs
    # Note: scikit-learn precision_recall_curve returns thresholds of len(n_classes) - 1
    if len(thresholds) > 0:

        def draw_point(thresh_val: float, color: str, label_prefix: str):
            idx = int(np.argmin(np.abs(thresholds - thresh_val)))
            r_pt, p_pt = recalls[idx], precisions[idx]
            ax.scatter(
                r_pt,
                p_pt,
                color=color,
                s=45,
                zorder=5,
                label=f"{label_prefix} ({thresh_val:.2f})",
            )

        draw_point(default_threshold, "black", "Default Thresh")
        if optimal_threshold is not None:
            draw_point(optimal_threshold, "green", "Optimal Thresh")

    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlabel("Recall / Sensitivity (TPR)", fontsize=9)
    ax.set_ylabel("Precision / PPV", fontsize=9)
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    ax.set_yticks(np.arange(0, 1.1, 0.2))
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, linestyle="--", alpha=0.5)

    if subtitle:
        fig.suptitle(title, fontweight="bold", fontsize=11, y=0.98)
        ax.set_title(subtitle, fontweight="bold", fontsize=9)
    else:
        ax.set_title(title, fontweight="bold", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def render_report_image(
    report_text: str,
    output_path: Path,
    title: str = "Classification Report",
) -> Path:
    """Renders monospace classification report text directly to an image file."""
    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=150)
    fig.patch.set_facecolor("#222222")
    ax.set_facecolor("#222222")
    ax.axis("off")

    ax.text(
        0.05,
        0.90,
        title,
        color="white",
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
        transform=ax.transAxes,
    )
    ax.text(
        0.05,
        0.10,
        report_text,
        color="#e0e0e0",
        fontsize=9,
        fontfamily="monospace",
        va="bottom",
        transform=ax.transAxes,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    return output_path
