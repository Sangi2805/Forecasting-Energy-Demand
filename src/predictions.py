from pathlib import Path

import pandas as pd


def save_combined_predictions_csv(
    horizon_frames: dict[int, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    """Merge per-horizon frames into one CSV matching the XGBoost layout."""
    combined = None
    for horizon in sorted(horizon_frames):
        frame = horizon_frames[horizon][["date", "y", "yhat"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.rename(
            columns={"y": f"actual_day{horizon}", "yhat": f"pred_day{horizon}"}
        )
        combined = frame if combined is None else combined.merge(frame, on="date", how="inner")

    combined = combined.sort_values("date").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined
