"""
Phase 3 gate: is the live forecast credible?

Nothing here is ground truth -- the hours being forecast have not happened. But
NYISO publishes its own zonal load forecast (isolf) for the same hours, produced
by a utility with far more information than we have. If our curve sits close to
theirs, the live pipeline is sane end to end. If it were 30% off, something is
broken and no amount of internal consistency would reveal it.

For scale: on the held-out test set this model scored 2.21% day-ahead against
realised load, and NYISO's own day-ahead error is typically ~2-3%. Two forecasts
of the same thing should therefore agree to within a few percent.

Usage:  python -m scripts.validate_live_forecast
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import live_forecast as lf  # noqa: E402
from app import tft_core as tc  # noqa: E402

ISOLF_ZIP = "http://mis.nyiso.com/public/csv/isolf/{ym}01isolf_csv.zip"
UA = {"User-Agent": "EnergyAI-dashboard/1.0"}
TZ_OFFSET_H = {"EST": 5, "EDT": 4}


def _norm(c: str) -> str:
    return " ".join(str(c).strip().upper().split())


def fetch_isolf(ym: str) -> pd.DataFrame:
    """NYISO zonal load forecast. Monthly ZIP of per-issue-date daily CSVs."""
    r = requests.get(ISOLF_ZIP.format(ym=ym), headers=UA, timeout=90)
    r.raise_for_status()

    rows = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for name in sorted(zf.namelist()):
            if not name.lower().endswith(".csv"):
                continue
            issue = pd.to_datetime(Path(name).name[:8], format="%Y%m%d").date()
            df = pd.read_csv(zf.open(name))
            cols = {_norm(c): c for c in df.columns}
            have = [cols[z] for z in tc.ZONES if z in cols]
            if not have:
                continue

            ts = pd.to_datetime(df[cols["TIME STAMP"]])
            if "TIME ZONE" in cols:
                off = df[cols["TIME ZONE"]].map(TZ_OFFSET_H)
            else:  # no tz column: infer from the date's standing offset
                off = pd.Series(4, index=df.index)
            utc = ts + pd.to_timedelta(off, unit="h")

            total = df[have].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            rows.append(pd.DataFrame({"issue": issue, "ts_utc": utc, "nyiso_mw": total}))

    if not rows:
        raise RuntimeError(f"no usable isolf CSVs in {ym}")
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    ours, meta = lf.load_cached()
    if ours is None:
        print("no cached forecast; run `python -m app.live_forecast` first")
        return 2

    origin = pd.Timestamp(meta["origin"])
    mine = (ours[ours["pred_mw"].notna()]
            .groupby("ts_utc")["pred_mw"].sum()
            .rename("ours_mw"))
    print(f"our forecast : {len(mine)} hours, {mine.index.min()} -> {mine.index.max()} UTC")
    print(f"generated at : {meta['generated_at']}")
    print(f"model        : {meta['model'].get('checkpoint')}")
    print()

    months = sorted({origin.strftime("%Y%m"),
                     (origin + pd.Timedelta(days=6)).strftime("%Y%m")})
    frames = []
    for ym in months:
        try:
            frames.append(fetch_isolf(ym))
        except Exception as exc:
            print(f"isolf {ym}: {type(exc).__name__}: {exc}")
    if not frames:
        print("could not retrieve NYISO isolf; skipping benchmark")
        return 2
    isolf = pd.concat(frames, ignore_index=True)

    # Use the most recent issue that does not postdate our origin -- the same
    # information cutoff we had.
    usable = sorted(d for d in isolf["issue"].unique() if d <= origin.date())
    if not usable:
        print("no NYISO issue at or before our origin")
        return 2
    issue = usable[-1]
    theirs = (isolf[isolf["issue"] == issue]
              .drop_duplicates("ts_utc")
              .set_index("ts_utc")["nyiso_mw"]
              .sort_index())
    print(f"NYISO issue  : {issue}  ({len(theirs)} hours, "
          f"{theirs.index.min()} -> {theirs.index.max()} UTC)")

    cmp = pd.concat([mine, theirs], axis=1).dropna()
    if cmp.empty:
        print("no overlapping hours between the two forecasts")
        return 2

    diff = cmp["ours_mw"] - cmp["nyiso_mw"]
    ape = (diff.abs() / cmp["nyiso_mw"]).mul(100)

    print(f"overlap      : {len(cmp)} hours")
    print()
    print("=" * 60)
    print(f"mean abs deviation from NYISO : {ape.mean():.2f} %")
    print(f"median                        : {ape.median():.2f} %")
    print(f"worst hour                    : {ape.max():.2f} %  at {ape.idxmax()}")
    print(f"mean signed bias              : {diff.mean():+,.0f} MW "
          f"({diff.mean() / cmp['nyiso_mw'].mean() * 100:+.2f} %)")
    print(f"our peak / NYISO peak         : {cmp['ours_mw'].max():,.0f} / "
          f"{cmp['nyiso_mw'].max():,.0f} MW")
    print("=" * 60)
    print()
    print(cmp.head(12).round(0).to_string())

    ok = bool(ape.mean() < 8.0)
    print()
    print("PASS -- live forecast tracks NYISO's own" if ok
          else "FAIL -- live forecast diverges from NYISO, investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
