# Presentation Script — Group 1, Iteration 1 (~15 min)

**Deck:** `Capstone_Iteration1_Presentation.pptx` (slides 1–24 main · 25+ appendix)  
**Audience:** Technical team

---

## SLIDE 1 — Title
**[Whoever opens — ~15 sec]**

> "Good [morning/afternoon]. We're Group 1, and today we're presenting Iteration 1 of our capstone: *End-to-End Multivariate Time Series Forecasting for Energy Demand* for New York State. I'll hand it over to Sai to kick us off."

---

## SLIDE 2 — Progress
**Sai Susanth · ~1.5 min**

> "Thanks. Let me quickly set the context for what we set out to do and where we stand."

> "For Iteration 1, our goal was to build a solid foundation — get the data, understand it, train baseline models, and hook it all together into a working UI."

> "**Completed:** we collected demand, weather, and holiday data; ran a full preprocessing pipeline; froze a shared train/test dataset for the whole team; trained four baseline models — XGBoost, LightGBM, Prophet, and SARIMAX; tracked experiments with MLflow; and built a live Streamlit dashboard."

> "**Deferred:** the LSTM deep-learning model, a properly tuned SARIMAX, and two engineered features identified from EDA — an explicit V-shape feature `|temp − 14°C|` and a soft snowfall flag."

> "The honest reason: getting four models trained and compared on the same data, while keeping a five-person team in sync, took the full iteration. These move to Iteration 2 with clear justification."

> "I'll pass to Shahriar to walk through how we built the data pipeline."

---

## SLIDE 3 — Data & preprocessing
**Shahriar · ~2 min**

> "Thanks Sai."

> "We pulled data from several sources. Electricity demand came from the **EIA Grid Monitor** — hourly demand in megawatts for New York State, from July 2015 to May 2026. Weather came from the **Open-Meteo API** — temperature, apparent temperature, humidity, snowfall, precipitation, cloud cover, and wind. We enriched that with a **US holiday calendar**, plus New York State **GDP and population** data."

> "The preprocessing pipeline merges everything on a common timestamp, engineers features — day-of-week, month, season, holiday flags, and 1–2 day weather lags — and produces a model-ready daily dataset aligned with our **3-day-ahead** forecast task."

> "One critical decision: we froze the train/test split at an **80/20 chronological hold-out** — the first 80% of dates for training, the last 20% for testing. No random shuffle, so there's no leakage. Every team member trained against the same frozen files, versioned with **DVC**."

> "I'll hand over to Toan for what the data told us."

---

## SLIDES 4–6 — EDA key findings
**Toan · ~1.5 min**

> "Thank you Shahriar."

> "Before we chose our models, we spent time understanding the data. Four things stood out."

> "**First — time features dominate.** Season, month, and day-of-week all have a major impact. Weekdays are consistently higher than weekends; summer and winter peaks bookend the year."

> "**Second — temperature is V-shaped and non-linear.** Demand is highest at cold and hot extremes, and lowest in the roughly **10–18°C** comfort zone. A model needs to capture non-linearity, not just a straight correlation."

> "**Third — snowfall days show elevated demand.** People stay home; heating runs longer."

> "**Fourth — holidays introduce measurable shifts.** Days like New Year's and Thanksgiving change the load mix — commercial and industrial drop even when residential rises."

> "These four findings directly shaped our feature choices. I'll hand back to Sangar and the team to explain why we chose the models we did."

*(Slides 4–5 expand on patterns and weather; slide 6 is the four summary cards — speak to slide 6, glance at 4–5 if time.)*

---

## SLIDES 7–10 — Pipeline setup *(brief)* + Model choices
**Shahriar (slides 7–9, ~45 sec) · Sangar + Shakeel + Shahriar (model choices, ~1.5 min)**

**Shahriar — slides 7–9 (optional quick bridge):**

> "Quick pipeline note: architecture is source data → preprocessing → DVC on DagsHub → model training with MLflow → prediction CSVs → Streamlit. We had a DVC push mirroring issue on DagsHub — Shakeel will cover that at the end."

**Sangar + Shakeel + Shahriar — model choices (slide 10 or stand at slide 11 section break):**

> "Thanks Toan. We chose four model families deliberately, to cover the main approaches."

> "**XGBoost and LightGBM** — [Sangar] gradient-boosted tree ensembles. These are our primary models. They handle non-linear relationships well — important for the V-shaped temperature effect Toan described. They scale to many features and are interpretable through feature importance."

> "**Prophet** — [Shakeel] Facebook's time-series library. It captures seasonality and trend automatically and accepts external regressors like weather. Our classical time-series baseline."

> "**SARIMAX** — [Shahriar] statistical ARMA with seasonality and exogenous variables. The most traditional approach — a comparison point against ML models."

> "Together: tree ensembles, prophet-style forecasting, and classical statistics. LSTM is deferred to Iteration 2. I'll continue with the results."

---

## SLIDES 11–14 — Results
**Sangar + Shahriar · ~2.5 min**

**Sangar — slides 11–13:**

> "Now for results. Before the numbers: on each day in our test set — **878 days from January 2024 to May 2026** — each model makes three predictions: demand for the next day, two days ahead, and three days ahead. We call these Day 1, Day 2, Day 3."

> "We report **MAE** (average absolute error in MW), **RMSE** (penalises large misses), and **MAPE** (error as a percentage of actual demand), averaged across the three horizons."

> *(Slide 12 — chart)* "The bar chart tells the story. **XGBoost and LightGBM cluster around ~29,600 MW average RMSE.** Prophet is slightly worse at ~32,000. SARIMAX is a clear outlier at ~97,000."

> *(Slide 13 — table)* "On the leaderboard: **XGBoost is our overall winner on RMSE** — 21,037 MW MAE and **29,617 MW RMSE**, with Day 1 MAPE around **3.7%**. **LightGBM is essentially tied** — best MAPE at 5.79% versus XGBoost's 5.93%, RMSE only ~180 MW higher. Prophet performs reasonably for an automated seasonal model."

**Shahriar — slide 13–14:**

> "On **SARIMAX** — an important honest finding. In-sample, SARIMAX looked strong (~15.5K MAE). Out-of-sample, it collapsed to **96,838 MW RMSE**. It leaned almost entirely on the **`is_holiday` flag** — we'll see that in feature importance. It overfit holiday patterns and failed to generalise. Clear Iteration 2 fix."

> "What worked: tree models capturing non-linear demand and weather. What didn't: high-dimensional SARIMAX with our exog set. Sangar will show why the features tell the same story."

---

## SLIDES 19–22 — Feature importance ↔ EDA alignment
**Sangar · ~1.5 min**

> "The reason XGBoost worked shows up in feature importance."

> *(Slide 20)* "Top drivers are **demand history** — rolling 7-day mean and lags — plus **`apparent_temperature` at rank #2 (~12% contribution)** and **`day_of_week` in the top 5**. That matches EDA: autocorrelation plus temperature and calendar effects."

> *(Slide 21 — matrix)* "Across all four models: temperature and demand history dominate for XGBoost, LightGBM, and Prophet. **Prophet** ranks apparent temperature **#1**. **SARIMAX** is the outlier — **`is_holiday` at ~98% importance**, which explains the poor generalisation."

> "One flag on **LightGBM**: **`total_interchange` ranks #6** — potentially leaky; we need to review that for Iteration 2."

> "Two honest **gaps**: snowfall mattered in EDA but isn't in any model's top features yet. We also never engineered **`|temp − 14°C|`** to encode the V-shape directly. Both are Iteration 2 items."

> "This is post-hoc validation — EDA ran before final feature selection — but it confirms the models largely learned the right signals. Let me show the dashboard."

*(Slide 22 — 2×2 detail grid: optional; skip if short on time.)*

---

## SLIDES 15–18 — UI demo
**Sangar · ~1 min**

> *(Slide 15 — section break)* "We built a Streamlit dashboard so anyone can interact with forecasts without touching code."

> *(Slide 16)* "Select a model — **XGBoost pre-selected** as best on RMSE. You get a **3-day forecast table** (actual vs predicted for each horizon), **MAE / RMSE / MAPE** for that window, and an **actual-vs-predicted chart**. Expand **Compare all models** for the full test-set leaderboard and an overlay of every model."

> *(Slide 17 — screenshot or live demo)*  
> **Live:** `streamlit run app/streamlit_app.py` — show model selector, demand table, metrics, chart; expand compare panel.  
> **Screenshot fallback:** walk through slide 17 top-to-bottom.

> *(Slide 18 — optional)* "Overlay view: actual in green; tree models track closely; SARIMAX diverges."

> "Iteration 2: live data feed, ensemble, confidence intervals. I'll hand to Shakeel for challenges and the plan forward."

---

## SLIDE 23 — Challenges + Iteration 2
**Shakeel · ~1.5 min**

> "Thanks Sangar."

> "Three honest challenges from this iteration."

> "**First — team data sync.** Five people, DagsHub, DVC, and Git. `dvc push` to DagsHub broke on a mirroring issue; some teammates relied on local copies. That compromised the workflow we designed."

> "**Second — the SARIMAX evaluation trap.** In-sample metrics looked convincing; out-of-sample exposed overfitting. That cost iteration time before we traced it to holiday over-reliance."

> "**Third — not everything is trained yet.** LSTM is deferred; feature selection is still in progress."

> "**Iteration 2 priorities:** finish LSTM and tune SARIMAX; fix the DVC remote for a single push-pull workflow; complete feature selection including V-shape and snowfall features; integrate the best model or ensemble into a production pipeline; expand the UI with confidence intervals and explainability."

> "Iteration 1 established the baseline — tree models work, we know what features matter, and we know the gaps. Iteration 2 closes them and moves toward deployment."

> "Thank you. We're happy to take questions."

---

## SLIDE 24 — Thank you
**Anyone · ~10 sec**

> "Thank you — questions welcome. Detailed EDA charts and per-model alignment are in the appendix."

---

## Speaker timing summary

| Speaker | Slides | Topic | ~Time |
|---------|--------|-------|-------|
| Opener | 1 | Title | 0:15 |
| **Sai** | 2 | Progress | 1:30 |
| **Shahriar** | 3, 7–9 (brief) | Data + pipeline | 2:30 |
| **Toan** | 4–6 | EDA findings | 1:30 |
| **Sangar, Shakeel, Shahriar** | 10–11 | Model choices | 1:30 |
| **Sangar & Shahriar** | 11–14 | Results | 2:30 |
| **Sangar** | 19–22 | Feature importance / EDA | 1:30 |
| **Sangar** | 15–18 | UI demo | 1:00 |
| **Shakeel** | 23 | Challenges + Iter 2 | 1:30 |
| All | 24 | Q&A | — |

**Total:** ~14–15 min (skip slides 18, 22 if running long)

---

## Fact-check card (use if questioned)

| Claim | Verified value |
|-------|----------------|
| Test set size | 878 days (2024-01-01 → 2026-05-27) |
| XGBoost avg RMSE / MAE | **29,617 MW** / **21,037 MW** |
| LightGBM avg RMSE / MAE | **29,799 MW** / **20,877 MW** |
| Prophet avg RMSE | **32,048 MW** |
| SARIMAX avg RMSE | **96,838 MW** |
| XGBoost Day 1 MAPE | ~**3.7%** |
| `apparent_temperature` (XGBoost) | Rank **#4**, ~**11.8%** contribution |
| LightGBM `total_interchange` | Combined rank **#6** (gain rank #3) |
| UI horizon control | **3-day window shown together** — no single-horizon toggle |

---

## Demo checklist (Sangar)

- [ ] `streamlit run app/streamlit_app.py` running before slide 15
- [ ] XGBoost pre-selected in model dropdown
- [ ] Expand **Compare all models (full test set)**
- [ ] Fallback: slides 17–18 if live demo fails
