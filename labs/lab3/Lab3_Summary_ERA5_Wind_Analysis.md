# Lab 3 Summary: ERA5 Wind Analysis (Berlin vs Munich, UTC)

## Findings (3–5 sentences)
Using ERA5 10 m wind components for Berlin and Munich, I computed wind speed as \(\sqrt{u10m^2 + v10m^2}\) and analyzed all timestamps in **UTC**. Monthly means show clear variability, with Munich and Berlin exhibiting comparable magnitudes but different month-to-month patterns. Seasonal aggregation (DJF/MAM/JJA/SON) highlights noticeable differences between warm and cold seasons; note that **DJF may be partially represented** depending on the data window. The diurnal cycle (UTC) reveals modest hour‑of‑day structure, with peaks and lulls varying by city; this should be interpreted cautiously since local time effects are intentionally not included. Overall distributional statistics are summarized below.

## Key Stats (10 m wind speed, m/s)
| City   | Mean | p95  | Max  | N   |
|--------|-----:|-----:|-----:|----:|
| Berlin | 3.312 | 6.241 | 8.547 | 1100 |
| Munich | 2.502 | 5.237 | 9.055 | 1100 |

> Notes: “p95” is the 95th percentile of hourly wind speed. Counts reflect rows retained after cleaning.

## How I Might Use Skyrim in a Civil/Environmental Project
**Skyrim** is an open‑source toolkit that streamlines access to numerical weather prediction and forecast data (e.g., downloading, organizing, and programmatically querying model outputs). In a civil/environmental engineering context, I would use Skyrim to automate retrieval of wind (and other meteorological) forecasts for **construction scheduling**, **crane operation safety envelopes**, and **wind loading checks** on temporary structures. By integrating Skyrim into a pipeline with **pandas** and **matplotlib**, I could pull fresh forecasts, compute **site‑specific wind exceedance probabilities**, and trigger **alerts/dashboards** that inform field teams. This approach reduces manual data handling, increases reproducibility, and supports better **risk‑aware decision‑making** on job sites.

## Artifacts Produced
- **Scripts:** `labs/lab3/lab3_era5_analysis_v2.py`
- **CSVs:** `city_wind_speed_stats.csv`, `monthly_wind_speed.csv`, `seasonal_wind_speed.csv`, `diurnal_wind_speed_utc.csv`, `top5_hourly_extremes.csv`
- **Figures:** `fig_monthly_wind_speed.png`, `fig_seasonal_wind_speed.png`, `fig_diurnal_wind_speed_utc.png`

## Notes
- All computations use **UTC**, per instruction.
- Only **wind‑speed aggregates** were computed (no temperature), per instruction.
- Seasonal means may represent **partial seasons** if the dataset does not cover a complete year.
- Extreme dates/times are listed in `top5_hourly_extremes.csv`; you can cite notable events from public sources for those timestamps.

## Links (to include in your email)
- **GitHub (lab folder):** _paste the link to `labs/lab3/` here_
- **Skyrim repo (starred):** https://github.com/secondlaw-ai/skyrim
