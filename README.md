# CE49X-Labs
CE49X Lab Assignments
CE49X Labs

Bu depo, CE49X: Introduction to Computational Thinking and Data Science dersi için hazırlanan laboratuvar çalışmalarını içerir.

Contents

## Lab01 — Energy Calculator (summary)

- **Purpose:** Compute, classify, and analyze building energy consumption and costs.
- **Contents:**
  - **Exercise 1:** Daily/monthly energy totals, kWh→MJ/GJ conversion, monthly and annual cost.
  - **Exercise 2:** Energy intensity (kWh/m²/month) for a sample building set; total/average/min–max; list buildings above average.
  - **Exercise 3:** Efficiency rating **A–F** based on annual energy intensity (thresholds: 50/100/150/200 kWh/m²/year).
  - **Exercise 4:** Simple cost function and tiered (peak/off-peak) pricing calculator.
  - **Exercise 5:** Best/worst performer, percentage gap, and total kWh and $ savings if all buildings reach **B** level (100 kWh/m²/year).
  - **Bonus:** Interactive calculator that takes user inputs (monthly kWh, floor area) and reports intensity, rating, and monthly tiered cost.
- **How to run:** Execute the notebook cells in order. The bonus section prompts for console input.

# Lab02 — Soil Test Data Analysis

- **lab2_soil_analysis.py:** Loads the CSV, cleans the data (coerces to numeric, fills NaNs with column means, removes `soil_ph` outliers using ±3σ), and generates output + statistics files.
- **soil_test.csv:** Raw input data (include in the repo if required).
- **Outputs:** `soil_test_cleaned.csv`, `soil_test_stats.csv`
