# Advanced Time Series Forecasting with Prophet and Optuna

## 📌 Project Overview

This project demonstrates a complete, production-grade pipeline for **advanced time series forecasting** using:

* **Facebook Prophet** (forecasting model)
* **Optuna** (hyperparameter optimization framework)
* **Walk-forward cross-validation** (robust evaluation method)
* **Synthetic multi-seasonal dataset** (daily data, 3+ years)
* **Baseline comparison** using **Exponential Smoothing**

The project focuses on building, tuning, and validating a high‑performance forecasting system suitable for complex datasets with multiple seasonal patterns (weekly, yearly) and trend components.

---

## 🎯 Project Goals

1. Generate or load a realistic time series dataset (daily frequency, 3–5 years).
2. Implement **walk-forward cross-validation**, replacing improper k-fold CV.
3. Build a **baseline Prophet model** with default settings.
4. Use **Optuna** to optimize Prophet hyperparameters:

   * `changepoint_prior_scale`
   * `seasonality_prior_scale`
   * `seasonality_mode`
   * `changepoint_range`
   * `n_changepoints`
5. Train the final optimized model and forecast a **held‑out future horizon**.
6. Compare optimized Prophet performance against a **baseline exponential smoothing model**.
7. Produce detailed metrics (RMSE, MAE, MAPE) and export final results.

---

## 📁 Repository Structure

```
project/
│
├── prophet_optuna_pipeline.py        # Main script
├── final_holdout_forecasts.csv       # Forecast output
├── optuna_study_trials.csv           # Optuna trial log
├── forecast_comparison.png           # Visualization output
└── README.md                         # Documentation
```

---

## 📦 Dependencies

Install all required packages:

```bash
pip install prophet optuna pandas numpy matplotlib scikit-learn statsmodels
```

Prophet requires pystan/prophet wheels compatible with your Python version.

---

## 🧪 Synthetic Dataset Description

The dataset generated includes:

* **Daily timestamps (2016–2020)**
* **Linear upward trend**
* **Yearly seasonality** (cosine-based)
* **Weekly seasonality** (weekday/weekend pattern)
* **Holiday-like random spikes**
* **Gaussian noise**

This results in a realistic multi-seasonal forecasting challenge.

---

## 🔁 Walk‑Forward Cross‑Validation Strategy

Walk‑forward CV ensures realistic evaluation by simulating real-time forecasting.

Each split contains:

* A fixed **training window** (2 years)
* A **forecast horizon** of 90 days
* A **step size** of 90 days for the next split

This prevents leakage and mimics production forecasting workflows.

---

## ⚙️ Hyperparameter Optimization (Optuna)

A custom Optuna objective performs CV on each trial.

**Search Space**:

```
changepoint_prior_scale ∈ [0.001, 0.5]
seasonality_prior_scale ∈ [0.01, 20]
seasonality_mode ∈ {additive, multiplicative}
changepoint_range ∈ [0.7, 0.95]
n_changepoints ∈ [5, 50]
```

**Optimization target**: **Mean RMSE across CV splits**

Optuna uses TPE Sampler with 40 trials by default.

---

## 🧠 Final Model Training

The best hyperparameters from Optuna are used to train a **full Prophet model** on the entire training dataset (except final hold‑out window).

This model predicts the last **180 days**, simulating true forecasting conditions.

---

## 🔍 Baseline Comparison

A baseline **Holt-Winters Exponential Smoothing** model (additive trend, yearly seasonality) is trained and evaluated on the same holdout window.

Metrics compared:

* RMSE
* MAE
* MAPE

This ensures the optimized Prophet model is meaningfully better than traditional statistical methods.

---

## 📊 Outputs

1. **final_holdout_forecasts.csv**

   * True values, optimized Prophet predictions, ES forecasts
2. **optuna_study_trials.csv**

   * Trial-by-trial hyperparameters and losses
3. **forecast_comparison.png**

   * Full history + forecasts plot
4. Console-based summary showing:

   * Baseline CV metrics
   * Optimized CV metrics
   * Holdout metrics
   * Best hyperparameters selected

---

## ▶️ Running the Script

Simply run:

```bash
python prophet_optuna_pipeline.py
```

The script automatically:

* Generates dataset
* Performs walk-forward CV
* Runs optimization
* Trains final model
* Saves outputs
* Prints summary metrics and forecast samples

---

## 📈 Example Summary (Printed by Script)

* Best trial hyperparameters
* Baseline Prophet CV metrics
* Optimized Prophet CV metrics
* Holdout performance comparison
* First forecast rows displayed as text

---

## 🧭 Extensions & Suggestions

You can expand the project by adding:

* SARIMA or SARIMAX baseline comparison
* Custom holidays
* Additional regressors (weather, events)
* Cross-validation visualization for hyperparameter convergence
* MLflow logging for experiments
* Deployment with FastAPI + Docker

---

## 🏁 Conclusion

This project provides a **complete, industry-level forecasting pipeline** that demonstrates:

* Proper model evaluation (walk-forward CV)
* Systematic tuning with Optuna
* Practical baseline comparison
* Modular and production‑ready code structure

It is an ideal foundation for advanced forecasting research, MLOps integration, or real-world business forecasting tasks.

---

## 👨‍💻 Author

This README was generated to accompany the **Prophet Optuna Pipeline** forecasting project.
