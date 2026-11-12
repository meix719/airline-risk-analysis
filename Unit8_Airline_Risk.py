
"""Unit 8 airline risk analysis. Original analysis with a verified smoothing comparison.
Run: python Unit8_Airline_Risk.py path/to/course_workbook.xlsx
"""
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
import sys

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay,
    classification_report
)

folder = Path(__file__).resolve().parent
source = Path(sys.argv[1]) if len(sys.argv) > 1 else folder / "Risk Analytics - Class 8 - Dataset.xlsx"

df = pd.read_excel(
    source,
    sheet_name="Data"
)

print("\nFIRST FIVE ROWS:")
print(df.head())

print("\nDATASET SHAPE:")
print(df.shape)

print("\nMISSING VALUES:")
print(df.isnull().sum())

predictors = [
    "Airline_Name",
    "Scheduled_Departure_Time",
    "Flight_Duration (Hours)",
    "Airplane_Age (Years)",
    "Crew_Experience (Years)",
    "Weather_Condition",
    "Passenger_Count"
]

# Stop rather than silently fit a model with incomplete inputs.
if df[predictors + ["Delay_Status"]].isna().any().any():
    raise ValueError("Missing predictor or delay-status values.")
if not df["Delay_Status"].isin([0, 1]).all():
    raise ValueError("Delay_Status must be 0 or 1.")
# Monthly totals repeat across flight rows; verify agreement before taking one.
if not (df.groupby("Month")["Total_Incidents"].nunique() == 1).all():
    raise ValueError("Conflicting incident totals for the same month.")
X = df[predictors]

y = df["Delay_Status"]

X = pd.get_dummies(X, dtype=int)

print("\nPREDICTOR DATA SHAPE:")
print(X.shape)

print("\nFLIGHT DELAY DISTRIBUTION:")
print(y.value_counts())

print("\nPREPARED PREDICTOR VARIABLES:")
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nDECISION TREE MODEL RESULTS:")
print("Training Flights:", len(X_train))
print("Testing Flights:", len(X_test))
print(f"Model Accuracy: {accuracy:.2%}")

print("\nDECISION TREE COMPARISON:")

for depth in [2, 3, 4, 5]:

    test_model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=3,
        random_state=42
    )

    test_model.fit(X_train, y_train)

    train_accuracy = accuracy_score(
        y_train,
        test_model.predict(X_train)
    )

    test_accuracy = accuracy_score(
        y_test,
        test_model.predict(X_test)
    )

    print(f"\nTree Depth: {depth}")
    print(f"Training Accuracy: {train_accuracy:.2%}")
    print(f"Testing Accuracy: {test_accuracy:.2%}")

model = DecisionTreeClassifier(
    max_depth=2,
    min_samples_leaf=3,
    random_state=42
)

model.fit(X_train, y_train)

print("\nFINAL DECISION TREE:")
print("Actual Tree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())

plt.figure(figsize=(16, 8))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["On Time", "Delayed"],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree for Flight Delay Prediction")
plt.tight_layout()

tree_file = folder / "decision_tree.png"

plt.savefig(
    tree_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDecision Tree saved to:")
print(tree_file)

y_pred = model.predict(X_test)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=[0, 1],
    display_labels=["On Time", "Delayed"],
    cmap="Blues",
    values_format="d"
)

plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()

matrix_file = folder / "confusion_matrix.png"

plt.savefig(
    matrix_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion Matrix saved to:")
print(matrix_file)

print("\nCLASSIFICATION REPORT:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["On Time", "Delayed"],
        zero_division=0
    )
)

print("\nFINAL MODEL ACCURACY:")
print(f"{accuracy_score(y_test, y_pred):.2%}")


print("\nFILES CREATED:")
print("1. decision_tree.png")
print("2. confusion_matrix.png")

if sys.platform == "darwin":
    subprocess.run(["open", str(tree_file)], check=False)
    subprocess.run(["open", str(matrix_file)], check=False)

print("\nANALYSIS COMPLETE!")

import matplotlib.dates as mdates

monthly = (
    df.groupby("Month", as_index=False)["Total_Incidents"]
    .first()
    .sort_values("Month")
)

if pd.api.types.is_numeric_dtype(monthly["Month"]):
    monthly["Month"] = pd.to_datetime(
        monthly["Month"], unit="D", origin="1899-12-30"
    )
else:
    monthly["Month"] = pd.to_datetime(monthly["Month"])

print("\nMONTHLY INCIDENT DATA:")
print(monthly.to_string(index=False))

plt.figure(figsize=(12, 6))

plt.plot(
    monthly["Month"],
    monthly["Total_Incidents"],
    marker="o",
    linewidth=2
)

plt.title("Monthly Airline Incidents (2022–2023)")
plt.xlabel("Month")
plt.ylabel("Total Incidents")
plt.grid(True, alpha=0.3)

plt.gca().xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)
plt.gca().xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.xticks(rotation=45)
plt.tight_layout()

incident_file = folder / "monthly_incidents.png"
plt.savefig(incident_file, dpi=300)
plt.close()

print("\nGraph saved to:", incident_file)

if sys.platform == "darwin":
    subprocess.run(["open", str(incident_file)], check=False)



monthly["Month"] = (
    pd.to_datetime(monthly["Month"])
    .dt.to_period("M")
    .dt.to_timestamp()
)

ts = (
    monthly.set_index("Month")["Total_Incidents"]
    .sort_index()
    .asfreq("MS")
)

future_dates = pd.date_range(
    start=ts.index[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)

forecast = pd.Series(
    ts.loc[
        future_dates - pd.DateOffset(years=1)
    ].to_numpy(),
    index=future_dates
)

print("\n6-MONTH INCIDENT FORECAST:")
print(forecast)

plt.figure(figsize=(12, 6))

plt.plot(
    ts.index,
    ts.values,
    marker="o",
    label="Historical Incidents"
)

plt.plot(
    forecast.index,
    forecast.values,
    marker="o",
    linestyle="--",
    color="red",
    label="Forecast"
)

plt.title("Airline Incident Forecast (2024)")
plt.xlabel("Month")
plt.ylabel("Total Incidents")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

forecast_file = folder / "incident_forecast.png"
plt.savefig(forecast_file, dpi=300)
plt.close()

print("\nForecast saved to:", forecast_file)

if sys.platform == "darwin":
    subprocess.run(
        ["open", str(forecast_file)],
        check=False
    )

actual_2023 = ts.loc["2023"]
predicted_2023 = ts.shift(12).loc["2023"]

mae = (actual_2023 - predicted_2023).abs().mean()

print("\nTIME SERIES MODEL EVALUATION")
print("Mean Absolute Error (MAE):", round(mae, 2))

print("\nActual vs Predicted 2023:")
print(pd.DataFrame({
    "Actual": actual_2023,
    "Predicted": predicted_2023
}))

average_2022 = ts.loc["2022"].mean()

average_forecast = pd.Series(
    average_2022,
    index=actual_2023.index
)

average_mae = (
    actual_2023 - average_forecast
).abs().mean()

print("\nFORECAST MODEL COMPARISON")
print(f"Seasonal Naive MAE: {mae:.2f}")
print(f"Average Forecast MAE: {average_mae:.2f}")

weather_summary = df.groupby(
    "Weather_Condition"
)["Delay_Status"].agg(["count", "sum", "mean"])

weather_summary.columns = [
    "Total_Flights",
    "Delayed_Flights",
    "Delay_Rate"
]

weather_summary["Delay_Rate"] *= 100

print("\nWEATHER IMPACT ANALYSIS:")
print(weather_summary.round(2))

plt.figure(figsize=(10, 6))

weather_summary["Delay_Rate"].plot(
    kind="bar",
    color="steelblue"
)

plt.title("Flight Delay Rate by Weather Condition")
plt.xlabel("Weather Condition")
plt.ylabel("Delay Rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

weather_file = folder / "weather_delay_analysis.png"
plt.savefig(weather_file, dpi=300)
plt.close()

if sys.platform == "darwin":
    subprocess.run(["open", str(weather_file)], check=False)

weather_table = df.groupby("Weather_Condition").agg(
    Total_Flights=("Delay_Status", "count"),
    Delayed_Flights=("Delay_Status", "sum"),
    Delay_Rate=("Delay_Status", "mean")
)

weather_table["Delay_Rate"] *= 100

weather_table = weather_table.sort_values(
    "Delay_Rate", ascending=False
)

weather_table["Delay_Rate"] = (
    weather_table["Delay_Rate"].round(2)
)

print("\nWEATHER CATEGORY COMPARISON:")
print(weather_table.to_string())

weather_table.to_csv(folder / "weather_comparison.csv")



departure_summary = df.groupby(
    "Scheduled_Departure_Time"
)["Delay_Status"].agg(["count", "sum", "mean"])

departure_summary.columns = [
    "Total_Flights",
    "Delayed_Flights",
    "Delay_Rate"
]

departure_summary["Delay_Rate"] *= 100

print("\nFLIGHT DELAYS BY DEPARTURE TIME:")
print(departure_summary.round(2))

weather_time = pd.crosstab(
    df["Scheduled_Departure_Time"],
    df["Weather_Condition"]
)

print("\nWEATHER CONDITIONS BY DEPARTURE TIME:")
print(weather_time.to_string())

delay_comparison = df.pivot_table(
    index="Scheduled_Departure_Time",
    columns="Weather_Condition",
    values="Delay_Status",
    aggfunc="mean"
) * 100

print("\nDELAY RATE BY DEPARTURE TIME AND WEATHER:")
print(delay_comparison.round(2).to_string())



# Compare forecasts fairly: fit on 2022 only and evaluate on all 12 months of 2023.
# Simple exponential smoothing models a level, not a seasonal pattern.
# Explicit simple exponential smoothing recurrence; initialize at the first value.
def smooth_forecast(series, periods, alpha=0.7):
    level = float(series.iloc[0])
    for observation in series.iloc[1:]:
        level = alpha * float(observation) + (1-alpha) * level
    dates = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    return pd.Series(level, index=dates)
import json
train_ts = ts.loc["2022"]
ses_test = smooth_forecast(train_ts, 12)
ses_mae = (actual_2023 - ses_test).abs().mean()
ses_future = smooth_forecast(ts, 6)
comparison = pd.DataFrame({"Model": ["Seasonal naive", "2022 mean baseline", "Simple exponential smoothing (alpha 0.7)"],
    "2023 holdout MAE": [mae, average_mae, ses_mae]})
comparison.to_csv(folder / "forecast_comparison.csv", index=False)
forecast_table = pd.DataFrame({"Seasonal naive": forecast, "Simple exponential smoothing": ses_future})
forecast_table.to_csv(folder / "six_month_forecast.csv", index_label="Month")
departure_summary.to_csv(folder / "departure_comparison.csv")
weather_time.to_csv(folder / "weather_by_departure.csv")
monthly.to_csv(folder / "monthly_incidents.csv", index=False)
print(comparison.to_string(index=False))
print(forecast_table.to_string())
plt.figure(figsize=(10,4.5))
plt.plot(ts.index, ts, marker="o", label="Observed")
plt.plot(forecast.index, forecast, marker="o", linestyle="--", label="Seasonal naive")
plt.plot(ses_future.index, ses_future, marker="s", linestyle="--", label="Simple exponential smoothing")
plt.xlabel("Month"); plt.ylabel("Monthly incidents"); plt.title("Monthly incidents and six-month forecasts")
plt.legend(); plt.grid(alpha=0.2); plt.tight_layout()
plt.savefig(folder / "forecast_comparison.png", dpi=180); plt.close()
summary = {"accuracy": accuracy_score(y_test,y_pred),
    "classification": classification_report(y_test,y_pred,output_dict=True,zero_division=0),
    "feature_importance": dict(zip(X.columns,model.feature_importances_)),
    "mae_seasonal": float(mae), "mae_mean": float(average_mae), "mae_ses": float(ses_mae),
    "mean_baseline_reduction_pct": float((1-mae/average_mae)*100),
    "monthly_consistency": bool((df.groupby("Month")["Total_Incidents"].nunique()==1).all())}
(folder / "analysis_summary.json").write_text(json.dumps(summary,indent=2))
