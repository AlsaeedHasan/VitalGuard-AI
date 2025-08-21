"""
Specialized training for Diastolic Blood Pressure forecasting model.
Enhanced approach with blood pressure specific features and modeling.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import after path setup
try:
    from ml.forecasting import generate_time_series_data
except ImportError:
    # Fallback - define minimal function locally
    def generate_time_series_data(*args, **kwargs):
        return []


from prophet import Prophet
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedBloodPressureForecaster:
    """Enhanced forecaster specifically designed for blood pressure prediction."""

    def __init__(self, pressure_type: str = "diastolic_bp"):
        """
        Initialize enhanced blood pressure forecaster.

        Args:
            pressure_type: 'systolic_bp' or 'diastolic_bp'
        """
        self.pressure_type = pressure_type
        self.is_trained = False
        self.last_training_date = None

        # Hybrid approach: Prophet + Random Forest
        self.prophet_model = self._get_optimized_prophet()
        self.rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        self.scaler = StandardScaler()

        # Feature engineering components
        self.feature_columns = []

    def _get_optimized_prophet(self) -> Prophet:
        """Get specially optimized Prophet for blood pressure."""
        if self.pressure_type == "diastolic_bp":
            # Diastolic-specific optimization
            prophet_params = {
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False,
                "interval_width": 0.85,  # Lower confidence intervals
                "changepoint_prior_scale": 0.15,  # More sensitive to changes
                "seasonality_prior_scale": 20.0,  # Stronger seasonality
                "holidays_prior_scale": 15.0,
                "seasonality_mode": "multiplicative",  # Better for BP patterns
                "growth": "linear",
                "n_changepoints": 35,  # More changepoints for BP variations
            }
        else:  # systolic_bp
            prophet_params = {
                "daily_seasonality": True,
                "weekly_seasonality": True,
                "yearly_seasonality": False,
                "interval_width": 0.90,
                "changepoint_prior_scale": 0.12,
                "seasonality_prior_scale": 18.0,
                "holidays_prior_scale": 12.0,
                "seasonality_mode": "multiplicative",
                "growth": "linear",
                "n_changepoints": 30,
            }

        model = Prophet(**prophet_params)

        # Add custom seasonalities for blood pressure patterns
        model.add_seasonality(
            name="hourly_bp",
            period=24,
            fourier_order=12,  # Higher order for complex BP patterns
            mode="multiplicative",
        )

        # Add stress cycles (work week patterns)
        model.add_seasonality(
            name="work_cycle", period=7, fourier_order=6, mode="additive"
        )

        return model

    def prepare_enhanced_data(self, vitals_data):
        """Enhanced data preparation with blood pressure specific features."""
        df = pd.DataFrame(vitals_data)

        if df.empty:
            raise ValueError("No data provided for forecasting")

        # Ensure required columns exist
        required_cols = ["timestamp", "systolic_bp", "diastolic_bp", "heart_rate"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")

        # Convert and clean data
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Calculate blood pressure derived features
        df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        df["mean_arterial_pressure"] = (df["systolic_bp"] + 2 * df["diastolic_bp"]) / 3
        df["bp_ratio"] = df["diastolic_bp"] / df["systolic_bp"]

        # Cardiovascular stress indicators
        df["pressure_product"] = (
            df["heart_rate"] * df["systolic_bp"]
        )  # Rate-pressure product
        df["cardiovascular_load"] = np.sqrt(
            df["heart_rate"] ** 2 + df["systolic_bp"] ** 2
        )

        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_work_hours"] = ((df["hour"] >= 8) & (df["hour"] <= 18)).astype(int)
        df["is_sleep_hours"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

        # Circadian rhythm features
        df["circadian_bp"] = 10 * np.sin(
            2 * np.pi * (df["hour"] - 6) / 24
        )  # BP peaks afternoon
        df["sleep_recovery"] = 5 * np.cos(
            2 * np.pi * df["hour"] / 24
        )  # Recovery during sleep

        # Stress patterns (work week effect)
        df["work_stress"] = np.where(
            (df["day_of_week"] >= 0) & (df["day_of_week"] <= 4) & df["is_work_hours"],
            5,
            0,
        )

        # Rolling features (trend indicators)
        if len(df) > 12:  # At least 12 hours of data
            df["bp_trend_6h"] = (
                df[self.pressure_type].rolling(window=6, min_periods=3).mean()
            )
            df["bp_volatility_6h"] = (
                df[self.pressure_type].rolling(window=6, min_periods=3).std()
            )
            df["hr_correlation"] = (
                df["heart_rate"]
                .rolling(window=12, min_periods=6)
                .corr(df[self.pressure_type])
            )
        else:
            df["bp_trend_6h"] = df[self.pressure_type]
            df["bp_volatility_6h"] = 0
            df["hr_correlation"] = 0

        # Fill NaN values
        df = df.fillna(method="bfill").fillna(method="ffill")

        # Remove outliers specific to blood pressure
        if self.pressure_type == "diastolic_bp":
            df = df[(df[self.pressure_type] >= 40) & (df[self.pressure_type] <= 130)]
        else:
            df = df[(df[self.pressure_type] >= 70) & (df[self.pressure_type] <= 220)]

        return df

    def train(self, training_data):
        """Enhanced training with hybrid approach."""
        logger.info(f"Training enhanced BP forecaster for {self.pressure_type}")

        # Prepare enhanced data
        df = self.prepare_enhanced_data(training_data)

        if len(df) < 48:  # Need at least 48 hours of data
            raise ValueError(
                f"Need at least 48 data points for BP training, got {len(df)}"
            )

        # Prepare Prophet data
        prophet_df = pd.DataFrame({"ds": df["timestamp"], "y": df[self.pressure_type]})

        # Add regressors to Prophet
        regressors = [
            "pulse_pressure",
            "mean_arterial_pressure",
            "heart_rate",
            "pressure_product",
            "cardiovascular_load",
            "circadian_bp",
            "work_stress",
            "is_weekend",
            "is_sleep_hours",
        ]

        for regressor in regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor]
                self.prophet_model.add_regressor(regressor)

        # Train Prophet model
        logger.info("Training Prophet component...")
        self.prophet_model.fit(prophet_df)

        # Prepare features for Random Forest
        feature_cols = [
            "hour",
            "day_of_week",
            "is_weekend",
            "is_work_hours",
            "is_sleep_hours",
            "pulse_pressure",
            "mean_arterial_pressure",
            "bp_ratio",
            "heart_rate",
            "pressure_product",
            "cardiovascular_load",
            "circadian_bp",
            "sleep_recovery",
            "work_stress",
            "bp_trend_6h",
            "bp_volatility_6h",
            "hr_correlation",
        ]

        # Create lagged features
        for lag in [1, 2, 3, 6, 12]:  # 1,2,3,6,12 hours ago
            if len(df) > lag:
                df[f"{self.pressure_type}_lag_{lag}h"] = df[self.pressure_type].shift(
                    lag
                )
                df[f"heart_rate_lag_{lag}h"] = df["heart_rate"].shift(lag)
                feature_cols.extend(
                    [f"{self.pressure_type}_lag_{lag}h", f"heart_rate_lag_{lag}h"]
                )

        # Clean features
        df = df.dropna()
        if len(df) < 24:
            raise ValueError("Not enough data after feature engineering")

        self.feature_columns = [col for col in feature_cols if col in df.columns]
        X = df[self.feature_columns]
        y = df[self.pressure_type]

        # Scale features and train Random Forest
        logger.info("Training Random Forest component...")
        X_scaled = self.scaler.fit_transform(X)
        self.rf_model.fit(X_scaled, y)

        # Calculate feature importance
        feature_importance = dict(
            zip(self.feature_columns, self.rf_model.feature_importances_)
        )
        logger.info("Top 5 most important features:")
        for feature, importance in sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            logger.info(f"  {feature}: {importance:.3f}")

        self.is_trained = True
        self.last_training_date = datetime.now()
        logger.info(f"Enhanced BP forecaster training completed")

    def predict(self, historical_data, forecast_horizon=24):
        """Enhanced prediction using hybrid approach."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare data
        df = self.prepare_enhanced_data(historical_data)

        if df.empty:
            raise ValueError("No valid historical data for forecasting")

        # Prophet prediction
        prophet_df = pd.DataFrame({"ds": df["timestamp"], "y": df[self.pressure_type]})

        # Add regressors for Prophet
        regressors = [
            "pulse_pressure",
            "mean_arterial_pressure",
            "heart_rate",
            "pressure_product",
            "cardiovascular_load",
            "circadian_bp",
            "work_stress",
            "is_weekend",
            "is_sleep_hours",
        ]

        for regressor in regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor]

        # Create future dataframe for Prophet
        last_timestamp = df["timestamp"].max()
        future_dates = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=forecast_horizon,
            freq="h",
        )

        # Extend regressors for future predictions
        future_df = prophet_df.copy()

        # Add future dates with projected regressor values
        for i, future_date in enumerate(future_dates):
            hour = future_date.hour
            day_of_week = future_date.weekday()

            # Estimate future regressor values based on patterns
            future_row = {
                "ds": future_date,
                "y": None,  # This is what we're predicting
                "heart_rate": df["heart_rate"].iloc[-12:].mean(),  # Recent average
                "circadian_bp": 10 * np.sin(2 * np.pi * (hour - 6) / 24),
                "work_stress": 5 if (day_of_week <= 4 and 8 <= hour <= 18) else 0,
                "is_weekend": 1 if day_of_week >= 5 else 0,
                "is_sleep_hours": 1 if (hour >= 22 or hour <= 6) else 0,
            }

            # Estimate BP-related features from recent trends
            recent_systolic = df["systolic_bp"].iloc[-6:].mean()
            recent_diastolic = df["diastolic_bp"].iloc[-6:].mean()

            if self.pressure_type == "diastolic_bp":
                estimated_systolic = recent_systolic
                estimated_diastolic = recent_diastolic
                future_row["pulse_pressure"] = estimated_systolic - recent_diastolic
                future_row["mean_arterial_pressure"] = (
                    estimated_systolic + 2 * recent_diastolic
                ) / 3
                reference_pressure = estimated_systolic
            else:
                estimated_diastolic = recent_diastolic
                estimated_systolic = recent_systolic
                future_row["pulse_pressure"] = recent_systolic - estimated_diastolic
                future_row["mean_arterial_pressure"] = (
                    recent_systolic + 2 * estimated_diastolic
                ) / 3
                reference_pressure = estimated_systolic

            future_row["pressure_product"] = (
                future_row["heart_rate"] * reference_pressure
            )
            future_row["cardiovascular_load"] = np.sqrt(
                future_row["heart_rate"] ** 2 + reference_pressure**2
            )

            future_df = pd.concat(
                [future_df, pd.DataFrame([future_row])], ignore_index=True
            )

        # Prophet forecast
        prophet_forecast = self.prophet_model.predict(future_df)
        prophet_predictions = prophet_forecast.iloc[-forecast_horizon:]["yhat"].values

        # Evaluate model performance on historical data
        historical_forecast = self.prophet_model.predict(prophet_df)
        actual_values = df[self.pressure_type].values
        predicted_values = historical_forecast["yhat"].values[: len(actual_values)]

        # Calculate enhanced metrics
        performance = self._calculate_enhanced_performance(
            actual_values, predicted_values
        )

        # Get trend analysis
        trend_component = prophet_forecast.iloc[-forecast_horizon:]["trend"].values
        trend_direction = (
            "increasing" if trend_component[-1] > trend_component[0] else "decreasing"
        )
        trend_strength = abs(trend_component[-1] - trend_component[0]) / len(
            trend_component
        )

        return {
            "vital_type": self.pressure_type,
            "predictions": {
                "values": [round(x, 1) for x in prophet_predictions],
                "timestamps": [d.strftime("%Y-%m-%d %H:%M:%S") for d in future_dates],
            },
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "model_performance": performance,
            "forecast_method": "Enhanced Hybrid (Prophet + Feature Engineering)",
        }

    def _calculate_enhanced_performance(self, actual, predicted):
        """Calculate enhanced performance metrics for blood pressure."""
        # Ensure same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]

        # Standard metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        # Blood pressure specific metrics
        bp_ranges = {
            "diastolic_bp": {
                "normal": (60, 80),
                "elevated": (80, 90),
                "high": (90, 130),
            },
            "systolic_bp": {
                "normal": (90, 120),
                "elevated": (120, 140),
                "high": (140, 220),
            },
        }

        ranges = bp_ranges.get(self.pressure_type, bp_ranges["diastolic_bp"])

        # Classification accuracy for BP ranges
        actual_categories = []
        predicted_categories = []

        for a, p in zip(actual, predicted):
            actual_cat = self._categorize_bp(a, ranges)
            predicted_cat = self._categorize_bp(p, ranges)
            actual_categories.append(actual_cat)
            predicted_categories.append(predicted_cat)

        category_accuracy = np.mean(
            [a == p for a, p in zip(actual_categories, predicted_categories)]
        )

        # Clinical significance (within 5 mmHg for BP)
        clinical_accuracy = np.mean(np.abs(actual - predicted) <= 5)

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r_squared": float(r2),
            "category_accuracy": float(category_accuracy),
            "clinical_accuracy": float(clinical_accuracy),
            "model_type": "Enhanced_BP_Forecaster",
        }

    def _categorize_bp(self, value, ranges):
        """Categorize blood pressure value."""
        if ranges["normal"][0] <= value <= ranges["normal"][1]:
            return "normal"
        elif ranges["elevated"][0] <= value <= ranges["elevated"][1]:
            return "elevated"
        else:
            return "high"

    def save_model(self, filepath):
        """Save the enhanced model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "prophet_model": self.prophet_model,
            "rf_model": self.rf_model,
            "scaler": self.scaler,
            "pressure_type": self.pressure_type,
            "feature_columns": self.feature_columns,
            "last_training_date": self.last_training_date.isoformat(),
            "is_trained": self.is_trained,
            "model_version": "Enhanced_BP_v2.0",
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Enhanced BP forecaster saved to {filepath}")


def generate_enhanced_bp_data(n_days=120, freq_hours=1, random_state=42):
    """Generate enhanced blood pressure training data with more realistic patterns."""
    np.random.seed(random_state)

    # Generate base time series
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = pd.date_range(
        start=start_date, periods=n_days * 24 // freq_hours, freq=f"{freq_hours}h"
    )

    data = []

    # Simulate different patient types with BP-specific patterns
    patient_types = ["normal", "prehypertensive", "stage1_hypertensive", "elderly"]
    patient_type = np.random.choice(patient_types, p=[0.4, 0.3, 0.2, 0.1])

    # Base BP values by patient type
    bp_baselines = {
        "normal": {"systolic": 115, "diastolic": 75, "hr": 70},
        "prehypertensive": {"systolic": 130, "diastolic": 85, "hr": 75},
        "stage1_hypertensive": {"systolic": 145, "diastolic": 95, "hr": 80},
        "elderly": {"systolic": 135, "diastolic": 80, "hr": 65},
    }

    baseline = bp_baselines[patient_type]

    # Add medication effects (simulate treatment patterns)
    on_medication = np.random.choice([True, False], p=[0.3, 0.7])
    medication_effectiveness = np.random.uniform(0.7, 0.9) if on_medication else 1.0

    for i, timestamp in enumerate(timestamps):
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        day_progress = i / len(timestamps)

        # Complex blood pressure patterns

        # 1. Circadian rhythm (stronger for BP)
        circadian_systolic = 15 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak afternoon
        circadian_diastolic = 8 * np.sin(
            2 * np.pi * (hour - 8) / 24
        )  # Slightly delayed

        # 2. Weekly stress patterns
        work_stress = 0
        if day_of_week <= 4:  # Weekdays
            if 8 <= hour <= 18:  # Work hours
                work_stress = np.random.uniform(5, 15)
            elif 18 <= hour <= 20:  # Peak stress after work
                work_stress = np.random.uniform(10, 20)

        # 3. Sleep recovery effect
        sleep_recovery = 0
        if 22 <= hour or hour <= 6:  # Sleep hours
            sleep_recovery = -np.random.uniform(5, 12)

        # 4. Physical activity simulation
        activity_effect = 0
        if np.random.random() < 0.15:  # 15% chance of activity
            activity_effect = np.random.uniform(10, 25)

        # 5. Meal effects
        meal_effect = 0
        if hour in [7, 12, 19]:  # Meal times
            meal_effect = np.random.uniform(-3, 8)

        # 6. Long-term trends
        seasonal_trend = 3 * np.sin(
            2 * np.pi * day_progress * 4
        )  # Quarterly variations
        aging_trend = day_progress * 2  # Slight increase over time

        # 7. Random events (illness, stress, etc.)
        random_event = 0
        if np.random.random() < 0.02:  # 2% chance
            random_event = np.random.uniform(-15, 25)

        # Calculate systolic BP
        systolic_bp = (
            baseline["systolic"]
            + circadian_systolic
            + work_stress
            + sleep_recovery
            + activity_effect
            + meal_effect
            + seasonal_trend
            + aging_trend
            + random_event
        ) * medication_effectiveness

        systolic_bp += np.random.normal(0, 4)  # Natural variation
        systolic_bp = max(85, min(200, systolic_bp))

        # Calculate diastolic BP with complex relationship to systolic
        base_ratio = 0.65 + np.random.normal(
            0, 0.03
        )  # Diastolic/Systolic ratio with variation

        diastolic_bp = (
            baseline["diastolic"]
            + circadian_diastolic
            + work_stress * 0.6
            + sleep_recovery * 0.8
            + activity_effect * 0.4
            + meal_effect * 0.3
            + seasonal_trend * 0.7
            + aging_trend * 1.2
            + random_event * 0.5
        ) * medication_effectiveness

        # Ensure physiological relationship
        diastolic_bp = min(diastolic_bp, systolic_bp * base_ratio)
        diastolic_bp += np.random.normal(0, 3)
        diastolic_bp = max(45, min(systolic_bp - 20, min(120, diastolic_bp)))

        # Heart rate with BP correlation
        hr_bp_correlation = 0.3  # Positive correlation
        hr_base = baseline["hr"]

        # BP-related HR changes
        bp_stress_factor = (systolic_bp - baseline["systolic"]) / 50
        hr_from_bp = hr_bp_correlation * bp_stress_factor * 15

        # Independent HR factors
        hr_circadian = 8 * np.sin(2 * np.pi * (hour - 14) / 24)
        hr_activity = activity_effect * 0.8
        hr_sleep = sleep_recovery * 0.5

        heart_rate = hr_base + hr_from_bp + hr_circadian + hr_activity + hr_sleep
        heart_rate += np.random.normal(0, 4)
        heart_rate = max(45, min(150, heart_rate))

        # Other vitals (simplified but correlated)
        spo2 = 98 + np.random.normal(0, 0.8)
        spo2 = max(94, min(100, spo2))

        temperature = (
            36.8 + 0.2 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 0.15)
        )
        temperature = max(35.8, min(37.5, temperature))

        data.append(
            {
                "timestamp": timestamp.isoformat(),
                "heart_rate": round(heart_rate, 1),
                "spo2": round(spo2, 1),
                "temperature": round(temperature, 2),
                "systolic_bp": round(systolic_bp, 1),
                "diastolic_bp": round(diastolic_bp, 1),
            }
        )

    logger.info(f"Generated {len(data)} enhanced BP samples for {patient_type} patient")
    if on_medication:
        logger.info(
            f"Patient on medication (effectiveness: {medication_effectiveness:.2f})"
        )

    return data


def main():
    """Train enhanced diastolic BP forecaster."""
    logger.info("Starting Enhanced Diastolic BP Forecaster Training...")

    # Create output directory
    models_dir = project_root / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate enhanced BP-specific training data
        logger.info("Generating enhanced BP training data...")
        training_data = generate_enhanced_bp_data(
            n_days=120, freq_hours=1, random_state=42
        )
        logger.info(f"Generated {len(training_data)} training samples")

        # Train enhanced diastolic forecaster
        logger.info("Training Enhanced Diastolic BP Forecaster...")
        diastolic_forecaster = EnhancedBloodPressureForecaster("diastolic_bp")
        diastolic_forecaster.train(training_data)

        # Test the enhanced model
        logger.info("Testing Enhanced Diastolic BP Model...")
        recent_data = training_data[-72:]  # Last 72 hours

        result = diastolic_forecaster.predict(recent_data, forecast_horizon=24)
        perf = result["model_performance"]

        logger.info("\n=== ENHANCED DIASTOLIC BP RESULTS ===")
        logger.info(f"Trend: {result['trend_direction']}")
        logger.info(f"R²: {perf['r_squared']:.3f}")
        logger.info(f"RMSE: {perf['rmse']:.2f} mmHg")
        logger.info(f"MAE: {perf['mae']:.2f} mmHg")
        logger.info(f"Category Accuracy: {perf['category_accuracy']:.3f}")
        logger.info(f"Clinical Accuracy (±5mmHg): {perf['clinical_accuracy']:.3f}")
        logger.info(f"Method: {result['forecast_method']}")

        # Save the enhanced model
        enhanced_model_path = models_dir / "forecaster_diastolic_bp.joblib"
        diastolic_forecaster.save_model(str(enhanced_model_path))

        # Also train enhanced systolic for comparison
        logger.info("\nTraining Enhanced Systolic BP Forecaster for comparison...")
        systolic_forecaster = EnhancedBloodPressureForecaster("systolic_bp")
        systolic_forecaster.train(training_data)

        result_systolic = systolic_forecaster.predict(recent_data, forecast_horizon=24)
        perf_systolic = result_systolic["model_performance"]

        logger.info("\n=== ENHANCED SYSTOLIC BP RESULTS ===")
        logger.info(f"R²: {perf_systolic['r_squared']:.3f}")
        logger.info(f"RMSE: {perf_systolic['rmse']:.2f} mmHg")
        logger.info(f"Clinical Accuracy: {perf_systolic['clinical_accuracy']:.3f}")

        # Save enhanced systolic model
        enhanced_systolic_path = models_dir / "forecaster_systolic_bp.joblib"
        systolic_forecaster.save_model(str(enhanced_systolic_path))

        # Performance comparison
        logger.info("\n=== PERFORMANCE COMPARISON ===")
        logger.info(f"Diastolic BP Enhanced R²: {perf['r_squared']:.3f}")
        logger.info(f"Systolic BP Enhanced R²: {perf_systolic['r_squared']:.3f}")

        if perf["r_squared"] > 0.4:
            logger.info("✅ Diastolic BP model significantly improved!")
        elif perf["r_squared"] > 0.2:
            logger.info("⚡ Diastolic BP model moderately improved")
        else:
            logger.warning("⚠️ Diastolic BP model still needs work")

        # Generate comparison with simple model
        logger.info("\nTraining simple Prophet for comparison...")
        try:
            from ml.train_forecast_simple import SimpleVitalForecaster

            simple_forecaster = SimpleVitalForecaster("diastolic_bp")
            simple_forecaster.train(training_data)
            simple_result = simple_forecaster.predict(recent_data, forecast_horizon=24)
            simple_r2 = simple_result["model_performance"]["r_squared"]

            improvement = (
                ((perf["r_squared"] - simple_r2) / abs(simple_r2)) * 100
                if simple_r2 != 0
                else float("inf")
            )

            logger.info(f"\n=== IMPROVEMENT ANALYSIS ===")
            logger.info(f"Simple Prophet R²: {simple_r2:.3f}")
            logger.info(f"Enhanced Model R²: {perf['r_squared']:.3f}")
            logger.info(f"Improvement: {improvement:+.1f}%")
        except ImportError:
            logger.info("Simple forecaster not available for comparison")

    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    logger.info("Enhanced Diastolic BP Forecaster training completed!")
    return 0


if __name__ == "__main__":
    exit(main())
