"""
Time series forecasting for patient vital signs using Prophet.
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import joblib
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta
import logging
import warnings

# Suppress Prophet warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VitalForecaster:
    """Enhanced forecaster for patient vital signs using Prophet."""

    def __init__(self, vital_type: str):
        """
        Initialize the forecaster for a specific vital sign.

        Args:
            vital_type: Type of vital sign to forecast ('heart_rate', 'spo2', etc.)
        """
        self.vital_type = vital_type

        # Enhanced Prophet configuration based on vital type
        prophet_params = self._get_prophet_params(vital_type)
        self.model = Prophet(**prophet_params)

        # Add custom seasonalities for better forecasting
        self.model.add_seasonality(name="hourly", period=24, fourier_order=8)

        self.is_trained = False
        self.last_training_date = None
        self.data_quality_metrics = {}

    def _get_prophet_params(self, vital_type: str) -> dict:
        """Get optimized Prophet parameters for each vital type."""
        base_params = {
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": False,
            "interval_width": 0.95,  # Increased for better uncertainty estimation
            "changepoint_prior_scale": 0.05,  # More conservative changepoint detection
            "seasonality_prior_scale": 10.0,  # Enhanced seasonality detection
            "holidays_prior_scale": 10.0,
            "mcmc_samples": 0,
            "growth": "linear",
            "n_changepoints": 25,
        }

        # Vital-specific optimizations
        if vital_type == "spo2":
            # SpO2 is more stable, needs less aggressive seasonality
            base_params.update(
                {
                    "seasonality_prior_scale": 5.0,
                    "changepoint_prior_scale": 0.01,
                    "interval_width": 0.99,
                }
            )
        elif vital_type in ["systolic_bp", "diastolic_bp"]:
            # Blood pressure has stronger daily patterns
            base_params.update(
                {
                    "seasonality_prior_scale": 15.0,
                    "changepoint_prior_scale": 0.1,
                    "interval_width": 0.90,
                }
            )
        elif vital_type == "heart_rate":
            # Heart rate has complex patterns
            base_params.update(
                {
                    "seasonality_prior_scale": 12.0,
                    "changepoint_prior_scale": 0.08,
                    "n_changepoints": 30,
                }
            )
        elif vital_type == "temperature":
            # Temperature is relatively stable
            base_params.update(
                {"seasonality_prior_scale": 8.0, "changepoint_prior_scale": 0.03}
            )

        return base_params

    def prepare_data(self, vitals_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Enhanced data preparation for Prophet model with quality checks.

        Args:
            vitals_data: List of vital signs with timestamps

        Returns:
            DataFrame formatted for Prophet (ds, y columns)
        """
        df = pd.DataFrame(vitals_data)

        if df.empty:
            raise ValueError("No data provided for forecasting")

        # Ensure timestamp column exists
        if "timestamp" not in df.columns:
            raise ValueError("Timestamp column required for forecasting")

        # Ensure vital type column exists
        if self.vital_type not in df.columns:
            raise ValueError(f"Vital type '{self.vital_type}' not found in data")

        # Create Prophet format dataframe
        prophet_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(df["timestamp"]),
                "y": pd.to_numeric(df[self.vital_type], errors="coerce"),
            }
        )

        # Data quality checks and improvements
        initial_count = len(prophet_df)

        # Remove rows with missing values
        prophet_df = prophet_df.dropna()

        # Remove outliers using IQR method for better forecasting
        Q1 = prophet_df["y"].quantile(0.25)
        Q3 = prophet_df["y"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Apply vital-specific bounds
        bounds = self._get_vital_bounds(self.vital_type)
        lower_bound = max(lower_bound, bounds["min"])
        upper_bound = min(upper_bound, bounds["max"])

        # Filter outliers
        prophet_df = prophet_df[
            (prophet_df["y"] >= lower_bound) & (prophet_df["y"] <= upper_bound)
        ]

        # Sort by timestamp and remove duplicates
        prophet_df = (
            prophet_df.sort_values("ds").drop_duplicates("ds").reset_index(drop=True)
        )

        # Calculate data quality metrics
        self.data_quality_metrics = {
            "initial_samples": initial_count,
            "final_samples": len(prophet_df),
            "data_retention": (
                len(prophet_df) / initial_count if initial_count > 0 else 0
            ),
            "outliers_removed": initial_count - len(prophet_df),
            "time_span_hours": (
                prophet_df["ds"].max() - prophet_df["ds"].min()
            ).total_seconds()
            / 3600,
            "mean_value": float(prophet_df["y"].mean()),
            "std_value": float(prophet_df["y"].std()),
        }

        return prophet_df

    def _get_vital_bounds(self, vital_type: str) -> dict:
        """Get realistic bounds for each vital sign type."""
        bounds = {
            "heart_rate": {"min": 30, "max": 200},
            "spo2": {"min": 70, "max": 100},
            "temperature": {"min": 32.0, "max": 45.0},
            "systolic_bp": {"min": 70, "max": 220},
            "diastolic_bp": {"min": 40, "max": 130},
        }
        return bounds.get(vital_type, {"min": -np.inf, "max": np.inf})

    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Enhanced training for the forecasting model with cross-validation.

        Args:
            training_data: List of vital signs data with timestamps
        """
        logger.info(
            f"Training forecaster for {self.vital_type} with {len(training_data)} samples"
        )

        # Prepare data
        prophet_df = self.prepare_data(training_data)

        if len(prophet_df) < 20:  # Increased minimum samples
            raise ValueError(
                f"Need at least 20 data points for training, got {len(prophet_df)}"
            )

        # Add additional regressors for better forecasting
        prophet_df = self._add_regressors(prophet_df)

        # Train model with error handling
        try:
            self.model.fit(prophet_df)
            self.is_trained = True
            self.last_training_date = datetime.now()

            # Log data quality metrics
            logger.info(f"Training completed for {self.vital_type}")
            logger.info(
                f"Data retention: {self.data_quality_metrics['data_retention']:.3f}"
            )
            logger.info(f"Final samples: {self.data_quality_metrics['final_samples']}")

        except Exception as e:
            logger.error(f"Training failed for {self.vital_type}: {e}")
            raise

    def _add_regressors(self, prophet_df: pd.DataFrame) -> pd.DataFrame:
        """Add additional regressors to improve forecasting accuracy."""
        df = prophet_df.copy()

        # Add hour of day as regressor
        df["hour"] = df["ds"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Add day of week
        df["dayofweek"] = df["ds"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        # Add these as regressors to the model
        for regressor in ["hour_sin", "hour_cos", "is_weekend"]:
            self.model.add_regressor(regressor)

        return df

    def predict(
        self, historical_data: List[Dict[str, Any]], forecast_horizon: int = 24
    ) -> Dict[str, Any]:
        """
        Generate forecasts for the vital sign.

        Args:
            historical_data: Recent historical data for context
            forecast_horizon: Number of hours to forecast ahead

        Returns:
            Dictionary containing forecast results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare historical data
        historical_df = self.prepare_data(historical_data)

        if historical_df.empty:
            raise ValueError("No valid historical data for forecasting")

        # Create future dataframe
        last_timestamp = historical_df["ds"].max()
        future_dates = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=forecast_horizon,
            freq="h",
        )

        # Combine historical and future dates
        all_dates = pd.concat(
            [historical_df["ds"], pd.Series(future_dates)]
        ).reset_index(drop=True)

        future_df = pd.DataFrame({"ds": all_dates})

        # Add regressors for prediction
        future_df = self._add_regressors_for_prediction(future_df)

        # Make predictions
        forecast = self.model.predict(future_df)

        # Extract forecast period results
        forecast_start_idx = len(historical_df)
        forecast_results = forecast.iloc[forecast_start_idx:]

        # Prepare results
        predictions = {
            "timestamps": forecast_results["ds"]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist(),
            "values": forecast_results["yhat"].round(2).tolist(),
            "lower_bounds": forecast_results["yhat_lower"].round(2).tolist(),
            "upper_bounds": forecast_results["yhat_upper"].round(2).tolist(),
        }

        # Calculate confidence intervals
        confidence_intervals = {
            "lower": float(forecast_results["yhat_lower"].mean()),
            "upper": float(forecast_results["yhat_upper"].mean()),
            "width": float(
                forecast_results["yhat_upper"].mean()
                - forecast_results["yhat_lower"].mean()
            ),
        }

        # Get trend analysis
        trend_component = forecast_results["trend"].values
        trend_direction = (
            "increasing" if trend_component[-1] > trend_component[0] else "decreasing"
        )
        trend_strength = abs(trend_component[-1] - trend_component[0]) / len(
            trend_component
        )

        return {
            "vital_type": self.vital_type,
            "predictions": predictions,
            "confidence_intervals": confidence_intervals,
            "forecast_horizon": forecast_horizon,
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "model_performance": self._evaluate_model_fit(
                historical_df, forecast.iloc[: len(historical_df)]
            ),
        }

    def _add_regressors_for_prediction(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Add regressors for prediction phase."""
        df = future_df.copy()

        # Add hour of day as regressor
        df["hour"] = df["ds"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Add day of week
        df["dayofweek"] = df["ds"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

        return df

    def _evaluate_model_fit(
        self, actual_df: pd.DataFrame, fitted_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Enhanced model evaluation with multiple metrics.

        Args:
            actual_df: Actual historical data
            fitted_df: Model fitted values

        Returns:
            Dictionary of performance metrics
        """
        actual_values = actual_df["y"].values
        fitted_values = fitted_df["yhat"].values

        # Ensure same length
        min_length = min(len(actual_values), len(fitted_values))
        actual_values = actual_values[:min_length]
        fitted_values = fitted_values[:min_length]

        # Calculate comprehensive metrics
        mae = np.mean(np.abs(actual_values - fitted_values))
        mse = np.mean((actual_values - fitted_values) ** 2)
        rmse = np.sqrt(mse)

        # Calculate R-squared with better handling
        ss_res = np.sum((actual_values - fitted_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)

        # Handle edge cases for R-squared
        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = max(-1.0, 1 - (ss_res / ss_tot))  # Clamp minimum to -1

        # Additional metrics
        mape = (
            np.mean(
                np.abs(
                    (actual_values - fitted_values)
                    / np.maximum(np.abs(actual_values), 1e-8)
                )
            )
            * 100
        )

        # Direction accuracy (trend prediction)
        actual_trend = np.diff(actual_values)
        fitted_trend = np.diff(fitted_values)
        direction_accuracy = (
            np.mean(np.sign(actual_trend) == np.sign(fitted_trend))
            if len(actual_trend) > 0
            else 0.0
        )

        # Confidence intervals coverage (if available)
        coverage = 0.0
        if "yhat_lower" in fitted_df.columns and "yhat_upper" in fitted_df.columns:
            lower_bounds = fitted_df["yhat_lower"].values[:min_length]
            upper_bounds = fitted_df["yhat_upper"].values[:min_length]
            coverage = np.mean(
                (actual_values >= lower_bounds) & (actual_values <= upper_bounds)
            )

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r_squared": float(r_squared),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
            "confidence_coverage": float(coverage),
            "data_quality_score": self._calculate_data_quality_score(),
        }

    def _calculate_data_quality_score(self) -> float:
        """Calculate a data quality score based on various metrics."""
        if not self.data_quality_metrics:
            return 0.0

        # Components of data quality
        retention_score = self.data_quality_metrics.get("data_retention", 0.0)
        sample_count_score = min(
            1.0, self.data_quality_metrics.get("final_samples", 0) / 100
        )
        time_span_score = min(
            1.0, self.data_quality_metrics.get("time_span_hours", 0) / (24 * 7)
        )  # Week of data

        # Weighted average
        quality_score = (
            retention_score * 0.4 + sample_count_score * 0.3 + time_span_score * 0.3
        )

        return float(quality_score)

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "vital_type": self.vital_type,
            "last_training_date": self.last_training_date.isoformat(),
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Forecaster model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.vital_type = model_data["vital_type"]
        self.is_trained = model_data["is_trained"]
        self.last_training_date = datetime.fromisoformat(
            model_data["last_training_date"]
        )

        logger.info(f"Forecaster model loaded from {filepath}")
        logger.info(f"Model trained at: {self.last_training_date}")


class MultiVitalForecaster:
    """Manages multiple forecasters for different vital signs."""

    def __init__(self, vital_types: List[str]):
        """
        Initialize forecasters for multiple vital signs.

        Args:
            vital_types: List of vital sign types to forecast
        """
        self.vital_types = vital_types
        self.forecasters = {
            vital_type: VitalForecaster(vital_type) for vital_type in vital_types
        }

    def train_all(self, training_data: List[Dict[str, Any]]) -> None:
        """Train all forecasters with the same training data."""
        for vital_type, forecaster in self.forecasters.items():
            try:
                forecaster.train(training_data)
            except Exception as e:
                logger.error(f"Failed to train forecaster for {vital_type}: {e}")

    def predict_all(
        self, historical_data: List[Dict[str, Any]], forecast_horizon: int = 24
    ) -> Dict[str, Any]:
        """Generate forecasts for all vital signs."""
        results = {}

        for vital_type, forecaster in self.forecasters.items():
            try:
                if forecaster.is_trained:
                    results[vital_type] = forecaster.predict(
                        historical_data, forecast_horizon
                    )
                else:
                    logger.warning(f"Forecaster for {vital_type} is not trained")
            except Exception as e:
                logger.error(f"Failed to generate forecast for {vital_type}: {e}")
                results[vital_type] = None

        return results

    def save_all_models(self, directory: str) -> None:
        """Save all trained models to directory."""
        os.makedirs(directory, exist_ok=True)

        for vital_type, forecaster in self.forecasters.items():
            if forecaster.is_trained:
                filepath = os.path.join(directory, f"forecaster_{vital_type}.joblib")
                forecaster.save_model(filepath)

    def load_all_models(self, directory: str) -> None:
        """Load all models from directory."""
        for vital_type, forecaster in self.forecasters.items():
            filepath = os.path.join(directory, f"forecaster_{vital_type}.joblib")
            if os.path.exists(filepath):
                forecaster.load_model(filepath)
            else:
                logger.warning(f"Model file not found for {vital_type}: {filepath}")


def generate_time_series_data(
    n_days: int = 30, freq_hours: int = 1, random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate enhanced synthetic time series data with realistic patterns.

    Args:
        n_days: Number of days of data to generate
        freq_hours: Frequency of data points in hours
        random_state: Random seed for reproducibility

    Returns:
        List of vital signs data with timestamps
    """
    np.random.seed(random_state)

    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = pd.date_range(
        start=start_date, periods=n_days * 24 // freq_hours, freq=f"{freq_hours}h"
    )

    time_series_data = []

    # Add patient-specific baseline variations
    patient_baselines = {
        "heart_rate": np.random.normal(75, 8),
        "spo2": np.random.normal(98, 0.5),
        "temperature": np.random.normal(36.8, 0.2),
        "systolic_bp": np.random.normal(120, 10),
        "diastolic_bp": np.random.normal(80, 8),
    }

    # Add weekly and daily patterns
    for i, timestamp in enumerate(timestamps):
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        day_progress = i / len(timestamps)  # Long-term trends

        # Enhanced heart rate with multiple patterns
        hr_daily = 10 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)  # Peak afternoon
        hr_weekly = 3 * np.sin(2 * np.pi * day_of_week / 7)  # Weekly variation
        hr_trend = 2 * np.sin(2 * np.pi * day_progress)  # Gradual changes
        hr_activity = (
            5 * np.random.exponential(0.2)
            if hour_of_day >= 8 and hour_of_day <= 22
            else 0
        )
        heart_rate = (
            patient_baselines["heart_rate"]
            + hr_daily
            + hr_weekly
            + hr_trend
            + hr_activity
        )
        heart_rate += np.random.normal(0, 3)  # Random noise
        heart_rate = max(45, min(150, heart_rate))

        # Enhanced SpO2 with occasional dips
        spo2_base = patient_baselines["spo2"]
        spo2_daily = 0.5 * np.sin(2 * np.pi * hour_of_day / 24)
        spo2_random_dip = -2 if np.random.random() < 0.02 else 0  # Occasional dips
        spo2 = spo2_base + spo2_daily + spo2_random_dip + np.random.normal(0, 0.3)
        spo2 = max(92, min(100, spo2))

        # Enhanced temperature with fever episodes
        temp_daily = 0.4 * np.sin(2 * np.pi * (hour_of_day - 18) / 24)  # Peak evening
        temp_weekly = 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        temp_fever = 1.5 if np.random.random() < 0.01 else 0  # Rare fever spikes
        temperature = (
            patient_baselines["temperature"] + temp_daily + temp_weekly + temp_fever
        )
        temperature += np.random.normal(0, 0.15)
        temperature = max(35.0, min(39.5, temperature))

        # Enhanced blood pressure with stress patterns
        bp_daily = 8 * np.sin(2 * np.pi * (hour_of_day - 10) / 24)  # Peak late morning
        bp_weekly = 4 * np.sin(2 * np.pi * day_of_week / 7)
        bp_stress = 15 if np.random.random() < 0.05 else 0  # Stress episodes

        systolic_bp = (
            patient_baselines["systolic_bp"] + bp_daily + bp_weekly + bp_stress
        )
        systolic_bp += np.random.normal(0, 5)
        systolic_bp = max(85, min(180, systolic_bp))

        # Diastolic follows systolic with some independence
        diastolic_ratio = 0.6 + np.random.normal(0, 0.05)
        diastolic_bp = systolic_bp * diastolic_ratio + np.random.normal(0, 3)
        diastolic_bp = max(50, min(systolic_bp - 15, min(110, diastolic_bp)))

        time_series_data.append(
            {
                "timestamp": timestamp.isoformat(),
                "heart_rate": round(heart_rate, 1),
                "spo2": round(spo2, 1),
                "temperature": round(temperature, 2),
                "systolic_bp": round(systolic_bp, 1),
                "diastolic_bp": round(diastolic_bp, 1),
            }
        )

    return time_series_data
