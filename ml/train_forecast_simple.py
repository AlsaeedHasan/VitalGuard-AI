"""
Training script for simplified forecasting models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.forecasting import generate_time_series_data
from prophet import Prophet
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleVitalForecaster:
    """Simplified forecaster for patient vital signs using Prophet."""

    def __init__(self, vital_type: str):
        self.vital_type = vital_type

        # Optimized Prophet configuration based on vital type
        prophet_params = self._get_prophet_params(vital_type)
        self.model = Prophet(**prophet_params)

        self.is_trained = False
        self.last_training_date = None
        self.data_quality_metrics = {}

    def _get_prophet_params(self, vital_type: str) -> dict:
        """Get optimized Prophet parameters for each vital type."""
        base_params = {
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": False,
            "interval_width": 0.95,
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "holidays_prior_scale": 10.0,
            "mcmc_samples": 0,
            "growth": "linear",
            "n_changepoints": 25,
        }

        # Vital-specific optimizations
        if vital_type == "spo2":
            base_params.update(
                {
                    "seasonality_prior_scale": 5.0,
                    "changepoint_prior_scale": 0.01,
                    "interval_width": 0.99,
                }
            )
        elif vital_type in ["systolic_bp", "diastolic_bp"]:
            base_params.update(
                {
                    "seasonality_prior_scale": 15.0,
                    "changepoint_prior_scale": 0.1,
                    "interval_width": 0.90,
                }
            )
        elif vital_type == "heart_rate":
            base_params.update(
                {
                    "seasonality_prior_scale": 12.0,
                    "changepoint_prior_scale": 0.08,
                    "n_changepoints": 30,
                }
            )
        elif vital_type == "temperature":
            base_params.update(
                {"seasonality_prior_scale": 8.0, "changepoint_prior_scale": 0.03}
            )

        return base_params

    def prepare_data(self, vitals_data):
        """Prepare data for Prophet model."""
        df = pd.DataFrame(vitals_data)

        if df.empty:
            raise ValueError("No data provided for forecasting")

        # Create Prophet format dataframe
        prophet_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(df["timestamp"]),
                "y": pd.to_numeric(df[self.vital_type], errors="coerce"),
            }
        )

        # Remove rows with missing values
        prophet_df = prophet_df.dropna()

        # Remove outliers using IQR method
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

    def train(self, training_data):
        """Train the forecasting model."""
        logger.info(
            f"Training forecaster for {self.vital_type} with {len(training_data)} samples"
        )

        # Prepare data
        prophet_df = self.prepare_data(training_data)

        if len(prophet_df) < 20:
            raise ValueError(
                f"Need at least 20 data points for training, got {len(prophet_df)}"
            )

        # Train model
        self.model.fit(prophet_df)
        self.is_trained = True
        self.last_training_date = datetime.now()

        logger.info(f"Training completed for {self.vital_type}")

    def predict(self, historical_data, forecast_horizon=24):
        """Generate forecasts for the vital sign."""
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

        # Make predictions
        forecast = self.model.predict(future_df)

        # Extract forecast period results
        forecast_start_idx = len(historical_df)
        forecast_results = forecast.iloc[forecast_start_idx:]

        # Get trend analysis
        trend_component = forecast_results["trend"].values
        trend_direction = (
            "increasing" if trend_component[-1] > trend_component[0] else "decreasing"
        )

        # Calculate model performance
        performance = self._evaluate_model_fit(
            historical_df, forecast.iloc[: len(historical_df)]
        )

        return {
            "vital_type": self.vital_type,
            "trend_direction": trend_direction,
            "model_performance": performance,
        }

    def _evaluate_model_fit(self, actual_df, fitted_df):
        """Evaluate model fit on historical data."""
        actual_values = actual_df["y"].values
        fitted_values = fitted_df["yhat"].values

        # Ensure same length
        min_length = min(len(actual_values), len(fitted_values))
        actual_values = actual_values[:min_length]
        fitted_values = fitted_values[:min_length]

        # Calculate metrics
        mae = np.mean(np.abs(actual_values - fitted_values))
        mse = np.mean((actual_values - fitted_values) ** 2)
        rmse = np.sqrt(mse)

        # Calculate R-squared with better handling
        ss_res = np.sum((actual_values - fitted_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)

        if ss_tot == 0:
            r_squared = 1.0 if ss_res == 0 else 0.0
        else:
            r_squared = max(-1.0, 1 - (ss_res / ss_tot))

        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r_squared": float(r_squared),
        }

    def save_model(self, filepath):
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


def main():
    """Train and save the enhanced forecasting models."""
    logger.info("Starting enhanced forecasting models training...")

    # Create output directory
    models_dir = project_root / "ml" / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate more comprehensive time series training data
    logger.info("Generating enhanced synthetic time series data...")
    training_data = generate_time_series_data(n_days=90, freq_hours=1, random_state=42)
    logger.info(f"Generated {len(training_data)} time series samples")

    # Initialize forecasters
    vital_types = ["heart_rate", "spo2", "temperature", "systolic_bp", "diastolic_bp"]
    forecasters = {}

    try:
        # Train all forecasters
        logger.info("Training enhanced forecasting models...")
        for vital_type in vital_types:
            forecaster = SimpleVitalForecaster(vital_type)
            forecaster.train(training_data)
            forecasters[vital_type] = forecaster

        # Enhanced testing
        logger.info("Testing forecasting models...")
        recent_data = training_data[-72:]  # Last 72 hours for context

        logger.info("\n=== FORECASTING RESULTS ===")
        best_performers = []
        poor_performers = []

        for vital_type, forecaster in forecasters.items():
            try:
                result = forecaster.predict(recent_data, forecast_horizon=24)
                perf = result["model_performance"]

                logger.info(f"\nForecast for {vital_type}:")
                logger.info(f"  Trend: {result['trend_direction']}")
                logger.info(f"  R²: {perf['r_squared']:.3f}")
                logger.info(f"  RMSE: {perf['rmse']:.2f}")
                logger.info(f"  MAE: {perf['mae']:.2f}")

                # Interpret results
                if perf["r_squared"] < 0:
                    logger.warning(
                        f"  ⚠️  Negative R² indicates poor model performance!"
                    )
                    poor_performers.append((vital_type, perf["r_squared"]))
                elif perf["r_squared"] < 0.3:
                    logger.warning(
                        f"  ⚠️  Low R² ({perf['r_squared']:.3f}) - model needs improvement"
                    )
                    poor_performers.append((vital_type, perf["r_squared"]))
                elif perf["r_squared"] > 0.7:
                    logger.info(
                        f"  ✅  Good R² ({perf['r_squared']:.3f}) - reliable forecasting"
                    )
                    best_performers.append((vital_type, perf["r_squared"]))
                else:
                    best_performers.append((vital_type, perf["r_squared"]))

            except Exception as e:
                logger.error(f"Failed to test {vital_type}: {e}")

        # Save all models
        logger.info("\nSaving enhanced forecasting models...")
        for vital_type, forecaster in forecasters.items():
            filepath = models_dir / f"forecaster_{vital_type}.joblib"
            forecaster.save_model(str(filepath))

        # Summary report
        logger.info("\n=== FORECASTING TRAINING SUMMARY ===")

        if best_performers:
            logger.info("Best performing models:")
            for vital, r2 in sorted(best_performers, key=lambda x: x[1], reverse=True):
                logger.info(f"  {vital}: R² = {r2:.3f}")

        if poor_performers:
            logger.warning("Models needing improvement:")
            for vital, r2 in sorted(poor_performers, key=lambda x: x[1]):
                logger.warning(
                    f"  {vital}: R² = {r2:.3f} - Consider retraining with different parameters"
                )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    logger.info("Enhanced forecasting models training completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
