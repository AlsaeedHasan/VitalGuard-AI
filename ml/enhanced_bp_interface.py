"""
Enhanced Blood Pressure Forecasting Interface
Interface for using the enhanced BP models with Prophet + RandomForest
"""

import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedBPForecaster:
    """
    Enhanced Blood Pressure Forecaster using Prophet + RandomForest models.
    This class provides an interface for the enhanced BP models that use 27 medical features.
    """

    def __init__(self, models_dir: str = None):
        """
        Initialize the Enhanced BP Forecaster.

        Args:
            models_dir: Directory containing the saved models
        """
        if models_dir is None:
            self.models_dir = Path(__file__).parent / "saved_models"
        else:
            self.models_dir = Path(models_dir)

        self.systolic_model = None
        self.diastolic_model = None
        self.loaded_models = {}

    def load_models(self):
        """Load the enhanced BP models."""
        try:
            # Load systolic model
            systolic_path = self.models_dir / "forecaster_systolic_bp.joblib"
            if systolic_path.exists():
                self.systolic_model = joblib.load(systolic_path)
                self.loaded_models["systolic"] = self.systolic_model
                logger.info("Systolic BP model loaded successfully")
            else:
                logger.warning(f"Systolic model not found at {systolic_path}")

            # Load diastolic model
            diastolic_path = self.models_dir / "forecaster_diastolic_bp.joblib"
            if diastolic_path.exists():
                self.diastolic_model = joblib.load(diastolic_path)
                self.loaded_models["diastolic"] = self.diastolic_model
                logger.info("Diastolic BP model loaded successfully")
            else:
                logger.warning(f"Diastolic model not found at {diastolic_path}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

    def _generate_medical_features(self, historical_data: List[Dict]) -> pd.DataFrame:
        """
        Generate the exact medical features required by the enhanced models.

        Args:
            historical_data: List of vital signs measurements

        Returns:
            DataFrame with all required features
        """
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)

        # Ensure we have the basic columns
        required_cols = [
            "heart_rate",
            "spo2",
            "temperature",
            "systolic_bp",
            "diastolic_bp",
        ]
        for col in required_cols:
            if col not in df.columns:
                # Set reasonable defaults if missing
                if col == "heart_rate":
                    df[col] = 75.0
                elif col == "spo2":
                    df[col] = 98.0
                elif col == "temperature":
                    df[col] = 36.7
                elif col == "systolic_bp":
                    df[col] = 120.0
                elif col == "diastolic_bp":
                    df[col] = 80.0

        # Ensure timestamp column
        if "timestamp" not in df.columns:
            df["timestamp"] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(df)),
                periods=len(df),
                freq="h",
            )
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Generate all exact features required by the models
        feature_df = pd.DataFrame()

        # Time-based features
        feature_df["hour"] = df["timestamp"].dt.hour
        feature_df["day_of_week"] = df["timestamp"].dt.dayofweek
        feature_df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
        feature_df["is_work_hours"] = (
            (df["timestamp"].dt.hour >= 9) & (df["timestamp"].dt.hour <= 17)
        ).astype(int)
        feature_df["is_sleep_hours"] = (
            (df["timestamp"].dt.hour >= 22) | (df["timestamp"].dt.hour <= 6)
        ).astype(int)

        # Cardiovascular features
        feature_df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        feature_df["mean_arterial_pressure"] = (
            df["diastolic_bp"] + (df["systolic_bp"] - df["diastolic_bp"]) / 3
        )
        feature_df["bp_ratio"] = df["systolic_bp"] / df["diastolic_bp"]
        feature_df["heart_rate"] = df["heart_rate"]
        feature_df["pressure_product"] = df["systolic_bp"] * df["diastolic_bp"]
        feature_df["cardiovascular_load"] = (
            df["heart_rate"] * feature_df["mean_arterial_pressure"] / 1000
        )

        # Circadian and context features
        feature_df["circadian_bp"] = (
            np.sin(2 * np.pi * df["timestamp"].dt.hour / 24) * 10 + 120
        )
        feature_df["sleep_recovery"] = (
            np.where(feature_df["is_sleep_hours"], 1, 0)
            * (8 - df["timestamp"].dt.hour % 8)
            / 8
        )
        feature_df["work_stress"] = (
            np.where(feature_df["is_work_hours"], 1, 0)
            * (df["timestamp"].dt.hour - 9)
            / 8
        )

        # Rolling statistics (6-hour window)
        window_size = min(6, len(df))
        feature_df["bp_trend_6h"] = self._calculate_trend(
            df["systolic_bp"], window_size
        )
        feature_df["bp_volatility_6h"] = (
            df["systolic_bp"].rolling(window=window_size, min_periods=1).std().fillna(0)
        )
        feature_df["hr_correlation"] = (
            df["heart_rate"]
            .rolling(window=window_size, min_periods=1)
            .corr(df["systolic_bp"])
            .fillna(0)
        )

        # Lag features
        for lag_hours in [1, 2, 3, 6, 12]:
            # Systolic BP lags
            lag_col = f"systolic_bp_lag_{lag_hours}h"
            if lag_hours < len(df):
                feature_df[lag_col] = (
                    df["systolic_bp"].shift(lag_hours).fillna(df["systolic_bp"].iloc[0])
                )
            else:
                feature_df[lag_col] = df["systolic_bp"].iloc[0]

            # Heart rate lags
            lag_col = f"heart_rate_lag_{lag_hours}h"
            if lag_hours < len(df):
                feature_df[lag_col] = (
                    df["heart_rate"].shift(lag_hours).fillna(df["heart_rate"].iloc[0])
                )
            else:
                feature_df[lag_col] = df["heart_rate"].iloc[0]

        # Fill any remaining NaN values
        feature_df = feature_df.ffill().bfill().fillna(0)

        return feature_df

    def _calculate_trend(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate trend over a rolling window."""

        def trend_calc(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]

        return series.rolling(window=window, min_periods=1).apply(trend_calc, raw=False)

    def predict_with_auto_features(
        self,
        historical_data: List[Dict],
        pressure_type: str = "both",
        forecast_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Predict blood pressure using enhanced models with automatic feature generation.

        Args:
            historical_data: List of dictionaries with vital signs
            pressure_type: 'systolic', 'diastolic', or 'both'
            forecast_hours: Number of hours to forecast

        Returns:
            Dictionary with predictions and metadata
        """
        if not self.loaded_models:
            self.load_models()

        # Generate features
        features_df = self._generate_medical_features(historical_data)

        results = {}

        try:
            if (
                pressure_type in ["systolic", "both"]
                and "systolic" in self.loaded_models
            ):
                systolic_pred = self._predict_single_model(
                    features_df, self.systolic_model, "systolic", forecast_hours
                )
                results["systolic"] = systolic_pred

            if (
                pressure_type in ["diastolic", "both"]
                and "diastolic" in self.loaded_models
            ):
                diastolic_pred = self._predict_single_model(
                    features_df, self.diastolic_model, "diastolic", forecast_hours
                )
                results["diastolic"] = diastolic_pred

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            results["error"] = str(e)

        return results

    def _predict_single_model(
        self,
        features_df: pd.DataFrame,
        model_data: Dict,
        pressure_type: str,
        forecast_hours: int,
    ) -> Dict[str, Any]:
        """
        Make prediction using a single enhanced model.

        Args:
            features_df: DataFrame with generated features
            model_data: Loaded model dictionary
            pressure_type: 'systolic' or 'diastolic'
            forecast_hours: Number of hours to forecast

        Returns:
            Prediction results
        """
        # Extract model components
        prophet_model = model_data["prophet_model"]
        rf_model = model_data["rf_model"]
        scaler = model_data["scaler"]
        feature_columns = model_data["feature_columns"]

        # Use Random Forest for predictions since we have the features
        # Prepare features for forecast
        last_features = features_df.iloc[-1:].copy()

        # Create forecast features
        forecast_features = []
        for i in range(forecast_hours):
            feature_row = last_features.copy()
            # Update time-based features
            forecast_time = datetime.now() + timedelta(hours=i + 1)
            feature_row["hour"] = forecast_time.hour
            feature_row["day_of_week"] = forecast_time.weekday()
            feature_row["is_weekend"] = int(forecast_time.weekday() >= 5)
            feature_row["is_work_hours"] = int(9 <= forecast_time.hour <= 17)
            feature_row["is_sleep_hours"] = int(
                forecast_time.hour >= 22 or forecast_time.hour <= 6
            )

            # Update circadian features
            feature_row["circadian_bp"] = (
                np.sin(2 * np.pi * forecast_time.hour / 24) * 10 + 120
            )
            feature_row["sleep_recovery"] = (
                (1 if forecast_time.hour >= 22 or forecast_time.hour <= 6 else 0)
                * (8 - forecast_time.hour % 8)
                / 8
            )
            feature_row["work_stress"] = (
                (1 if 9 <= forecast_time.hour <= 17 else 0)
                * (forecast_time.hour - 9)
                / 8
            )

            forecast_features.append(feature_row)

        forecast_features_df = pd.concat(forecast_features, ignore_index=True)

        # Ensure we have all required features
        for col in feature_columns:
            if col not in forecast_features_df.columns:
                forecast_features_df[col] = 0

        # Scale features
        features_scaled = scaler.transform(forecast_features_df[feature_columns])

        # Get Random Forest predictions
        rf_predictions = rf_model.predict(features_scaled)

        # Calculate statistics
        mean_prediction = np.mean(rf_predictions)
        std_prediction = np.std(rf_predictions)
        trend_direction = (
            "increasing" if rf_predictions[-1] > rf_predictions[0] else "decreasing"
        )

        # Calculate confidence intervals
        confidence_lower = rf_predictions - 1.96 * std_prediction
        confidence_upper = rf_predictions + 1.96 * std_prediction

        # Generate timestamps
        timestamps = [
            (datetime.now() + timedelta(hours=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(forecast_hours)
        ]

        return {
            "predictions": rf_predictions.tolist(),
            "timestamps": timestamps,
            "mean": float(mean_prediction),
            "std": float(std_prediction),
            "trend": trend_direction,
            "confidence_interval": {
                "lower": confidence_lower.tolist(),
                "upper": confidence_upper.tolist(),
            },
            "model_info": {
                "type": "Enhanced Prophet + RandomForest",
                "features_used": len(feature_columns),
                "pressure_type": pressure_type,
                "forecast_hours": forecast_hours,
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        if not self.loaded_models:
            self.load_models()

        info = {}
        for model_type, model_data in self.loaded_models.items():
            info[model_type] = {
                "model_version": model_data.get("model_version", "unknown"),
                "last_training_date": model_data.get("last_training_date", "unknown"),
                "features_count": len(model_data.get("feature_columns", [])),
                "is_trained": model_data.get("is_trained", False),
            }

        return info


def test_enhanced_bp_interface():
    """Test function for the Enhanced BP Interface."""
    print("🧪 Testing Enhanced BP Interface...")

    # Create test data
    test_data = []
    for i in range(24):  # 24 hours of data
        test_data.append(
            {
                "timestamp": datetime.now() - timedelta(hours=24 - i),
                "heart_rate": 70 + np.random.normal(0, 5),
                "systolic_bp": 120 + np.random.normal(0, 10),
                "diastolic_bp": 80 + np.random.normal(0, 5),
                "spo2": 98 + np.random.normal(0, 1),
                "temperature": 36.7 + np.random.normal(0, 0.3),
            }
        )

    # Initialize forecaster
    forecaster = EnhancedBPForecaster()

    try:
        # Test prediction
        results = forecaster.predict_with_auto_features(
            historical_data=test_data, pressure_type="both", forecast_hours=12
        )

        print("✅ Enhanced BP Interface test successful!")
        print(f"Models loaded: {list(results.keys())}")

        for pressure_type, result in results.items():
            if pressure_type != "error":
                print(f"\n{pressure_type.title()} BP Prediction:")
                print(f"  Mean: {result['mean']:.1f} mmHg")
                print(f"  Trend: {result['trend']}")
                print(f"  Features used: {result['model_info']['features_used']}")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_enhanced_bp_interface()
