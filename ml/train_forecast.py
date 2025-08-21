"""
Training script for the forecasting models.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.forecasting import MultiVitalForecaster, generate_time_series_data
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Train and save the enhanced forecasting models."""
    logger.info("Starting enhanced forecasting models training...")

    # Create output directory
    models_dir = project_root / "ml" / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate more comprehensive time series training data
    logger.info("Generating enhanced synthetic time series data...")
    training_data = generate_time_series_data(
        n_days=90, freq_hours=1, random_state=42
    )  # More data
    logger.info(f"Generated {len(training_data)} time series samples")

    # Initialize multi-vital forecaster
    vital_types = ["heart_rate", "spo2", "temperature", "systolic_bp", "diastolic_bp"]
    forecaster = MultiVitalForecaster(vital_types)

    try:
        # Train all forecasters
        logger.info("Training enhanced forecasting models...")
        forecaster.train_all(training_data)

        # Enhanced testing with multiple horizons
        logger.info("Testing forecasting models with multiple horizons...")
        recent_data = training_data[-72:]  # Last 72 hours for context

        # Test different forecast horizons
        horizons = [12, 24, 48]  # 12h, 24h, 48h

        for horizon in horizons:
            logger.info(f"\n--- Testing {horizon}-hour forecasts ---")
            forecast_results = forecaster.predict_all(
                recent_data, forecast_horizon=horizon
            )

            for vital_type, result in forecast_results.items():
                if result:
                    perf = result["model_performance"]
                    logger.info(f"Forecast for {vital_type} ({horizon}h horizon):")
                    logger.info(f"  Trend: {result['trend_direction']}")
                    logger.info(f"  R²: {perf['r_squared']:.3f}")
                    logger.info(f"  RMSE: {perf['rmse']:.2f}")
                    logger.info(f"  MAE: {perf['mae']:.2f}")
                    logger.info(f"  MAPE: {perf['mape']:.1f}%")
                    logger.info(
                        f"  Direction Accuracy: {perf['direction_accuracy']:.3f}"
                    )
                    logger.info(
                        f"  Data Quality Score: {perf['data_quality_score']:.3f}"
                    )

                    # Interpret results
                    if perf["r_squared"] < 0:
                        logger.warning(
                            f"  ⚠️  Negative R² indicates poor model performance!"
                        )
                    elif perf["r_squared"] < 0.3:
                        logger.warning(
                            f"  ⚠️  Low R² ({perf['r_squared']:.3f}) - model needs improvement"
                        )
                    elif perf["r_squared"] > 0.7:
                        logger.info(
                            f"  ✅  Good R² ({perf['r_squared']:.3f}) - reliable forecasting"
                        )

                else:
                    logger.warning(f"No forecast result for {vital_type}")

        # Save all models
        logger.info("\nSaving enhanced forecasting models...")
        forecaster.save_all_models(str(models_dir))

        # Summary report
        logger.info("\n=== FORECASTING TRAINING SUMMARY ===")
        best_performers = []
        poor_performers = []

        # Use 24h horizon for summary
        forecast_results = forecaster.predict_all(recent_data, forecast_horizon=24)

        for vital_type, result in forecast_results.items():
            if result:
                r2 = result["model_performance"]["r_squared"]
                if r2 > 0.5:
                    best_performers.append((vital_type, r2))
                elif r2 < 0.1:
                    poor_performers.append((vital_type, r2))

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
