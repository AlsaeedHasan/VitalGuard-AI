"""
Training script for the anomaly detection model.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ml.anomaly_detection import VitalAnomalyDetector, generate_training_data
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Train and save the enhanced anomaly detection model."""
    logger.info("Starting enhanced anomaly detection model training...")

    # Create output directory
    models_dir = project_root / "ml" / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate enhanced training data
    logger.info("Generating enhanced synthetic training data...")
    training_data = generate_training_data(
        n_samples=3000, random_state=42
    )  # More samples
    logger.info(f"Generated {len(training_data)} training samples")

    # Initialize and train enhanced model
    detector = VitalAnomalyDetector(
        contamination=0.15, random_state=42
    )  # Slightly higher for better detection

    try:
        detector.train(training_data)
        logger.info("Enhanced model training completed successfully")

        # Enhanced testing with multiple scenarios
        logger.info("\n=== TESTING ENHANCED ANOMALY DETECTION ===")

        # Test 1: Normal vitals
        logger.info("\n--- Test 1: Normal Patient ---")
        test_vitals_normal = {
            "heart_rate": 72.0,
            "spo2": 98.5,
            "temperature": 36.7,
            "systolic_bp": 118.0,
            "diastolic_bp": 76.0,
        }

        result_normal = detector.predict(test_vitals_normal)
        logger.info(f"Normal patient result:")
        logger.info(f"  Is Anomaly: {result_normal['is_anomaly']}")
        logger.info(f"  Anomaly Score: {result_normal['anomaly_score']:.3f}")
        logger.info(f"  Confidence: {result_normal['confidence']:.3f}")
        logger.info(f"  Severity: {result_normal['severity']}")
        logger.info(f"  Decision Score: {result_normal['decision_score']:.3f}")

        # Test 2: Mild anomaly (elderly patient)
        logger.info("\n--- Test 2: Mild Anomaly (Elderly Pattern) ---")
        test_vitals_mild = {
            "heart_rate": 95.0,
            "spo2": 96.0,
            "temperature": 36.9,
            "systolic_bp": 145.0,
            "diastolic_bp": 88.0,
        }

        result_mild = detector.predict(test_vitals_mild)
        logger.info(f"Mild anomaly result:")
        logger.info(f"  Is Anomaly: {result_mild['is_anomaly']}")
        logger.info(f"  Anomaly Score: {result_mild['anomaly_score']:.3f}")
        logger.info(f"  Confidence: {result_mild['confidence']:.3f}")
        logger.info(f"  Severity: {result_mild['severity']}")

        # Test 3: Severe anomaly (critical condition)
        logger.info("\n--- Test 3: Severe Anomaly (Critical Condition) ---")
        test_vitals_severe = {
            "heart_rate": 155.0,
            "spo2": 88.0,
            "temperature": 39.2,
            "systolic_bp": 85.0,
            "diastolic_bp": 45.0,
        }

        result_severe = detector.predict(test_vitals_severe)
        logger.info(f"Severe anomaly result:")
        logger.info(f"  Is Anomaly: {result_severe['is_anomaly']}")
        logger.info(f"  Anomaly Score: {result_severe['anomaly_score']:.3f}")
        logger.info(f"  Confidence: {result_severe['confidence']:.3f}")
        logger.info(f"  Severity: {result_severe['severity']}")
        logger.info(f"  Most anomalous vitals:")
        for vital, score in sorted(
            result_severe["vital_scores"].items(), key=lambda x: x[1], reverse=True
        )[:3]:
            logger.info(f"    {vital}: {score:.3f}")

        # Test 4: Hypoxia case
        logger.info("\n--- Test 4: Hypoxia Case ---")
        test_vitals_hypoxia = {
            "heart_rate": 110.0,
            "spo2": 82.0,  # Critical low
            "temperature": 36.8,
            "systolic_bp": 125.0,
            "diastolic_bp": 80.0,
        }

        result_hypoxia = detector.predict(test_vitals_hypoxia)
        logger.info(f"Hypoxia case result:")
        logger.info(f"  Is Anomaly: {result_hypoxia['is_anomaly']}")
        logger.info(f"  Anomaly Score: {result_hypoxia['anomaly_score']:.3f}")
        logger.info(f"  Severity: {result_hypoxia['severity']}")
        logger.info(
            f"  SpO2 contribution: {result_hypoxia['vital_scores']['spo2']:.3f}"
        )

        # Save the enhanced model
        model_path = models_dir / "anomaly_detector.joblib"
        detector.save_model(str(model_path))
        logger.info(f"\nEnhanced model saved to {model_path}")

        # Model performance summary
        logger.info("\n=== MODEL PERFORMANCE SUMMARY ===")
        logger.info(f"Training samples: {detector.training_stats['n_samples']}")
        logger.info(f"Decision threshold: {detector.decision_threshold:.4f}")
        logger.info("Feature importance rankings:")
        for feature, importance in sorted(
            detector.feature_importance.items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"  {feature}: {importance:.3f}")

        # Consistency check
        logger.info("\n=== CONSISTENCY CHECKS ===")
        consistency_issues = []

        # Check if normal case is correctly classified
        if result_normal["is_anomaly"]:
            consistency_issues.append("Normal case incorrectly classified as anomaly")

        # Check if severe case is correctly classified
        if not result_severe["is_anomaly"]:
            consistency_issues.append("Severe case not detected as anomaly")

        # Check score consistency
        if result_normal["anomaly_score"] > result_severe["anomaly_score"]:
            consistency_issues.append(
                "Normal case has higher anomaly score than severe case"
            )

        if consistency_issues:
            logger.warning("Consistency issues found:")
            for issue in consistency_issues:
                logger.warning(f"  ⚠️  {issue}")
        else:
            logger.info("✅ All consistency checks passed!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    logger.info("Enhanced anomaly detection model training completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
