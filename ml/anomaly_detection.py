"""
Anomaly detection using Isolation Forest for patient vital signs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Any
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VitalAnomalyDetector:
    """Enhanced anomaly detector for patient vital signs using Isolation Forest."""

    def __init__(self, contamination: float = 0.25, random_state: int = 42):
        """
        Initialize the enhanced anomaly detector.

        Args:
            contamination: Expected proportion of outliers in the data (optimized value 0.25 for better borderline detection)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state

        # Enhanced Isolation Forest configuration
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=200,  # Increased for better performance
            max_samples="auto",
            max_features=1.0,
            bootstrap=False,
            warm_start=False,
        )

        self.scaler = StandardScaler()
        self.feature_names = [
            "heart_rate",
            "spo2",
            "temperature",
            "systolic_bp",
            "diastolic_bp",
        ]
        self.is_trained = False
        self.decision_threshold = 0.0  # Will be calculated during training
        self.feature_importance = {}
        self.training_stats = {}

    def prepare_features(self, vitals_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Prepare feature matrix from vital signs data.

        Args:
            vitals_data: List of vital signs dictionaries

        Returns:
            DataFrame with features ready for training/prediction
        """
        df = pd.DataFrame(vitals_data)

        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in df.columns:
                logger.warning(f"Missing feature: {feature}, filling with median")
                df[feature] = (
                    df[self.feature_names].median().iloc[0] if not df.empty else 0
                )

        # Select only the required features
        feature_df = df[self.feature_names].copy()

        # Handle missing values
        feature_df = feature_df.fillna(feature_df.median())

        # Add derived features
        feature_df["pulse_pressure"] = (
            feature_df["systolic_bp"] - feature_df["diastolic_bp"]
        )
        feature_df["mean_arterial_pressure"] = (
            feature_df["systolic_bp"] + 2 * feature_df["diastolic_bp"]
        ) / 3

        # Add borderline detection features (NEW)
        feature_df["borderline_hypertension"] = (
            (feature_df["systolic_bp"] >= 140) | (feature_df["diastolic_bp"] >= 90)
        ).astype(int)
        feature_df["mild_hypoxia"] = (feature_df["spo2"] <= 94).astype(int)
        feature_df["mild_tachycardia"] = (feature_df["heart_rate"] >= 100).astype(int)
        feature_df["mild_fever"] = (feature_df["temperature"] >= 37.5).astype(int)
        feature_df["mild_hypothermia"] = (feature_df["temperature"] <= 36.0).astype(int)

        # Normalized pulse pressure (normal range: 30-50 mmHg)
        feature_df["pulse_pressure_norm"] = feature_df["pulse_pressure"] / 40.0

        # Normalized MAP (normal range: 70-100 mmHg)
        feature_df["map_norm"] = feature_df["mean_arterial_pressure"] / 85.0

        # Risk indicators
        feature_df["bp_risk"] = (
            (feature_df["systolic_bp"] >= 130) | (feature_df["diastolic_bp"] >= 80)
        ).astype(int)
        feature_df["oxygen_risk"] = (feature_df["spo2"] <= 95).astype(int)

        return feature_df

    def train(self, training_data: List[Dict[str, Any]]) -> None:
        """
        Enhanced training for the anomaly detection model.

        Args:
            training_data: List of vital signs data for training
        """
        logger.info(
            f"Training enhanced anomaly detector with {len(training_data)} samples"
        )

        # Prepare features
        feature_df = self.prepare_features(training_data)

        if feature_df.empty:
            raise ValueError("No valid training data provided")

        # Calculate training statistics
        self.training_stats = {
            "n_samples": len(feature_df),
            "feature_means": feature_df.mean().to_dict(),
            "feature_stds": feature_df.std().to_dict(),
            "feature_mins": feature_df.min().to_dict(),
            "feature_maxs": feature_df.max().to_dict(),
        }

        # Scale features
        scaled_features = self.scaler.fit_transform(feature_df)

        # Train model
        self.model.fit(scaled_features)

        # Calculate optimal threshold using training data
        decision_scores = self.model.decision_function(scaled_features)

        # Use percentile-based threshold for better performance
        self.decision_threshold = np.percentile(
            decision_scores, (1 - self.contamination) * 100
        )

        # Calculate feature importance (approximate)
        self._calculate_feature_importance(scaled_features, decision_scores)

        self.is_trained = True

        logger.info("Enhanced anomaly detector training completed")
        logger.info(f"Decision threshold: {self.decision_threshold:.4f}")

    def _calculate_feature_importance(
        self, scaled_features: np.ndarray, decision_scores: np.ndarray
    ) -> None:
        """Calculate approximate feature importance scores."""
        n_features = scaled_features.shape[1]
        feature_names_extended = self.feature_names + [
            "pulse_pressure",
            "mean_arterial_pressure",
            "borderline_hypertension",
            "mild_hypoxia",
            "mild_tachycardia",
            "mild_fever",
            "mild_hypothermia",
            "pulse_pressure_norm",
            "map_norm",
            "bp_risk",
            "oxygen_risk",
        ]

        importance_scores = {}

        for i in range(min(n_features, len(feature_names_extended))):
            # Calculate correlation between feature values and anomaly scores
            feature_values = scaled_features[:, i]
            correlation = abs(np.corrcoef(feature_values, decision_scores)[0, 1])
            importance_scores[feature_names_extended[i]] = (
                correlation if not np.isnan(correlation) else 0.0
            )

        # Normalize importance scores
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            self.feature_importance = {
                k: v / total_importance for k, v in importance_scores.items()
            }
        else:
            self.feature_importance = {
                k: 1.0 / len(importance_scores) for k in importance_scores
            }

    def predict(self, vitals: Dict[str, float]) -> Dict[str, Any]:
        """
        Enhanced prediction with improved confidence and threshold logic.

        Args:
            vitals: Dictionary of vital signs

        Returns:
            Dictionary containing enhanced anomaly prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare single sample
        feature_df = self.prepare_features([vitals])
        scaled_features = self.scaler.transform(feature_df)

        # Make prediction
        anomaly_prediction = self.model.predict(scaled_features)[0]
        decision_score = self.model.decision_function(scaled_features)[0]

        # Enhanced anomaly score calculation
        # Convert decision score to 0-1 probability-like score
        # Decision scores below threshold are anomalies
        is_anomaly = decision_score < self.decision_threshold

        # Normalize score to 0-1 range where 1 = most anomalous
        if decision_score < self.decision_threshold:
            # Anomalous: score increases as decision_score decreases below threshold
            if abs(self.decision_threshold) > 1e-8:
                anomaly_score = 0.5 + (self.decision_threshold - decision_score) / (
                    2 * abs(self.decision_threshold)
                )
            else:
                anomaly_score = 1.0  # Maximum anomaly if threshold is near zero
        else:
            # Normal: score decreases as decision_score increases above threshold
            anomaly_score = 0.5 * (self.decision_threshold / max(decision_score, 1e-8))

        # Ensure score is in [0, 1] range
        anomaly_score = max(0.0, min(1.0, anomaly_score))

        # Enhanced confidence calculation
        confidence = self._calculate_confidence(decision_score, scaled_features[0])

        # Get individual feature contributions
        vital_scores = self._get_enhanced_feature_contributions(
            vitals, scaled_features[0]
        )

        # Calculate severity level
        severity = self._calculate_severity(anomaly_score, vital_scores)

        return {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": float(anomaly_score),
            "confidence": float(confidence),
            "severity": severity,
            "vital_scores": vital_scores,
            "threshold": 0.5,
            "decision_score": float(decision_score),
            "decision_threshold": float(self.decision_threshold),
            "feature_importance": self.feature_importance,
        }

    def _calculate_confidence(
        self, decision_score: float, scaled_sample: np.ndarray
    ) -> float:
        """Calculate prediction confidence based on multiple factors."""
        # Distance from decision boundary
        boundary_distance = abs(decision_score - self.decision_threshold)

        # Safe division to avoid divide by zero warning
        if abs(self.decision_threshold) > 1e-8:
            boundary_confidence = min(
                1.0, boundary_distance / abs(self.decision_threshold)
            )
        else:
            boundary_confidence = 0.5  # Default confidence when threshold is near zero

        # Feature consistency (how many features are unusual)
        feature_consistency = 1.0 - (
            np.mean(np.abs(scaled_sample) > 2.0)
        )  # Less than 2 std devs

        # Model agreement (simulate ensemble agreement)
        model_confidence = 0.8  # Base confidence for single model

        # Combined confidence
        confidence = (
            boundary_confidence * 0.4
            + feature_consistency * 0.4
            + model_confidence * 0.2
        )

        return max(0.1, min(1.0, confidence))

    def _calculate_severity(
        self,
        anomaly_score: float,
        vitals: Dict[str, float],
        rule_based_severity: bool = None,
    ) -> str:
        """Calculate anomaly severity level."""

        # إذا تم تحديد الشدة من القواعد، استخدمها أولاً
        if rule_based_severity:
            return rule_based_severity

        # فحص القيم الحرجة أولاً
        if vitals.get("spo2", 100) < 90:
            return "Critical"
        elif (
            vitals.get("systolic_bp", 0) >= 180 or vitals.get("diastolic_bp", 0) >= 110
        ):
            return "Critical"
        elif vitals.get("heart_rate", 0) >= 150:
            return "Critical"
        elif vitals.get("temperature", 0) >= 39.5:
            return "Critical"

        # فحص القيم المتوسطة
        elif vitals.get("spo2", 100) <= 94:
            return "Moderate"
        elif (
            vitals.get("systolic_bp", 0) >= 160 or vitals.get("diastolic_bp", 0) >= 100
        ):
            return "Moderate"
        elif vitals.get("heart_rate", 0) >= 120:
            return "Moderate"
        elif vitals.get("temperature", 0) >= 38.5:
            return "Moderate"

        # فحص القيم الخفيفة
        elif vitals.get("systolic_bp", 0) >= 130 or vitals.get("diastolic_bp", 0) >= 80:
            return "Mild"
        elif vitals.get("spo2", 100) <= 95:
            return "Mild"
        elif vitals.get("heart_rate", 0) >= 100:
            return "Mild"
        elif vitals.get("temperature", 0) >= 37.5:
            return "Mild"
        elif vitals.get("systolic_bp", 0) < 90 or vitals.get("diastolic_bp", 0) < 60:
            return "Mild"

        # استخدام درجة الشذوذ كمعيار أخير
        elif anomaly_score < 0.3:
            return "Normal"
        elif anomaly_score < 0.5:
            return "Mild"
        elif anomaly_score < 0.7:
            return "Moderate"
        elif anomaly_score < 0.9:
            return "High"
        else:
            return "Critical"

    def _get_enhanced_feature_contributions(
        self, vitals: Dict[str, float], scaled_sample: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate enhanced feature contributions to anomaly score.

        Args:
            vitals: Original vital signs
            scaled_sample: Scaled feature vector

        Returns:
            Dictionary of feature contributions with medical context
        """
        contributions = {}
        feature_names_extended = self.feature_names + [
            "pulse_pressure",
            "mean_arterial_pressure",
            "borderline_hypertension",
            "mild_hypoxia",
            "mild_tachycardia",
            "mild_fever",
            "mild_hypothermia",
            "pulse_pressure_norm",
            "map_norm",
            "bp_risk",
            "oxygen_risk",
        ]

        for i, feature_name in enumerate(feature_names_extended):
            if i < len(scaled_sample):
                # Enhanced contribution calculation
                scaled_value = scaled_sample[i]

                # Use feature importance and scaled value
                importance = self.feature_importance.get(
                    feature_name, 1.0 / len(feature_names_extended)
                )

                # Calculate contribution based on deviation and importance
                base_contribution = min(
                    1.0, abs(scaled_value) / 3.0
                )  # 3 std devs = max contribution
                weighted_contribution = base_contribution * importance

                # Apply medical knowledge for bounds checking
                medical_contribution = self._get_medical_contribution(
                    feature_name, vitals
                )

                # Combine both approaches
                final_contribution = max(weighted_contribution, medical_contribution)
                contributions[feature_name] = float(final_contribution)
            else:
                contributions[feature_name] = 0.0

        return contributions

    def _get_medical_contribution(
        self, feature_name: str, vitals: Dict[str, float]
    ) -> float:
        """Calculate medical-knowledge-based contribution scores."""
        # Medical thresholds for anomaly detection
        medical_thresholds = {
            "heart_rate": {
                "critical_low": 40,
                "low": 50,
                "high": 120,
                "critical_high": 150,
            },
            "spo2": {"critical_low": 85, "low": 90, "high": 100, "critical_high": 100},
            "temperature": {
                "critical_low": 35.0,
                "low": 36.0,
                "high": 38.0,
                "critical_high": 40.0,
            },
            "systolic_bp": {
                "critical_low": 70,
                "low": 90,
                "high": 140,
                "critical_high": 180,
            },
            "diastolic_bp": {
                "critical_low": 40,
                "low": 60,
                "high": 90,
                "critical_high": 110,
            },
        }

        if feature_name == "pulse_pressure":
            # Calculate pulse pressure
            pp = vitals.get("systolic_bp", 120) - vitals.get("diastolic_bp", 80)
            if pp < 30 or pp > 70:
                return 0.8
            elif pp < 35 or pp > 60:
                return 0.4
            else:
                return 0.1

        elif feature_name == "mean_arterial_pressure":
            # Calculate MAP
            sbp = vitals.get("systolic_bp", 120)
            dbp = vitals.get("diastolic_bp", 80)
            map_value = (sbp + 2 * dbp) / 3
            if map_value < 65 or map_value > 110:
                return 0.8
            elif map_value < 70 or map_value > 100:
                return 0.4
            else:
                return 0.1

        elif feature_name in medical_thresholds and feature_name in vitals:
            value = vitals[feature_name]
            thresholds = medical_thresholds[feature_name]

            if (
                value <= thresholds["critical_low"]
                or value >= thresholds["critical_high"]
            ):
                return 1.0
            elif value <= thresholds["low"] or value >= thresholds["high"]:
                return 0.6
            else:
                return 0.1

        return 0.0

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "random_state": self.random_state,
            "trained_at": datetime.now().isoformat(),
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_names = model_data["feature_names"]
        self.contamination = model_data["contamination"]
        self.random_state = model_data["random_state"]
        self.is_trained = True

        logger.info(f"Model loaded from {filepath}")
        logger.info(f"Model trained at: {model_data.get('trained_at', 'Unknown')}")

    def hybrid_predict(self, vitals: Dict[str, float]) -> Dict[str, Any]:
        """
        Hybrid anomaly detection combining ML model with medical rules for borderline cases.
        """
        # Get basic ML prediction
        basic_result = self.predict(vitals)

        # Apply medical rules for borderline cases
        rule_based_flags = []
        rule_based_severity = None
        rule_based_confidence_boost = 0.0

        # Extract values for easier access
        hr = vitals.get("heart_rate", 0)
        spo2 = vitals.get("spo2", 100)
        temp = vitals.get("temperature", 0)
        sbp = vitals.get("systolic_bp", 0)
        dbp = vitals.get("diastolic_bp", 0)

        # 1. Critical cases (highest priority)
        if spo2 < 90:
            rule_based_flags.append("نقص أكسجين حاد")
            rule_based_severity = "Critical"
            rule_based_confidence_boost = 0.3
        elif sbp >= 180 or dbp >= 110:
            rule_based_flags.append("ارتفاع ضغط حاد")
            rule_based_severity = "Critical"
            rule_based_confidence_boost = 0.3
        elif hr >= 150:
            rule_based_flags.append("تسارع قلب حاد")
            rule_based_severity = "Critical"
            rule_based_confidence_boost = 0.3
        elif temp >= 39.5:
            rule_based_flags.append("حمى عالية")
            rule_based_severity = "Critical"
            rule_based_confidence_boost = 0.3

        # 2. Moderate cases
        elif spo2 <= 94:
            rule_based_flags.append("انخفاض أكسجين متوسط")
            rule_based_severity = "Moderate"
            rule_based_confidence_boost = 0.25
        elif sbp >= 160 or dbp >= 100:
            rule_based_flags.append("ارتفاع ضغط متوسط")
            rule_based_severity = "Moderate"
            rule_based_confidence_boost = 0.25
        elif hr >= 120:
            rule_based_flags.append("تسارع قلب متوسط")
            rule_based_severity = "Moderate"
            rule_based_confidence_boost = 0.25
        elif temp >= 38.5:
            rule_based_flags.append("حمى متوسطة")
            rule_based_severity = "Moderate"
            rule_based_confidence_boost = 0.25

        # 3. Mild/Borderline cases (تعديل العتبات)
        elif sbp >= 130 or dbp >= 80:  # خفض العتبة من 140 إلى 130 و 90 إلى 80
            rule_based_flags.append("ارتفاع ضغط مرحلة أولى")
            rule_based_severity = "Mild"
            rule_based_confidence_boost = 0.15
        elif sbp >= 140 or dbp >= 90:  # عتبة أعلى للحالات الحدية
            rule_based_flags.append("ارتفاع ضغط حدي")
            rule_based_severity = "Mild"
            rule_based_confidence_boost = 0.2
        elif spo2 <= 95:
            rule_based_flags.append("انخفاض أكسجين خفيف")
            rule_based_severity = "Mild"
            rule_based_confidence_boost = 0.15
        elif hr >= 100:
            rule_based_flags.append("تسارع قلب خفيف")
            rule_based_severity = "Mild"
            rule_based_confidence_boost = 0.15
        elif temp >= 37.5:
            rule_based_flags.append("حمى خفيفة")
            rule_based_severity = "Mild"
            rule_based_confidence_boost = 0.15

        # 4. Special cases
        # Pregnancy with hypertension (تعديل الشرط)
        if sbp >= 130 and dbp >= 80:  # خفض العتبة
            rule_based_flags.append("حامل مع ارتفاع ضغط")
            rule_based_severity = "Moderate"  # خطورة أعلى للحوامل
            rule_based_confidence_boost = max(rule_based_confidence_boost, 0.25)

        # 5. Low blood pressure
        elif sbp < 90 or dbp < 60:
            rule_based_flags.append("انخفاض ضغط خفيف")
            rule_based_severity = "Mild"
            rule_based_confidence_boost = 0.15

        # 6. Derived metrics
        pulse_pressure = sbp - dbp
        if pulse_pressure < 30 or pulse_pressure > 60:
            rule_based_flags.append("ضغط نبض غير طبيعي")
            if rule_based_severity is None:
                rule_based_severity = "Mild"
            rule_based_confidence_boost = max(rule_based_confidence_boost, 0.1)

        # Calculate MAP
        map_pressure = (sbp + 2 * dbp) / 3
        if map_pressure < 70 or map_pressure > 100:
            rule_based_flags.append("ضغط شرياني متوسط غير طبيعي")
            if rule_based_severity is None:
                rule_based_severity = "Mild"
            rule_based_confidence_boost = max(rule_based_confidence_boost, 0.1)

        # If rules detected anomalies but ML didn't, override the result
        if rule_based_flags and not basic_result["is_anomaly"]:
            basic_result["is_anomaly"] = True
            basic_result["severity"] = rule_based_severity or "Mild"
            basic_result["anomaly_score"] = max(basic_result["anomaly_score"], 0.4)
            basic_result["confidence"] = min(
                basic_result["confidence"] + rule_based_confidence_boost, 1.0
            )
            basic_result["rule_based_flags"] = rule_based_flags
            basic_result["detection_method"] = "hybrid_rules"
        elif rule_based_flags and basic_result["is_anomaly"]:
            # Enhance existing anomaly detection with rule-based information
            basic_result["rule_based_flags"] = rule_based_flags
            basic_result["confidence"] = min(
                basic_result["confidence"] + rule_based_confidence_boost, 1.0
            )
            basic_result["detection_method"] = "hybrid_enhanced"
        else:
            # For normal cases, ensure severity is Normal
            if not basic_result["is_anomaly"]:
                basic_result["severity"] = "Normal"
            basic_result["detection_method"] = "ml_only"

        return basic_result


def generate_training_data(
    n_samples: int = 1000, random_state: int = 42
) -> List[Dict[str, float]]:
    """
    Generate enhanced synthetic training data with realistic medical patterns.

    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        List of synthetic vital signs data with realistic patterns
    """
    np.random.seed(random_state)

    # Enhanced normal ranges with age and condition variations
    normal_ranges = {
        "heart_rate": {"mean": 75, "std": 12, "min": 50, "max": 100},
        "spo2": {"mean": 98, "std": 1.2, "min": 95, "max": 100},
        "temperature": {"mean": 36.8, "std": 0.3, "min": 36.1, "max": 37.2},
        "systolic_bp": {"mean": 120, "std": 12, "min": 95, "max": 140},
        "diastolic_bp": {"mean": 80, "std": 8, "min": 65, "max": 90},
    }

    training_data = []

    # Generate normal samples with realistic correlations
    for i in range(n_samples):
        # Patient baseline variations (simulate different patient types)
        patient_type = np.random.choice(
            ["healthy", "elderly", "athlete", "hypertensive"], p=[0.6, 0.2, 0.1, 0.1]
        )

        vitals = {}

        # Adjust ranges based on patient type
        if patient_type == "elderly":
            bp_adjustment = 15
            hr_adjustment = -5
            temp_adjustment = -0.1
        elif patient_type == "athlete":
            bp_adjustment = -10
            hr_adjustment = -15
            temp_adjustment = 0
        elif patient_type == "hypertensive":
            bp_adjustment = 25
            hr_adjustment = 5
            temp_adjustment = 0
        else:  # healthy
            bp_adjustment = 0
            hr_adjustment = 0
            temp_adjustment = 0

        # Generate correlated vital signs
        base_hr = np.random.normal(
            normal_ranges["heart_rate"]["mean"] + hr_adjustment,
            normal_ranges["heart_rate"]["std"],
        )
        vitals["heart_rate"] = np.clip(base_hr, 45, 120)

        # SpO2 (generally stable, slight correlation with heart rate)
        spo2_base = normal_ranges["spo2"]["mean"]
        if vitals["heart_rate"] > 100:  # Slight decrease with high HR
            spo2_base -= 0.5
        vitals["spo2"] = np.clip(
            np.random.normal(spo2_base, normal_ranges["spo2"]["std"]), 94, 100
        )

        # Temperature (mostly independent)
        vitals["temperature"] = np.clip(
            np.random.normal(
                normal_ranges["temperature"]["mean"] + temp_adjustment,
                normal_ranges["temperature"]["std"],
            ),
            35.8,
            37.5,
        )

        # Blood pressure (correlated)
        systolic_base = normal_ranges["systolic_bp"]["mean"] + bp_adjustment
        # Add slight correlation with heart rate
        systolic_base += (vitals["heart_rate"] - 75) * 0.2

        vitals["systolic_bp"] = np.clip(
            np.random.normal(systolic_base, normal_ranges["systolic_bp"]["std"]),
            85,
            150,
        )

        # Diastolic correlated with systolic
        diastolic_ratio = np.random.normal(0.65, 0.05)  # Normal ratio with variation
        vitals["diastolic_bp"] = np.clip(
            vitals["systolic_bp"] * diastolic_ratio + np.random.normal(0, 3),
            55,
            min(vitals["systolic_bp"] - 20, 95),
        )

        # Round values
        for key in vitals:
            if key == "temperature":
                vitals[key] = round(float(vitals[key]), 2)
            else:
                vitals[key] = round(float(vitals[key]), 1)

        training_data.append(vitals)

    # Add realistic anomalous samples
    n_anomalies = int(
        n_samples * 0.20
    )  # Increased from 0.15 to include more borderline cases

    anomaly_types = [
        "fever",
        "hypotension",
        "hypertension",
        "tachycardia",
        "bradycardia",
        "hypoxia",
        "mixed_shock",
        "sepsis_like",
        # NEW: Borderline cases
        "borderline_hypertension",
        "mild_hypoxia",
        "mild_tachycardia",
        "stage1_hypertension",
        "elevated_bp",
    ]

    for _ in range(n_anomalies):
        anomaly_type = np.random.choice(anomaly_types)
        vitals = {}

        if anomaly_type == "fever":
            vitals["temperature"] = np.random.uniform(38.0, 40.5)
            vitals["heart_rate"] = np.random.uniform(90, 130)  # Elevated with fever
            vitals["spo2"] = np.random.uniform(94, 99)
            vitals["systolic_bp"] = np.random.uniform(110, 140)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(0.6, 0.7)

        elif anomaly_type == "hypotension":
            vitals["systolic_bp"] = np.random.uniform(70, 90)
            vitals["diastolic_bp"] = np.random.uniform(40, 60)
            vitals["heart_rate"] = np.random.uniform(
                100, 140
            )  # Compensatory tachycardia
            vitals["temperature"] = np.random.uniform(36.0, 37.5)
            vitals["spo2"] = np.random.uniform(92, 98)

        elif anomaly_type == "hypertension":
            vitals["systolic_bp"] = np.random.uniform(160, 200)
            vitals["diastolic_bp"] = np.random.uniform(100, 120)
            vitals["heart_rate"] = np.random.uniform(70, 95)
            vitals["temperature"] = np.random.uniform(36.5, 37.2)
            vitals["spo2"] = np.random.uniform(96, 100)

        elif anomaly_type == "tachycardia":
            vitals["heart_rate"] = np.random.uniform(120, 180)
            vitals["systolic_bp"] = np.random.uniform(100, 150)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(
                0.6, 0.75
            )
            vitals["temperature"] = np.random.uniform(36.5, 38.0)
            vitals["spo2"] = np.random.uniform(94, 99)

        elif anomaly_type == "bradycardia":
            vitals["heart_rate"] = np.random.uniform(35, 50)
            vitals["systolic_bp"] = np.random.uniform(90, 130)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(
                0.65, 0.75
            )
            vitals["temperature"] = np.random.uniform(36.0, 37.0)
            vitals["spo2"] = np.random.uniform(95, 100)

        elif anomaly_type == "hypoxia":
            vitals["spo2"] = np.random.uniform(75, 90)
            vitals["heart_rate"] = np.random.uniform(90, 130)  # Compensatory
            vitals["systolic_bp"] = np.random.uniform(100, 150)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(0.6, 0.7)
            vitals["temperature"] = np.random.uniform(36.0, 37.5)

        elif anomaly_type == "mixed_shock":
            vitals["systolic_bp"] = np.random.uniform(65, 85)
            vitals["diastolic_bp"] = np.random.uniform(35, 55)
            vitals["heart_rate"] = np.random.uniform(110, 160)
            vitals["temperature"] = np.random.uniform(35.0, 36.5)  # Hypothermia
            vitals["spo2"] = np.random.uniform(85, 95)

        elif anomaly_type == "sepsis_like":
            vitals["temperature"] = np.random.choice(
                [
                    np.random.uniform(38.5, 41.0),  # Hyperthermia
                    np.random.uniform(34.0, 36.0),  # Hypothermia
                ]
            )
            vitals["heart_rate"] = np.random.uniform(100, 150)
            vitals["systolic_bp"] = np.random.uniform(80, 110)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(
                0.55, 0.7
            )
            vitals["spo2"] = np.random.uniform(88, 96)

        # NEW: Borderline cases for improved detection
        elif anomaly_type == "borderline_hypertension":
            vitals["systolic_bp"] = np.random.uniform(140, 159)  # Stage 1 hypertension
            vitals["diastolic_bp"] = np.random.uniform(90, 99)
            vitals["heart_rate"] = np.random.uniform(70, 90)
            vitals["temperature"] = np.random.uniform(36.5, 37.2)
            vitals["spo2"] = np.random.uniform(96, 100)

        elif anomaly_type == "stage1_hypertension":
            vitals["systolic_bp"] = np.random.uniform(130, 139)  # Stage 1 hypertension
            vitals["diastolic_bp"] = np.random.uniform(80, 89)
            vitals["heart_rate"] = np.random.uniform(65, 85)
            vitals["temperature"] = np.random.uniform(36.4, 37.0)
            vitals["spo2"] = np.random.uniform(97, 100)

        elif anomaly_type == "elevated_bp":
            vitals["systolic_bp"] = np.random.uniform(120, 129)  # Elevated BP
            vitals["diastolic_bp"] = np.random.uniform(75, 84)
            vitals["heart_rate"] = np.random.uniform(70, 85)
            vitals["temperature"] = np.random.uniform(36.5, 37.1)
            vitals["spo2"] = np.random.uniform(97, 100)

        elif anomaly_type == "mild_hypoxia":
            vitals["spo2"] = np.random.uniform(91, 94)  # Mild hypoxia
            vitals["heart_rate"] = np.random.uniform(80, 110)
            vitals["systolic_bp"] = np.random.uniform(110, 140)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(
                0.65, 0.75
            )
            vitals["temperature"] = np.random.uniform(36.2, 37.2)

        elif anomaly_type == "mild_tachycardia":
            vitals["heart_rate"] = np.random.uniform(100, 119)  # Mild tachycardia
            vitals["systolic_bp"] = np.random.uniform(115, 145)
            vitals["diastolic_bp"] = vitals["systolic_bp"] * np.random.uniform(
                0.65, 0.75
            )
            vitals["temperature"] = np.random.uniform(36.8, 37.8)
            vitals["spo2"] = np.random.uniform(95, 99)

        # Round values
        for key in vitals:
            if key == "temperature":
                vitals[key] = round(float(vitals[key]), 2)
            else:
                vitals[key] = round(float(vitals[key]), 1)

        training_data.append(vitals)

    # Shuffle the data
    np.random.shuffle(training_data)

    return training_data
