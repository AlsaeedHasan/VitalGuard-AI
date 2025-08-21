#!/usr/bin/env python3
"""
إعادة تدريب نموذج كشف الشذوذ المحسن مع التحسينات الجديدة
Enhanced Anomaly Detection Model Retraining Script
"""

import os
import sys
import logging
from datetime import datetime
from anomaly_detection import VitalAnomalyDetector, generate_training_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    print("🚀 بدء إعادة تدريب نموذج كشف الشذوذ المحسن")
    print("=" * 60)

    try:
        # Create enhanced detector with new contamination rate
        print("🔧 إنشاء نموذج كشف شذوذ محسن...")
        detector = VitalAnomalyDetector(contamination=0.25)

        # Generate enhanced training data with borderline cases
        print("📊 إنشاء بيانات تدريب محسنة...")
        training_data = generate_training_data(n_samples=2000, random_state=42)

        print(f"✅ تم إنشاء {len(training_data)} عينة تدريب")

        # Train the model
        print("🎯 بدء تدريب النموذج...")
        detector.train(training_data)

        # Save the enhanced model
        model_dir = "ml/saved_models"
        os.makedirs(model_dir, exist_ok=True)

        # Backup old model
        old_model_path = os.path.join(model_dir, "anomaly_detector.joblib")
        if os.path.exists(old_model_path):
            backup_path = os.path.join(
                model_dir,
                f"anomaly_detector_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib",
            )
            os.rename(old_model_path, backup_path)
            print(f"📦 تم حفظ النسخة الاحتياطية في: {backup_path}")

        # Save new enhanced model
        enhanced_model_path = os.path.join(model_dir, "anomaly_detector.joblib")
        detector.save_model(enhanced_model_path)

        print(f"💾 تم حفظ النموذج المحسن في: {enhanced_model_path}")

        # Test the enhanced model with borderline cases
        print("\n🧪 اختبار النموذج المحسن مع الحالات الحدية...")
        test_borderline_cases(detector)

        print("\n✅ تم إعادة التدريب بنجاح!")
        print("🎯 النموذج المحسن جاهز للاستخدام")

    except Exception as e:
        logger.error(f"خطأ في إعادة التدريب: {e}")
        print(f"❌ فشل في إعادة التدريب: {e}")
        sys.exit(1)


def test_borderline_cases(detector):
    """Test the enhanced model with borderline cases."""

    borderline_test_cases = [
        {
            "name": "ارتفاع ضغط حدي",
            "vitals": {
                "heart_rate": 85,
                "spo2": 97,
                "temperature": 36.8,
                "systolic_bp": 140,
                "diastolic_bp": 90,
            },
            "expected": "شذوذ",
        },
        {
            "name": "ارتفاع ضغط مرحلة أولى",
            "vitals": {
                "heart_rate": 78,
                "spo2": 98,
                "temperature": 36.7,
                "systolic_bp": 135,
                "diastolic_bp": 85,
            },
            "expected": "شذوذ",
        },
        {
            "name": "انخفاض أكسجين خفيف",
            "vitals": {
                "heart_rate": 82,
                "spo2": 94,
                "temperature": 36.9,
                "systolic_bp": 125,
                "diastolic_bp": 82,
            },
            "expected": "شذوذ",
        },
        {
            "name": "تسارع ضربات قلب خفيف",
            "vitals": {
                "heart_rate": 105,
                "spo2": 98,
                "temperature": 37.1,
                "systolic_bp": 128,
                "diastolic_bp": 84,
            },
            "expected": "شذوذ",
        },
        {
            "name": "مريض طبيعي",
            "vitals": {
                "heart_rate": 72,
                "spo2": 98,
                "temperature": 36.8,
                "systolic_bp": 120,
                "diastolic_bp": 80,
            },
            "expected": "طبيعي",
        },
    ]

    print("\n📋 اختبار الحالات الحدية:")
    print("-" * 40)

    correct_predictions = 0
    total_predictions = len(borderline_test_cases)

    for test_case in borderline_test_cases:
        # Test with basic predict
        basic_result = detector.predict(test_case["vitals"])

        # Test with hybrid predict
        hybrid_result = detector.hybrid_predict(test_case["vitals"])

        # Check if hybrid prediction is better
        expected_anomaly = test_case["expected"] == "شذوذ"
        basic_correct = basic_result["is_anomaly"] == expected_anomaly
        hybrid_correct = hybrid_result["is_anomaly"] == expected_anomaly

        # Count hybrid as the main result
        if hybrid_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"

        print(f"{status} {test_case['name']}:")
        print(
            f"   الأساسي: {'شذوذ' if basic_result['is_anomaly'] else 'طبيعي'} "
            f"(ثقة: {basic_result['confidence']:.1%})"
        )
        print(
            f"   الهجين: {'شذوذ' if hybrid_result['is_anomaly'] else 'طبيعي'} "
            f"(ثقة: {hybrid_result['confidence']:.1%})"
        )

        if "rule_based_flags" in hybrid_result:
            print(f"   القواعد: {', '.join(hybrid_result['rule_based_flags'])}")

        print(f"   المتوقع: {test_case['expected']}")
        print()

    accuracy = (correct_predictions / total_predictions) * 100
    print(
        f"📊 دقة الكشف المحسن: {accuracy:.1f}% ({correct_predictions}/{total_predictions})"
    )

    return accuracy


if __name__ == "__main__":
    main()
