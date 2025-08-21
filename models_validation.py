#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكربت اختبار شامل وموسع لنماذج NeuroNexusModels
يشمل 30 حالة اختبار متنوعة لتقييم دقيق للأداء
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# إضافة مسار المودلز
sys.path.append(str(Path(__file__).parent / "ml"))

# استيراد المودلز المطلوبة
try:
    from ml.anomaly_detection import VitalAnomalyDetector
    from ml.forecasting import VitalForecaster
    from ml.enhanced_bp_interface import EnhancedBPForecaster
    from demo_usage import quick_health_check
except ImportError as e:
    print(f"❌ خطأ في استيراد المودلز: {e}")
    sys.exit(1)


class ComprehensiveModelTester:
    """فئة اختبار شاملة وموسعة لجميع النماذج"""

    def __init__(self):
        self.models_path = "ml/saved_models"
        self.anomaly_detector = None
        self.enhanced_bp = None
        self.regular_forecasters = {}
        self.test_results = {
            "anomaly_detection": {},
            "regular_forecasting": {},
            "enhanced_bp_forecasting": {},
            "system_integration": {},
            "performance_metrics": {},
        }
        self.scenarios = {
            "stable": {
                "name": "حالة مستقرة",
                "heart_rate_base": 75,
                "spo2_base": 98,
                "temperature_base": 36.8,
                "systolic_bp_base": 120,
                "diastolic_bp_base": 80,
                "noise_factor": 0.05,
            },
            "improving": {
                "name": "حالة في تحسن",
                "heart_rate_base": 90,
                "spo2_base": 95,
                "temperature_base": 37.5,
                "systolic_bp_base": 140,
                "diastolic_bp_base": 90,
                "noise_factor": 0.08,
                "trend": "improving",
            },
            "deteriorating": {
                "name": "حالة في تدهور",
                "heart_rate_base": 70,
                "spo2_base": 97,
                "temperature_base": 36.5,
                "systolic_bp_base": 110,
                "diastolic_bp_base": 70,
                "noise_factor": 0.1,
                "trend": "deteriorating",
            },
            "fluctuating": {
                "name": "حالة متقلبة",
                "heart_rate_base": 80,
                "spo2_base": 96,
                "temperature_base": 37.0,
                "systolic_bp_base": 125,
                "diastolic_bp_base": 82,
                "noise_factor": 0.15,
                "trend": "fluctuating",
            },
        }

    def load_all_models(self):
        """تحميل جميع النماذج"""
        print("🔄 تحميل جميع النماذج...")

        # تحميل نموذج كشف الشذوذ
        try:
            self.anomaly_detector = VitalAnomalyDetector()
            self.anomaly_detector.load_model(
                f"{self.models_path}/anomaly_detector.joblib"
            )
            print("✅ نموذج كشف الشذوذ محمل بنجاح")
        except Exception as e:
            print(f"❌ خطأ في تحميل نموذج كشف الشذوذ: {e}")
            return False

        # تحميل النماذج المحسنة لضغط الدم
        try:
            self.enhanced_bp = EnhancedBPForecaster()
            self.enhanced_bp.load_models()
            print("✅ النماذج المحسنة لضغط الدم محملة بنجاح")
        except Exception as e:
            print(f"❌ خطأ في تحميل النماذج المحسنة لضغط الدم: {e}")
            return False

        # تحميل النماذج العادية
        vital_types = ["heart_rate", "spo2", "temperature"]
        for vital_type in vital_types:
            try:
                forecaster = VitalForecaster(vital_type)
                forecaster.load_model(
                    f"{self.models_path}/forecaster_{vital_type}.joblib"
                )
                self.regular_forecasters[vital_type] = forecaster
                print(f"✅ نموذج التنبؤ لـ {vital_type} محمل بنجاح")
            except Exception as e:
                print(f"❌ خطأ في تحميل نموذج {vital_type}: {e}")
                return False

        return True

    def get_comprehensive_test_cases(self):
        """الحصول على 30 حالة اختبار متنوعة"""

        from cases import test_cases

        return test_cases

    def test_anomaly_detection_comprehensive(self):
        """اختبار شامل لنموذج كشف الشذوذ مع 30 حالة"""
        print("\n🔍 اختبار شامل لنموذج كشف الشذوذ (30 حالة)...")

        test_cases = self.get_comprehensive_test_cases()
        results = []

        # اختبار كل حالة
        for i, case in enumerate(test_cases, 1):
            try:
                result = self.anomaly_detector.hybrid_predict(case["vitals"])
                is_anomaly = result["is_anomaly"]
                correct = is_anomaly == case["expected"]

                # تحليل الدقة في تصنيف الشدة
                severity_match = False
                if "severity" in result:
                    predicted_severity = result["severity"].lower()
                    expected_severity = case["expected_severity"].lower()

                    severity_mapping = {
                        "normal": ["normal"],
                        "low": ["low", "mild", "بسيط"],
                        "medium": ["medium", "moderate", "متوسط"],
                        "high": ["high", "critical", "حرج", "عالي"],
                        "critical": ["critical", "حرج"],
                    }

                    for key, values in severity_mapping.items():
                        if expected_severity in values and predicted_severity in values:
                            severity_match = True
                            break

                case_result = {
                    "id": i,
                    "name": case["name"],
                    "category": case["category"],
                    "expected": case["expected"],
                    "expected_severity": case["expected_severity"],
                    "actual": is_anomaly,
                    "correct": correct,
                    "severity_match": severity_match,
                    "anomaly_score": result["anomaly_score"],
                    "predicted_severity": result.get("severity", "Unknown"),
                    "confidence": result["confidence"],
                }
                results.append(case_result)

                status = "✅" if correct else "❌"
                severity_status = "✅" if severity_match else "⚠️"
                print(f"{status} {i:2d}. {case['name']}")
                print(
                    f"     المتوقع: {'شذوذ' if case['expected'] else 'طبيعي'} ({case['expected_severity']})"
                )
                print(
                    f"     الناتج: {'شذوذ' if is_anomaly else 'طبيعي'} ({result.get('severity', 'Unknown')}) {severity_status}"
                )
                print(f"     الثقة: {result['confidence']:.1%}")

            except Exception as e:
                print(f"❌ خطأ في اختبار {case['name']}: {e}")
                results.append(
                    {
                        "id": i,
                        "name": case["name"],
                        "category": case["category"],
                        "error": str(e),
                    }
                )

        # حساب الإحصائيات التفصيلية
        successful_tests = [r for r in results if "error" not in r]
        if successful_tests:
            # الإحصائيات العامة
            accuracy = sum(1 for r in successful_tests if r["correct"]) / len(
                successful_tests
            )
            avg_confidence = np.mean([r["confidence"] for r in successful_tests])

            # الإحصائيات حسب الفئة
            category_stats = {}
            for category in [
                "normal",
                "normal_special",
                "borderline",
                "moderate",
                "critical",
                "multi_symptom",
                "special",
            ]:
                category_tests = [
                    r for r in successful_tests if r["category"] == category
                ]
                if category_tests:
                    category_accuracy = sum(
                        1 for r in category_tests if r["correct"]
                    ) / len(category_tests)
                    category_stats[category] = {
                        "count": len(category_tests),
                        "accuracy": category_accuracy,
                        "avg_confidence": np.mean(
                            [r["confidence"] for r in category_tests]
                        ),
                    }

            # إحصائيات الشدة
            severity_accuracy = sum(
                1 for r in successful_tests if r["severity_match"]
            ) / len(successful_tests)

            self.test_results["anomaly_detection"] = {
                "total_tests": len(test_cases),
                "successful_tests": len(successful_tests),
                "accuracy": accuracy,
                "avg_confidence": avg_confidence,
                "severity_accuracy": severity_accuracy,
                "category_stats": category_stats,
                "detailed_results": results,
            }

            print(f"\n📊 نتائج اختبار كشف الشذوذ الشامل:")
            print(f"  إجمالي الاختبارات: {len(test_cases)}")
            print(f"  الاختبارات الناجحة: {len(successful_tests)}")
            print(f"  الدقة الإجمالية: {accuracy:.1%}")
            print(f"  دقة تصنيف الشدة: {severity_accuracy:.1%}")
            print(f"  متوسط الثقة: {avg_confidence:.1%}")

            print(f"\n📈 الدقة حسب الفئة:")
            for category, stats in category_stats.items():
                print(
                    f"  {category}: {stats['accuracy']:.1%} ({stats['count']} اختبار)"
                )
        else:
            print("❌ جميع اختبارات كشف الشذوذ فشلت")

    def test_forecasting_with_multiple_scenarios(self):
        """اختبار التنبؤ مع سيناريوهات متعددة"""
        print("\n📈 اختبار التنبؤ مع سيناريوهات متعددة...")

        results = {}

        for scenario_id, scenario in self.scenarios.items():
            print(f"\n🔍 اختبار سيناريو: {scenario['name']}")

            # إنشاء بيانات تاريخية للسيناريو
            historical_data = self.create_scenario_historical_data(scenario, hours=72)

            # اختبار نماذج التنبؤ العادية
            regular_results = {}
            for vital_type in ["heart_rate", "spo2", "temperature"]:
                try:
                    forecaster = self.regular_forecasters[vital_type]
                    result = forecaster.predict(historical_data, forecast_horizon=24)

                    if (
                        isinstance(result["predictions"], dict)
                        and "values" in result["predictions"]
                    ):
                        predictions = result["predictions"]["values"]
                    else:
                        predictions = result["predictions"]

                    regular_results[vital_type] = {
                        "success": True,
                        "predictions_count": len(predictions),
                        "mean_prediction": np.mean(predictions),
                        "std_prediction": np.std(predictions),
                        "trend": result.get("trend_direction", "غير محدد"),
                        "sample_predictions": predictions[:5],
                    }

                    print(
                        f"  ✅ {vital_type}: متوسط = {np.mean(predictions):.2f}, اتجاه = {result.get('trend_direction', 'غير محدد')}"
                    )

                except Exception as e:
                    regular_results[vital_type] = {"success": False, "error": str(e)}
                    print(f"  ❌ {vital_type}: {e}")

            # اختبار النماذج المحسنة لضغط الدم
            enhanced_results = {}
            try:
                bp_results = self.enhanced_bp.predict_with_auto_features(
                    historical_data=historical_data,
                    pressure_type="both",
                    forecast_hours=24,
                )

                if "error" not in bp_results:
                    for pressure_type in ["systolic", "diastolic"]:
                        if pressure_type in bp_results:
                            result = bp_results[pressure_type]
                            enhanced_results[pressure_type] = {
                                "success": True,
                                "mean_prediction": result["mean"],
                                "trend": result["trend"],
                                "features_used": result["model_info"]["features_used"],
                                "sample_predictions": result["predictions"][:5],
                            }
                            print(
                                f"  ✅ {pressure_type.title()} BP: متوسط = {result['mean']:.1f}, اتجاه = {result['trend']}"
                            )
                else:
                    enhanced_results = {"error": bp_results["error"]}
                    print(f"  ❌ النماذج المحسنة: {bp_results['error']}")

            except Exception as e:
                enhanced_results = {"error": str(e)}
                print(f"  ❌ النماذج المحسنة: {e}")

            results[scenario_id] = {
                "scenario_name": scenario["name"],
                "regular_forecasting": regular_results,
                "enhanced_bp_forecasting": enhanced_results,
            }

        self.test_results["forecasting_scenarios"] = results

    def create_scenario_historical_data(self, scenario, hours=72):
        """إنشاء بيانات تاريخية لسيناريو معين"""
        data = []

        for i in range(hours):
            timestamp = datetime.now() - timedelta(hours=hours - i)
            hour_of_day = timestamp.hour

            # عوامل أساسية
            circadian = np.sin(2 * np.pi * hour_of_day / 24)
            noise = np.random.normal(0, scenario["noise_factor"])

            # تطبيق الاتجاه
            trend_factor = 0
            if scenario.get("trend") == "improving":
                trend_factor = (hours - i) / hours * 0.1  # تحسن تدريجي
            elif scenario.get("trend") == "deteriorating":
                trend_factor = -i / hours * 0.1  # تدهور تدريجي
            elif scenario.get("trend") == "fluctuating":
                trend_factor = np.sin(2 * np.pi * i / 12) * 0.05  # تقلبات

            data.append(
                {
                    "timestamp": timestamp,
                    "heart_rate": scenario["heart_rate_base"]
                    + circadian * 5
                    + noise * 10
                    + trend_factor * 20,
                    "systolic_bp": scenario["systolic_bp_base"]
                    + circadian * 8
                    + noise * 15
                    + trend_factor * 20,
                    "diastolic_bp": scenario["diastolic_bp_base"]
                    + circadian * 4
                    + noise * 8
                    + trend_factor * 10,
                    "spo2": scenario["spo2_base"] + noise * 2 + trend_factor * 3,
                    "temperature": scenario["temperature_base"]
                    + circadian * 0.2
                    + noise * 0.3
                    + trend_factor * 0.5,
                }
            )

        return data

    def test_performance_metrics(self):
        """اختبار مقاييس الأداء"""
        print("\n⚡ اختبار مقاييس الأداء...")

        performance_results = {
            "loading_time": {},
            "prediction_time": {},
            "memory_usage": {},
            "accuracy_metrics": {},
        }

        # اختبار وقت التحميل
        import time

        start_time = time.time()
        self.load_all_models()
        loading_time = time.time() - start_time
        performance_results["loading_time"]["total"] = loading_time
        print(f"✅ وقت تحميل جميع النماذج: {loading_time:.3f} ثانية")

        # اختبار وقت التنبؤ
        test_vitals = {
            "heart_rate": 75,
            "spo2": 98,
            "temperature": 36.8,
            "systolic_bp": 120,
            "diastolic_bp": 80,
        }

        # وقت كشف الشذوذ (باستخدام النظام الهجين المحسن)
        start_time = time.time()
        for _ in range(100):
            self.anomaly_detector.hybrid_predict(test_vitals)
        anomaly_time = (time.time() - start_time) / 100
        performance_results["prediction_time"]["anomaly_detection"] = anomaly_time
        print(f"✅ متوسط وقت كشف الشذوذ (النظام الهجين): {anomaly_time:.3f} ثانية")

        # وقت التنبؤ
        historical_data = self.create_scenario_historical_data(
            self.scenarios["stable"], hours=24
        )

        start_time = time.time()
        for vital_type in ["heart_rate", "spo2", "temperature"]:
            self.regular_forecasters[vital_type].predict(
                historical_data, forecast_horizon=12
            )
        regular_forecasting_time = time.time() - start_time
        performance_results["prediction_time"][
            "regular_forecasting"
        ] = regular_forecasting_time
        print(f"✅ وقت التنبؤ العادي (3 نماذج): {regular_forecasting_time:.3f} ثانية")

        # وقت التنبؤ المحسن
        start_time = time.time()
        self.enhanced_bp.predict_with_auto_features(
            historical_data=historical_data, pressure_type="both", forecast_hours=12
        )
        enhanced_forecasting_time = time.time() - start_time
        performance_results["prediction_time"][
            "enhanced_bp_forecasting"
        ] = enhanced_forecasting_time
        print(f"✅ وقت التنبؤ المحسن لضغط الدم: {enhanced_forecasting_time:.3f} ثانية")

        # اختبار استخدام الذاكرة (تقريبي)
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # MB
        performance_results["memory_usage"]["current"] = memory_usage
        print(f"✅ استخدام الذاكرة الحالي: {memory_usage:.1f} MB")

        self.test_results["performance_metrics"] = performance_results

    def test_system_integration_comprehensive(self):
        """اختبار تكامل النظام بشكل شامل"""
        print("\n🔄 اختبار تكامل النظام الشامل...")

        integration_results = {
            "quick_check": {},
            "comprehensive_analysis": {},
            "error_handling": {},
            "edge_cases": {},
        }

        # اختبار الفحص السريع مع حالات متعددة
        quick_check_cases = [
            {
                "name": "حالة طبيعية",
                "vitals": {
                    "heart_rate": 72,
                    "spo2": 98,
                    "temperature": 36.7,
                    "systolic_bp": 120,
                    "diastolic_bp": 80,
                },
            },
            {
                "name": "حالة شاذة",
                "vitals": {
                    "heart_rate": 120,
                    "spo2": 92,
                    "temperature": 38.0,
                    "systolic_bp": 150,
                    "diastolic_bp": 95,
                },
            },
            {
                "name": "حالة حدية",
                "vitals": {
                    "heart_rate": 95,
                    "spo2": 94,
                    "temperature": 37.3,
                    "systolic_bp": 138,
                    "diastolic_bp": 88,
                },
            },
        ]

        quick_check_results = []
        for case in quick_check_cases:
            try:
                result = quick_health_check(case["vitals"])
                quick_check_results.append(
                    {"name": case["name"], "result": result, "success": True}
                )
                print(f"✅ فحص سريع - {case['name']}: {result}")
            except Exception as e:
                quick_check_results.append(
                    {"name": case["name"], "error": str(e), "success": False}
                )
                print(f"❌ فحص سريع - {case['name']}: {e}")

        integration_results["quick_check"] = quick_check_results

        # اختبار معالجة الأخطاء
        error_cases = [
            {"name": "بيانات ناقصة", "vitals": {"heart_rate": 72, "spo2": 98}},
            {
                "name": "قيم غير صالحة",
                "vitals": {
                    "heart_rate": -10,
                    "spo2": 150,
                    "temperature": 50,
                    "systolic_bp": 300,
                    "diastolic_bp": 200,
                },
            },
            {
                "name": "نوع بيانات خاطئ",
                "vitals": {
                    "heart_rate": "72",
                    "spo2": "98",
                    "temperature": "36.7",
                    "systolic_bp": "120",
                    "diastolic_bp": "80",
                },
            },
        ]

        error_handling_results = []
        for case in error_cases:
            try:
                result = quick_health_check(case["vitals"])
                error_handling_results.append(
                    {"name": case["name"], "handled": True, "result": result}
                )
                print(f"✅ معالجة خطأ - {case['name']}: تم التعامل مع الخطأ")
            except Exception as e:
                error_handling_results.append(
                    {"name": case["name"], "handled": False, "error": str(e)}
                )
                print(f"❌ معالجة خطأ - {case['name']}: {e}")

        integration_results["error_handling"] = error_handling_results

        self.test_results["system_integration"] = integration_results

    def generate_comprehensive_report(self):
        """إنشاء تقرير شامل ومفصل"""
        print("\n" + "=" * 80)
        print("📊 تقرير اختبار شامل وموسع لنماذج NeuroNexusModels")
        print("=" * 80)

        # نتائج كشف الشذوذ
        if self.test_results.get("anomaly_detection"):
            ad = self.test_results["anomaly_detection"]
            print(f"\n🔍 نتائج كشف الشذوذ الشامل:")
            print(f"  إجمالي الاختبارات: {ad['total_tests']}")
            print(f"  الاختبارات الناجحة: {ad['successful_tests']}")
            print(f"  الدقة الإجمالية: {ad['accuracy']:.1%}")
            print(f"  دقة تصنيف الشدة: {ad['severity_accuracy']:.1%}")
            print(f"  متوسط الثقة: {ad['avg_confidence']:.1%}")

            print(f"\n📈 الدقة حسب الفئة:")
            for category, stats in ad["category_stats"].items():
                print(
                    f"  {category:15s}: {stats['accuracy']:.1%} ({stats['count']} اختبار)"
                )

            # عرض الحالات الفاشلة
            failed_cases = [
                r for r in ad["detailed_results"] if not r.get("correct", True)
            ]
            if failed_cases:
                print(f"\n❌ الحالات الفاشلة ({len(failed_cases)}):")
                for case in failed_cases:
                    print(
                        f"  - {case['name']}: المتوقع {'شذوذ' if case['expected'] else 'طبيعي'} -> الناتج {'شذوذ' if case['actual'] else 'طبيعي'}"
                    )

        # نتائج سيناريوهات التنبؤ
        if self.test_results.get("forecasting_scenarios"):
            fs = self.test_results["forecasting_scenarios"]
            print(f"\n📈 نتائج سيناريوهات التنبؤ:")

            for scenario_id, scenario_data in fs.items():
                print(f"\n  📊 {scenario_data['scenario_name']}:")

                # التنبؤ العادي
                regular = scenario_data["regular_forecasting"]
                regular_success = sum(1 for r in regular.values() if r.get("success"))
                print(f"    التنبؤ العادي: {regular_success}/3 نماذج ناجحة")

                for vital_type, result in regular.items():
                    if result.get("success"):
                        print(
                            f"      ✅ {vital_type}: متوسط = {result['mean_prediction']:.2f}, اتجاه = {result['trend']}"
                        )

                # التنبؤ المحسن
                enhanced = scenario_data["enhanced_bp_forecasting"]
                if "error" not in enhanced:
                    print(f"    التنبؤ المحسن: 2/2 نماذج ناجحة")
                    for pressure_type, result in enhanced.items():
                        print(
                            f"      ✅ {pressure_type.title()} BP: متوسط = {result['mean_prediction']:.1f}, اتجاه = {result['trend']}"
                        )
                else:
                    print(f"    ❌ التنبؤ المحسن: {enhanced['error']}")

        # نتائج الأداء
        if self.test_results.get("performance_metrics"):
            pm = self.test_results["performance_metrics"]
            print(f"\n⚡ مقاييس الأداء:")
            print(f"  وقت تحميل النماذج: {pm['loading_time']['total']:.3f} ثانية")
            print(
                f"  وقت كشف الشذوذ: {pm['prediction_time']['anomaly_detection']:.3f} ثانية"
            )
            print(
                f"  وقت التنبؤ العادي: {pm['prediction_time']['regular_forecasting']:.3f} ثانية"
            )
            print(
                f"  وقت التنبؤ المحسن: {pm['prediction_time']['enhanced_bp_forecasting']:.3f} ثانية"
            )
            print(f"  استخدام الذاكرة: {pm['memory_usage']['current']:.1f} MB")

        # نتائج التكامل
        if self.test_results.get("system_integration"):
            si = self.test_results["system_integration"]
            print(f"\n🔄 نتائج التكامل:")

            # الفحص السريع
            qc = si["quick_check"]
            qc_success = sum(1 for r in qc if r.get("success"))
            print(f"  الفحص السريع: {qc_success}/{len(qc)} ناجح")

            # معالجة الأخطاء
            eh = si["error_handling"]
            eh_success = sum(1 for r in eh if r.get("handled"))
            print(f"  معالجة الأخطاء: {eh_success}/{len(eh)} ناجح")

        # التقييم العام
        print(f"\n🎯 التقييم العام:")

        # حساب النتيجة الإجمالية
        total_score = 0
        max_score = 0

        # كشف الشذوذ (40% من النتيجة)
        if self.test_results.get("anomaly_detection"):
            ad_score = self.test_results["anomaly_detection"]["accuracy"] * 40
            total_score += ad_score
            max_score += 40
            print(f"  كشف الشذوذ: {ad_score/40:.1%} ({ad_score/40*100:.1f}/40 نقطة)")

        # التنبؤ (30% من النتيجة)
        if self.test_results.get("forecasting_scenarios"):
            forecasting_success = 0
            total_forecasts = 0

            for scenario_id, scenario_data in self.test_results[
                "forecasting_scenarios"
            ].items():
                regular = scenario_data["regular_forecasting"]
                enhanced = scenario_data["enhanced_bp_forecasting"]

                # التنبؤ العادي
                regular_success = sum(1 for r in regular.values() if r.get("success"))
                forecasting_success += regular_success
                total_forecasts += 3

                # التنبؤ المحسن
                if "error" not in enhanced:
                    forecasting_success += 2
                    total_forecasts += 2

            if total_forecasts > 0:
                forecasting_score = (forecasting_success / total_forecasts) * 30
                total_score += forecasting_score
                max_score += 30
                print(
                    f"  التنبؤ: {forecasting_success}/{total_forecasts} ({forecasting_score/30:.1%}) ({forecasting_score/30*100:.1f}/30 نقطة)"
                )

        # الأداء (15% من النتيجة)
        if self.test_results.get("performance_metrics"):
            pm = self.test_results["performance_metrics"]
            # تقييم الأداء بناءً على الوقت والذاكرة
            performance_score = 15
            if pm["prediction_time"]["anomaly_detection"] > 0.1:
                performance_score -= 5
            if pm["prediction_time"]["regular_forecasting"] > 1.0:
                performance_score -= 5
            if pm["memory_usage"]["current"] > 500:
                performance_score -= 5

            performance_score = max(0, performance_score)
            total_score += performance_score
            max_score += 15
            print(
                f"  الأداء: {performance_score/15:.1%} ({performance_score:.1f}/15 نقطة)"
            )

        # التكامل (15% من النتيجة)
        if self.test_results.get("system_integration"):
            si = self.test_results["system_integration"]
            qc_success = sum(1 for r in si["quick_check"] if r.get("success"))
            eh_success = sum(1 for r in si["error_handling"] if r.get("handled"))

            integration_score = (
                (qc_success + eh_success)
                / (len(si["quick_check"]) + len(si["error_handling"]))
            ) * 15
            total_score += integration_score
            max_score += 15
            print(
                f"  التكامل: {integration_score/15:.1%} ({integration_score:.1f}/15 نقطة)"
            )

        # النتيجة النهائية
        if max_score > 0:
            final_percentage = (total_score / max_score) * 100
            print(
                f"\n📊 النتيجة الإجمالية: {final_percentage:.1f}% ({total_score:.1f}/{max_score} نقطة)"
            )

            if final_percentage >= 90:
                print("🎉 النظام ممتاز وجاهز للإنتاج!")
            elif final_percentage >= 75:
                print("⚠️ النظام جيد لكن يحتاج بعض التحسينات")
            elif final_percentage >= 60:
                print("❌ النظام مقبول لكن يحتاج تحسينات كبيرة")
            else:
                print("🚨 النظام غير جاهز ويحتاج إعادة تطوير")

        # حفظ التقرير المفصل
        self.save_detailed_report()

    def save_detailed_report(self):
        """حفظ تقرير مفصل في ملف JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_report_{timestamp}.json"

        # إضافة معلومات إضافية للتقرير
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "test_results": self.test_results,
            "system_info": {
                "python_version": sys.version,
                "models_loaded": {
                    "anomaly_detector": self.anomaly_detector is not None,
                    "enhanced_bp": self.enhanced_bp is not None,
                    "regular_forecasters": list(self.regular_forecasters.keys()),
                },
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n💾 تم حفظ التقرير المفصل في: {filename}")


def main():
    """الدالة الرئيسية لتشغيل الاختبارات الشاملة"""
    print("🚀 بدء اختبار شامل وموسع لنماذج NeuroNexusModels")
    print("=" * 80)

    # إنشاء كائن الاختبار
    tester = ComprehensiveModelTester()

    # تحميل النماذج
    if not tester.load_all_models():
        print("❌ فشل تحميل النماذج. إنهاء الاختبارات.")
        return

    # تشغيل الاختبارات الشاملة
    tester.test_anomaly_detection_comprehensive()
    tester.test_forecasting_with_multiple_scenarios()
    tester.test_performance_metrics()
    tester.test_system_integration_comprehensive()

    # إنشاء التقرير
    tester.generate_comprehensive_report()


if __name__ == "__main__":
    main()
