#!/usr/bin/env python3
"""
Step 12: Model Monitoring & Retraining
ติดตามประสิทธิภาพโมเดลและ Retrain อัตโนมัติ
- Performance monitoring (accuracy, loss, latency)
- Data drift detection
- Auto-retraining triggers
- Logging and alerting
- A/B testing support
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, model_name, monitoring_window=7):
        self.model_name = model_name
        self.monitoring_window = monitoring_window  # days
        self.metrics_history = []
        self.alert_thresholds = {
            'accuracy_drop': 0.05,  # 5% drop triggers alert
            'latency_increase': 0.20,  # 20% increase
            'error_rate_increase': 0.10  # 10% increase
        }
    
    def log_prediction(self, question, answer, prediction, metrics):
        """Log a single prediction with metrics"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'prediction': prediction,
            'metrics': metrics
        }
        
        self.metrics_history.append(log_entry)
        
        # Save to file
        self.save_log_entry(log_entry)
        
        # Check for alerts
        self.check_alerts()
    
    def calculate_metrics(self, predictions, ground_truths):
        """Calculate performance metrics"""
        from sklearn.metrics import accuracy_score
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(predictions),
            'accuracy': 0.0,
            'avg_latency_ms': 0.0,
            'error_rate': 0.0
        }
        
        # Simple accuracy (exact match)
        exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p.strip().lower() == g.strip().lower())
        metrics['accuracy'] = exact_matches / len(predictions) if predictions else 0.0
        
        return metrics
    
    def check_alerts(self):
        """Check if any alert thresholds are exceeded"""
        if len(self.metrics_history) < 2:
            return
        
        # Get recent metrics
        recent = self.metrics_history[-100:]  # Last 100 predictions
        
        # Calculate current metrics
        current_accuracy = np.mean([m['metrics'].get('accuracy', 0) for m in recent if 'metrics' in m])
        
        # Compare with baseline (first 100)
        if len(self.metrics_history) > 100:
            baseline = self.metrics_history[:100]
            baseline_accuracy = np.mean([m['metrics'].get('accuracy', 0) for m in baseline if 'metrics' in m])
            
            accuracy_drop = baseline_accuracy - current_accuracy
            
            if accuracy_drop > self.alert_thresholds['accuracy_drop']:
                self.send_alert(
                    'ACCURACY_DROP',
                    f"Accuracy dropped by {accuracy_drop:.2%} (from {baseline_accuracy:.2%} to {current_accuracy:.2%})"
                )
    
    def send_alert(self, alert_type, message):
        """Send alert (log for now, can extend to email/slack)"""
        logger.warning(f"🚨 ALERT [{alert_type}]: {message}")
        
        # Save to alerts file
        alert_file = Path("logs/alerts.json")
        alert_file.parent.mkdir(parents=True, exist_ok=True)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'model': self.model_name
        }
        
        alerts = []
        if alert_file.exists():
            with open(alert_file, 'r', encoding='utf-8') as f:
                try:
                    alerts = json.load(f)
                except:
                    alerts = []
        
        alerts.append(alert)
        
        with open(alert_file, 'w', encoding='utf-8') as f:
            json.dump(alerts, f, indent=2)
    
    def save_log_entry(self, log_entry):
        """Save log entry to file"""
        log_dir = Path("logs/predictions")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"predictions_{date_str}.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def detect_data_drift(self, recent_data, reference_data):
        """Detect data drift using statistical tests"""
        drift_detected = False
        drift_details = {}
        
        # Feature distributions
        for feature in ['question_length', 'answer_length']:
            recent_values = [len(str(d.get('question', '')).split()) for d in recent_data]
            reference_values = [len(str(d.get('question', '')).split()) for d in reference_data]
            
            # Simple statistical test (KS test)
            from scipy import stats
            try:
                statistic, pvalue = stats.ks_2samp(recent_values, reference_values)
                
                if pvalue < 0.05:  # Significant difference
                    drift_detected = True
                    drift_details[feature] = {
                        'statistic': statistic,
                        'pvalue': pvalue,
                        'recent_mean': np.mean(recent_values),
                        'reference_mean': np.mean(reference_values)
                    }
            except Exception as e:
                logger.warning(f"Drift detection error for {feature}: {e}")
        
        if drift_detected:
            self.send_alert('DATA_DRIFT', f"Data drift detected: {list(drift_details.keys())}")
        
        return drift_detected, drift_details
    
    def should_retrain(self):
        """Determine if model should be retrained"""
        reasons = []
        
        # Check 1: Performance degradation
        if len(self.metrics_history) > 200:
            recent_accuracy = np.mean([m['metrics'].get('accuracy', 0) for m in self.metrics_history[-100:] if 'metrics' in m])
            baseline_accuracy = np.mean([m['metrics'].get('accuracy', 0) for m in self.metrics_history[:100] if 'metrics' in m])
            
            if baseline_accuracy - recent_accuracy > self.alert_thresholds['accuracy_drop']:
                reasons.append(f"Accuracy dropped by {(baseline_accuracy - recent_accuracy):.2%}")
        
        # Check 2: Time since last training
        last_training_file = Path("models/trained/last_training_timestamp.txt")
        if last_training_file.exists():
            with open(last_training_file, 'r') as f:
                last_training = datetime.fromisoformat(f.read().strip())
            
            days_since_training = (datetime.now() - last_training).days
            
            if days_since_training > 30:  # Retrain every 30 days
                reasons.append(f"Last training was {days_since_training} days ago")
        
        # Check 3: Sufficient new data
        new_data_count = len(self.metrics_history)
        if new_data_count > 10000:  # Enough new samples
            reasons.append(f"Accumulated {new_data_count:,} new samples")
        
        return len(reasons) > 0, reasons
    
    def trigger_retraining(self, reasons):
        """Trigger automatic retraining"""
        logger.info("🔄 Triggering automatic retraining...")
        logger.info(f"   Reasons: {reasons}")
        
        # Save retraining trigger
        retrain_trigger_file = Path("logs/retrain_triggers.json")
        retrain_trigger_file.parent.mkdir(parents=True, exist_ok=True)
        
        trigger = {
            'timestamp': datetime.now().isoformat(),
            'reasons': reasons,
            'model': self.model_name
        }
        
        triggers = []
        if retrain_trigger_file.exists():
            with open(retrain_trigger_file, 'r', encoding='utf-8') as f:
                try:
                    triggers = json.load(f)
                except:
                    triggers = []
        
        triggers.append(trigger)
        
        with open(retrain_trigger_file, 'w', encoding='utf-8') as f:
            json.dump(triggers, f, indent=2)
        
        # In production, this would trigger a retraining pipeline
        # For now, just log the event
        logger.info("✅ Retraining trigger saved")
        logger.info("💡 Manual action: Run Main_8_Model_Training.py to retrain")
        
        return True
    
    def generate_monitoring_report(self, output_dir="data/exports/evaluation"):
        """Generate comprehensive monitoring report"""
        logger.info("📊 Generating monitoring report...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"monitoring_report_{timestamp}.json"
        
        # Calculate summary statistics
        if len(self.metrics_history) > 0:
            recent_100 = self.metrics_history[-100:]
            
            report = {
                'timestamp': timestamp,
                'model': self.model_name,
                'total_predictions': len(self.metrics_history),
                'recent_predictions': len(recent_100),
                'summary': {
                    'avg_accuracy': np.mean([m['metrics'].get('accuracy', 0) for m in recent_100 if 'metrics' in m]),
                    'accuracy_trend': 'stable',  # TODO: calculate trend
                    'total_alerts': self.count_alerts()
                },
                'recommendations': self.get_recommendations()
            }
        else:
            report = {
                'timestamp': timestamp,
                'model': self.model_name,
                'total_predictions': 0,
                'message': 'No predictions logged yet'
            }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Saved: {report_file}")
        return report_file
    
    def count_alerts(self):
        """Count total alerts"""
        alert_file = Path("logs/alerts.json")
        if alert_file.exists():
            with open(alert_file, 'r', encoding='utf-8') as f:
                try:
                    alerts = json.load(f)
                    return len(alerts)
                except:
                    return 0
        return 0
    
    def get_recommendations(self):
        """Get recommendations based on monitoring data"""
        recommendations = []
        
        should_retrain, reasons = self.should_retrain()
        
        if should_retrain:
            recommendations.append({
                'action': 'retrain_model',
                'priority': 'high',
                'reasons': reasons
            })
        
        if len(self.metrics_history) < 1000:
            recommendations.append({
                'action': 'collect_more_data',
                'priority': 'medium',
                'message': 'Need more predictions for robust monitoring'
            })
        
        return recommendations

def main():
    """Main monitoring process"""
    print("📊 STEP 12: MODEL MONITORING & RETRAINING")
    print("=" * 60)
    print("📌 Track: Performance, Drift, Errors")
    print("📌 Auto: Retraining triggers, Alerts")
    print()
    
    import argparse
    parser = argparse.ArgumentParser(description="Step 12: Model Monitoring")
    parser.add_argument("--model", type=str, default="medical_qa_model",
                       help="Model name to monitor")
    parser.add_argument("--check-retrain", action="store_true",
                       help="Check if retraining is needed")
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ModelMonitor(model_name=args.model)
    
    # Load existing logs
    log_dir = Path("logs/predictions")
    if log_dir.exists():
        log_files = list(log_dir.glob("predictions_*.jsonl"))
        total_predictions = 0
        
        for log_file in log_files:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        monitor.metrics_history.append(entry)
                        total_predictions += 1
                    except:
                        pass
        
        logger.info(f"📊 Loaded {total_predictions:,} predictions from logs")
    else:
        logger.info("📊 No prediction logs found yet")
    
    # Check if retraining is needed
    if args.check_retrain:
        should_retrain, reasons = monitor.should_retrain()
        
        if should_retrain:
            print("\n🔄 RETRAINING RECOMMENDED")
            print("Reasons:")
            for reason in reasons:
                print(f"   - {reason}")
            
            # Ask user
            response = input("\nTrigger retraining? (y/n): ")
            if response.lower() == 'y':
                monitor.trigger_retraining(reasons)
        else:
            print("\n✅ Model performance is stable")
            print("💡 No retraining needed at this time")
    
    # Generate report
    report_file = monitor.generate_monitoring_report()
    
    print(f"\n✅ Monitoring check completed!")
    print(f"📁 Report: {report_file}")
    print(f"\n💡 Usage:")
    print(f"   - Run periodically (daily/weekly)")
    print(f"   - Check logs/alerts.json for issues")
    print(f"   - Use --check-retrain to evaluate retraining need")
    
    return 0

if __name__ == "__main__":
    exit(main())
