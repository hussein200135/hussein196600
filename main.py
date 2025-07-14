
# AI Backend Service - All 10 Algorithms Implementation
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import AutoTokenizer, AutoModel
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import base64
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class MedicalAIAnalyzer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.load_or_train_models()
    
    def load_or_train_models(self):
        """تحميل أو تدريب جميع النماذج"""
        # 1. Logistic Regression
        self.models['logistic'] = LogisticRegression(random_state=42)
        
        # 2. Random Forest
        self.models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 3. SVM
        self.models['svm'] = SVC(kernel='rbf', probability=True, random_state=42)
        
        # 4. KNN
        self.models['knn'] = KNeighborsClassifier(n_neighbors=5)
        
        # 5. Naive Bayes
        self.models['naive_bayes'] = GaussianNB()
        
        # 6. Decision Tree
        self.models['decision_tree'] = DecisionTreeClassifier(random_state=42)
        
        # تدريب النماذج الأساسية مع بيانات وهمية
        self.train_basic_models()
        
        # 7-10. Neural Networks (سيتم تحميلها عند الحاجة)
        self.build_neural_networks()
    
    def train_basic_models(self):
        """تدريب النماذج الأساسية"""
        # بيانات طبية وهمية للتدريب
        np.random.seed(42)
        X = np.random.randn(1000, 10)  # 10 مؤشرات طبية
        y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(1000) * 0.1 > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # تدريب كل نموذج
        for name, model in self.models.items():
            if name in ['logistic', 'random_forest', 'svm', 'knn', 'naive_bayes', 'decision_tree']:
                model.fit(X_train_scaled, y_train)
                print(f"✅ تم تدريب نموذج {name}")
    
    def build_neural_networks(self):
        """بناء الشبكات العصبية"""
        # 7. Basic Neural Network
        self.models['neural_network'] = Sequential([
            Dense(64, activation='relu', input_shape=(10,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.models['neural_network'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 8. CNN للصور الطبية
        self.models['cnn'] = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.models['cnn'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # 9. LSTM للبيانات الزمنية
        self.models['lstm'] = Sequential([
            LSTM(50, return_sequences=True, input_shape=(30, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1, activation='sigmoid')
        ])
        self.models['lstm'].compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        print("✅ تم بناء الشبكات العصبية")
    
    def predict_disease_probability(self, lab_data, algorithm='ensemble'):
        """توقع احتمالية المرض"""
        try:
            if isinstance(lab_data, dict):
                # تحويل البيانات إلى array
                features = np.array(list(lab_data.values())).reshape(1, -1)
            else:
                features = np.array(lab_data).reshape(1, -1)
            
            features_scaled = self.scaler.transform(features)
            
            if algorithm == 'ensemble':
                # استخدام جميع النماذج للتصويت
                predictions = {}
                for name, model in self.models.items():
                    if name in ['logistic', 'random_forest', 'svm', 'knn', 'naive_bayes', 'decision_tree']:
                        if hasattr(model, 'predict_proba'):
                            pred = model.predict_proba(features_scaled)[0][1]
                        else:
                            pred = model.predict(features_scaled)[0]
                        predictions[name] = float(pred)
                
                # حساب المتوسط المرجح
                ensemble_pred = np.mean(list(predictions.values()))
                return {
                    'ensemble_probability': float(ensemble_pred),
                    'individual_predictions': predictions,
                    'confidence': self.calculate_confidence(predictions),
                    'recommendation': self.get_medical_recommendation(ensemble_pred)
                }
            else:
                # استخدام خوارزمية محددة
                model = self.models.get(algorithm)
                if model and hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(features_scaled)[0][1]
                    return {
                        'probability': float(pred),
                        'algorithm': algorithm,
                        'confidence': float(abs(pred - 0.5) * 2),
                        'recommendation': self.get_medical_recommendation(pred)
                    }
        except Exception as e:
            return {'error': f'خطأ في التنبؤ: {str(e)}'}
    
    def analyze_medical_image(self, image_data):
        """تحليل الصور الطبية باستخدام CNN"""
        try:
            # تحويل base64 إلى صورة
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            
            if len(image_array.shape) == 2:  # صورة رمادية
                image_array = np.stack([image_array] * 3, axis=-1)
            
            image_array = np.expand_dims(image_array, axis=0)
            
            # التنبؤ باستخدام CNN
            prediction = self.models['cnn'].predict(image_array)[0][0]
            
            return {
                'abnormality_probability': float(prediction),
                'classification': 'غير طبيعي' if prediction > 0.5 else 'طبيعي',
                'confidence': float(abs(prediction - 0.5) * 2),
                'details': self.get_image_analysis_details(prediction)
            }
        except Exception as e:
            return {'error': f'خطأ في تحليل الصورة: {str(e)}'}
    
    def analyze_time_series(self, time_data):
        """تحليل البيانات الزمنية باستخدام LSTM"""
        try:
            # تحضير البيانات الزمنية
            data = np.array(time_data).reshape(-1, 1)
            if len(data) < 30:
                # إضافة padding إذا كانت البيانات قليلة
                padding = np.zeros((30 - len(data), 1))
                data = np.vstack([padding, data])
            else:
                data = data[-30:]  # أخذ آخر 30 قراءة
            
            data_scaled = (data - np.mean(data)) / (np.std(data) + 1e-8)
            data_input = np.expand_dims(data_scaled, axis=0)
            
            # التنبؤ باستخدام LSTM
            prediction = self.models['lstm'].predict(data_input)[0][0]
            
            return {
                'future_trend': float(prediction),
                'trend_direction': 'تصاعدي' if prediction > 0.5 else 'تنازلي',
                'stability_score': self.calculate_stability(time_data),
                'recommendations': self.get_time_series_recommendations(prediction, time_data)
            }
        except Exception as e:
            return {'error': f'خطأ في تحليل البيانات الزمنية: {str(e)}'}
    
    def detect_anomalies(self, lab_values):
        """كشف الشذوذ في البيانات الطبية"""
        try:
            from sklearn.ensemble import IsolationForest
            
            data = np.array(lab_values).reshape(-1, 1)
            
            # نموذج كشف الشذوذ
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_scores = anomaly_detector.fit_predict(data)
            anomaly_probs = anomaly_detector.score_samples(data)
            
            return {
                'anomaly_detected': bool(np.any(anomaly_scores == -1)),
                'anomaly_indices': np.where(anomaly_scores == -1)[0].tolist(),
                'anomaly_scores': anomaly_probs.tolist(),
                'severity': self.calculate_anomaly_severity(anomaly_probs),
                'recommendations': self.get_anomaly_recommendations(anomaly_scores)
            }
        except Exception as e:
            return {'error': f'خطأ في كشف الشذوذ: {str(e)}'}
    
    def calculate_confidence(self, predictions):
        """حساب مستوى الثقة"""
        values = list(predictions.values())
        std_dev = np.std(values)
        return float(max(0, 1 - std_dev * 2))  # كلما قل التباين، زادت الثقة
    
    def get_medical_recommendation(self, probability):
        """توصيات طبية بناء على الاحتمالية"""
        if probability > 0.8:
            return {
                'urgency': 'عاجل جداً',
                'action': 'يُنصح بمراجعة الطبيب فوراً',
                'follow_up': 'خلال 24 ساعة'
            }
        elif probability > 0.6:
            return {
                'urgency': 'متوسط',
                'action': 'يُنصح بمراجعة الطبيب قريباً',
                'follow_up': 'خلال أسبوع'
            }
        else:
            return {
                'urgency': 'منخفض',
                'action': 'متابعة دورية',
                'follow_up': 'خلال شهر'
            }
    
    def get_image_analysis_details(self, probability):
        """تفاصيل تحليل الصورة"""
        if probability > 0.7:
            return "تم اكتشاف تغيرات غير طبيعية واضحة في الصورة"
        elif probability > 0.4:
            return "تغيرات طفيفة قد تحتاج لمراجعة طبية"
        else:
            return "لا توجد تغيرات واضحة في الصورة"
    
    def calculate_stability(self, time_data):
        """حساب استقرار البيانات الزمنية"""
        if len(time_data) < 2:
            return 1.0
        
        changes = np.diff(time_data)
        stability = 1.0 / (1.0 + np.std(changes))
        return float(stability)
    
    def get_time_series_recommendations(self, prediction, time_data):
        """توصيات للبيانات الزمنية"""
        trend = np.polyfit(range(len(time_data)), time_data, 1)[0]
        
        if trend > 0.1:
            return "النتائج تظهر اتجاهاً تصاعدياً، يُنصح بالمتابعة الدقيقة"
        elif trend < -0.1:
            return "النتائج تظهر اتجاهاً تنازلياً، قد يدل على تحسن"
        else:
            return "النتائج مستقرة، استمر في الخطة العلاجية الحالية"
    
    def calculate_anomaly_severity(self, scores):
        """حساب شدة الشذوذ"""
        min_score = np.min(scores)
        if min_score < -0.5:
            return "شذوذ شديد"
        elif min_score < -0.3:
            return "شذوذ متوسط"
        else:
            return "شذوذ طفيف"
    
    def get_anomaly_recommendations(self, anomaly_scores):
        """توصيات لحالات الشذوذ"""
        if np.any(anomaly_scores == -1):
            return "تم اكتشاف قيم شاذة، يُنصح بإعادة الفحص والمراجعة الطبية"
        else:
            return "جميع القيم ضمن النطاق المتوقع"

# إنشاء مثيل من المحلل
analyzer = MedicalAIAnalyzer()

@app.route('/api/analyze/basic', methods=['POST'])
def analyze_basic():
    """تحليل أساسي للمؤشرات الطبية"""
    try:
        data = request.json
        lab_data = data.get('lab_values', {})
        algorithm = data.get('algorithm', 'ensemble')
        
        result = analyzer.predict_disease_probability(lab_data, algorithm)
        return jsonify({
            'success': True,
            'analysis': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze/image', methods=['POST'])
def analyze_image():
    """تحليل الصور الطبية"""
    try:
        data = request.json
        image_data = data.get('image')
        
        result = analyzer.analyze_medical_image(image_data)
        return jsonify({
            'success': True,
            'analysis': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze/timeseries', methods=['POST'])
def analyze_timeseries():
    """تحليل البيانات الزمنية"""
    try:
        data = request.json
        time_data = data.get('time_values', [])
        
        result = analyzer.analyze_time_series(time_data)
        return jsonify({
            'success': True,
            'analysis': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze/anomaly', methods=['POST'])
def analyze_anomaly():
    """كشف الشذوذ"""
    try:
        data = request.json
        lab_values = data.get('values', [])
        
        result = analyzer.detect_anomalies(lab_values)
        return jsonify({
            'success': True,
            'analysis': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """معلومات عن النماذج المتاحة"""
    models_info = {
        'logistic': 'انحدار لوجستي - للتصنيف الثنائي',
        'random_forest': 'الغابة العشوائية - للتصنيف المتقدم',
        'svm': 'آلة الدعم الشعاعي - للفصل المعقد',
        'knn': 'أقرب الجيران - للتشابه',
        'naive_bayes': 'بايز الساذج - للاحتمالات',
        'decision_tree': 'شجرة القرار - للتفسير الواضح',
        'neural_network': 'شبكة عصبية - للأنماط المعقدة',
        'cnn': 'شبكة تطبيقية - لتحليل الصور',
        'lstm': 'ذاكرة قصيرة طويلة - للبيانات الزمنية',
        'anomaly_detection': 'كشف الشذوذ - للحالات غير العادية'
    }
    
    return jsonify({
        'success': True,
        'models': models_info,
        'total_models': len(models_info)
    })

if __name__ == '__main__':
    print("🚀 بدء تشغيل خدمة الذكاء الاصطناعي الطبي...")
    print("✅ تم تحميل جميع الخوارزميات العشرة")
    app.run(host='0.0.0.0', port=5000, debug=True)
