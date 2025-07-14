
@echo off
echo 🚀 تثبيت نظام الذكاء الاصطناعي الطبي المتقدم...
echo.

REM التحقق من وجود Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python غير مثبت. يرجى تثبيت Python 3.8+ أولاً
    echo 📥 يمكنك تحميله من: https://python.org
    pause
    exit /b 1
)

echo ✅ تم العثور على Python

REM إنشاء بيئة افتراضية محسنة
echo 📦 إنشاء البيئة الافتراضية المحسنة...
python -m venv ai_medical_env

REM تفعيل البيئة الافتراضية
echo 🔧 تفعيل البيئة الافتراضية...
call ai_medical_env\Scripts\activate.bat

REM ترقية pip وأدوات التثبيت
echo ⬆️ ترقية أدوات التثبيت...
python -m pip install --upgrade pip setuptools wheel

REM تثبيت المتطلبات المحسنة
echo 📚 تثبيت مكتبات الذكاء الاصطناعي المتقدمة...
echo هذا قد يستغرق عدة دقائق...
pip install -r enhanced_requirements.txt

REM تحميل نماذج Transformers
echo 🤖 تحميل نماذج الذكاء الاصطناعي...
python -c "from transformers import AutoTokenizer, AutoModel; print('تحميل النماذج...')"

echo.
echo ✅ تم تثبيت النظام المتقدم بنجاح!
echo 🎯 المميزات المتاحة:
echo    - 10 خوارزميات ذكاء اصطناعي
echo    - تحليل شامل للبيانات الطبية
echo    - كشف الأنماط المتقدمة
echo    - تصنيف المخاطر الذكي
echo    - تحليل الصور الطبية
echo    - معالجة البيانات الزمنية
echo.
echo 🚀 لتشغيل النظام، استخدم: enhanced_run.bat
echo.
pause
