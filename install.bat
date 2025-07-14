
@echo off
echo 🚀 تثبيت خدمة الذكاء الاصطناعي الطبي...
echo.

REM التحقق من وجود Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python غير مثبت. يرجى تثبيت Python 3.8+ أولاً
    pause
    exit /b 1
)

echo ✅ تم العثور على Python

REM إنشاء بيئة افتراضية
echo 📦 إنشاء البيئة الافتراضية...
python -m venv ai_env

REM تفعيل البيئة الافتراضية
echo 🔧 تفعيل البيئة الافتراضية...
call ai_env\Scripts\activate.bat

REM ترقية pip
echo ⬆️ ترقية pip...
python -m pip install --upgrade pip

REM تثبيت المتطلبات
echo 📚 تثبيت المكتبات المطلوبة...
pip install -r requirements.txt

echo.
echo ✅ تم تثبيت جميع المتطلبات بنجاح!
echo 🚀 لتشغيل الخدمة، قم بتشغيل run.bat
echo.
pause
