
@echo off
echo 🤖 تشغيل نظام الذكاء الاصطناعي الطبي المتقدم...
echo.

REM تفعيل البيئة الافتراضية
call ai_medical_env\Scripts\activate.bat

REM تشغيل الخدمة المحسنة
echo 🚀 بدء تشغيل الخادم المتقدم على المنفذ 5000...
echo 🌐 الخدمة ستكون متاحة على: http://localhost:5000
echo 📊 لوحة التحكم: http://localhost:3000
echo.
echo 🔄 للتوقف، اضغط Ctrl+C
echo.

python enhanced_main.py

pau