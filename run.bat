
@echo off
echo 🤖 تشغيل خدمة الذكاء الاصطناعي الطبي...
echo.

REM تفعيل البيئة الافتراضية
call ai_env\Scripts\activate.bat

REM تشغيل الخدمة
echo 🚀 بدء تشغيل الخادم على المنفذ 5000...
python main.py

pause
