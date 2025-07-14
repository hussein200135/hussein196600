
@echo off
echo 🚀 إعداد تطبيق المختبر الذكي الكامل...
echo.

echo 📦 تثبيت حزم الواجهة الأمامية...
call npm install

echo.
echo 🐍 إعداد البيئة الخلفية للذكاء الاصطناعي...
cd ai-backend
call enhanced_install.bat
cd..

echo.
echo ✅ تم إعداد التطبيق بنجاح!
echo.
echo 🎯 للبدء:
echo    1. تشغيل الخدمة الخلفية: cd ai-backend && enhanced_run.bat
echo    2. تشغيل التطبيق: npm run electron:dev
echo.
echo 📦 لإنشاء حزمة التوزيع: npm run dist
echo.
pause
