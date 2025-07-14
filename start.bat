
@echo off
echo 🚀 تشغيل تطبيق المختبر الذكي...
echo.

echo 🐍 بدء تشغيل خدمة الذكاء الاصطناعي...
start "AI Backend" cmd /k "cd ai-backend && enhanced_run.bat"

echo ⏳ انتظار تحميل الخدمة الخلفية...
timeout /t 10

echo 🖥️ تشغيل تطبيق سطح المكتب...
npm run electron:dev

echo.
echo ✅ تم تشغيل جميع الخدمات!
