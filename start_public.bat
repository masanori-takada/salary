@echo off
echo 🚀 給与予測アプリを外部公開します...
echo.
echo 📋 手順:
echo 1. ngrok.com/download からWindows版ngrokをダウンロード
echo 2. ngrok.exeをこのフォルダに配置
echo 3. このバッチファイルを再実行
echo.

if exist ngrok.exe (
    echo ✅ ngrok.exeが見つかりました
    echo 🌍 外部公開を開始します...
    echo.
    echo 📱 アプリURL: http://127.0.0.1:5001
    echo 🌐 外部公開URL: 下記のngrokで表示されるhttps://xxxxxxxx.ngrok.ioを使用
    echo.
    echo ⚠️ このウィンドウを閉じると公開が停止します
    echo.
    ngrok.exe http 5001
) else (
    echo ❌ ngrok.exeが見つかりません
    echo.
    echo 📥 ngrok.com/download からダウンロードして、このフォルダに配置してください
    echo.
    pause
) 