import os
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>給与予測アプリ</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1>🎉 給与予測アプリが正常にデプロイされました！</h1>
        <p>📊 システム情報:</p>
        <ul>
            <li>✅ Status: 正常稼働中</li>
            <li>🐍 Python: {}</li>
            <li>🌐 Flask: 動作確認済み</li>
        </ul>
        <p><a href="/test">/test エンドポイントで詳細確認</a></p>
    </body>
    </html>
    """.format(os.sys.version)

@app.route("/test")
def test():
    return {
        "status": "success",
        "message": "給与予測アプリが正常に動作しています！",
        "python_version": os.sys.version,
        "platform": "Render.com"
    }

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 