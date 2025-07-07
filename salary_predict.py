import pickle
import numpy as np
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 機械学習モデルの読込
try:
    with open("./data/model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ モデルファイルの読み込み成功")
    print(f"🔍 モデルタイプ: {type(model)}")
    
    # モデルの状態確認
    if hasattr(model, 'booster_'):
        print("✅ LightGBMモデルが正しく訓練されています")
    else:
        print("⚠️ モデルが訓練されていない可能性があります")
        
except Exception as e:
    print(f"❌ モデル読み込みエラー: {e}")
    model = None

# データフレームの読込
try:
    with open("./data/df.pkl", "rb") as f:
        df = pickle.load(f)
    print("✅ データフレームの読み込み成功")
    print(f"🔍 データフレーム形状: {df.shape if df is not None else 'None'}")
except Exception as e:
    print(f"❌ データフレーム読み込みエラー: {e}")
    df = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    try:
        # 入力値取得
        position = int(request.form["position"])
        age = int(request.form["age"])
        area = int(request.form["area"])
        sex = int(request.form["sex"])
        partner = int(request.form["partner"])
        num_child = int(request.form["num_child"])
        education = int(request.form["education"])
        service_length = int(request.form["service_length"])
        study_time = int(request.form["study_time"])
        commute = int(request.form["commute"])
        overtime = int(request.form["overtime"])

        # 機械学習モデルに入力するパラメータ
        param = (
            position,
            age,
            area,
            sex,
            partner,
            num_child,
            education,
            service_length,
            study_time,
            commute,
            overtime,
        )

        print(f"🔍 予測パラメータ: {param}")
        
        # モデルが正常に読み込まれているかチェック
        if model is None:
            return render_template("result.html", param=param, salary="エラー: モデルが読み込まれていません", result="error")
        
        # 予測実行
        salary = model.predict([param])[0].round()
        print(f"✅ 予測完了: {salary}")
        
        return render_template("result.html", param=param, salary=salary, result="success")
        
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        error_message = f"予測エラー: {str(e)}"
        return render_template("result.html", param=param if 'param' in locals() else [], salary=error_message, result="error")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
