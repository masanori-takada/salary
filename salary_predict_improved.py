import pickle
import numpy as np
import pandas as pd
import os
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# 改良版機械学習モデルの読込
try:
    with open("./data/model_improved.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ 改良版モデルファイルの読み込み成功")
    print(f"🔍 モデルタイプ: {type(model)}")
    
    # 特徴量名の読み込み
    with open("./data/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    print(f"✅ 特徴量数: {len(feature_columns)}")
    
except Exception as e:
    print(f"❌ 改良版モデル読み込みエラー: {e}")
    print("⚠️ 基本モデルにフォールバック中...")
    
    # フォールバック: 基本モデルを使用
    try:
        with open("./data/model.pkl", "rb") as f:
            model = pickle.load(f)
        feature_columns = ['position', 'age', 'area', 'sex', 'partner', 'num_child', 
                          'education', 'service_length', 'study_time', 'commute', 'overtime']
        print("✅ 基本モデルでフォールバック成功")
    except Exception as e2:
        print(f"❌ フォールバックも失敗: {e2}")
        model = None

# データフレームの読込
try:
    with open("./data/df_improved.pkl", "rb") as f:
        df = pickle.load(f)
    print("✅ 改良版データフレームの読み込み成功")
    print(f"🔍 データフレーム形状: {df.shape if df is not None else 'None'}")
except Exception as e:
    print(f"❌ 改良版データフレーム読み込みエラー: {e}")
    # フォールバック
    try:
        with open("./data/df.pkl", "rb") as f:
            df = pickle.load(f)
        print("✅ 基本データフレームでフォールバック成功")
    except:
        df = None


def create_features(position, age, area, sex, partner, num_child, education, 
                   service_length, study_time, commute, overtime):
    """基本パラメータから改良版特徴量を生成"""
    
    # 基本特徴量
    features = {
        'position': position,
        'age': age,
        'area': area,
        'sex': sex,
        'partner': partner,
        'num_child': num_child,
        'education': education,
        'service_length': service_length,
        'study_time': study_time,
        'commute': commute,
        'overtime': overtime
    }
    
    # 新しい特徴量（改良版モデルの場合のみ）
    if len(feature_columns) > 11:  # 改良版の場合
        features['age_position_ratio'] = age / (position + 1)
        features['experience_ratio'] = service_length / age if age > 0 else 0
        features['study_overtime_ratio'] = study_time / (overtime + 1)
        features['total_time'] = commute + overtime
        features['family_burden'] = partner + num_child
        
        # 年齢カテゴリ
        if age <= 25:
            features['age_category'] = 1
        elif age <= 35:
            features['age_category'] = 2
        elif age <= 45:
            features['age_category'] = 3
        elif age <= 55:
            features['age_category'] = 4
        else:
            features['age_category'] = 5
    
    # 特徴量を正しい順序で配列に変換
    feature_array = []
    for col in feature_columns:
        feature_array.append(features.get(col, 0))
    
    return np.array(feature_array).reshape(1, -1)


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

        print(f"🔍 入力パラメータ: position={position}, age={age}, area={area}")
        
        # モデルが正常に読み込まれているかチェック
        if model is None:
            return render_template("result.html", 
                                 position=position, age=age, area=area,
                                 salary="エラー: モデルが読み込まれていません", 
                                 result="error", model_info="エラー")
        
        # 改良版特徴量を作成
        X = create_features(position, age, area, sex, partner, num_child, 
                           education, service_length, study_time, commute, overtime)
        
        print(f"🔍 特徴量配列形状: {X.shape}")
        print(f"🔍 使用特徴量数: {len(feature_columns)}")
        
        # 予測実行
        salary = model.predict(X)[0].round()
        print(f"✅ 改良版予測完了: {salary}万円")
        
        # モデル情報
        model_info = f"{type(model).__name__} (特徴量: {len(feature_columns)}次元)"
        
        return render_template("result.html", 
                             position=position, age=age, area=area,
                             salary=salary, result="success", 
                             model_info=model_info)
        
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        error_message = f"予測エラー: {str(e)}"
        return render_template("result.html", 
                             position=position if 'position' in locals() else 0,
                             age=age if 'age' in locals() else 0,
                             area=area if 'area' in locals() else 0,
                             salary=error_message, result="error", 
                             model_info="エラー")


if __name__ == "__main__":
    # 環境変数からポートを取得（Render、Heroku対応）
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)  # 本番環境ではdebug=False 