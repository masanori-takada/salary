import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# 機械学習モデルの読込
with open("./data/model.pkl", "rb") as f:
    model = pickle.load(f)
# データフレームの読込
with open("./data/df.pkl", "rb") as f:
    df = pickle.load(f)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
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

    salary = model.predict([param])[0].round()
    # pred_df = df[df["salary"] == pred_id].copy()
    # salary = df["salary"].values[0]

    return render_template("result.html", param=param, salary=salary, result="")

    # データフレームを昇順でソート。上から5つの行を取得
    # result_df = pred_df.sort_values(by="Distance", ascending="True")

    # # 容量(ounces)違いでビールが重複することがあるので、重複を除去する
    # result_df = result_df.drop_duplicates(["name"])
