import os
from flask import Flask, render_template, request, jsonify
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test():
    """デプロイテスト用エンドポイント"""
    return jsonify({
        "status": "success",
        "message": "給与予測アプリが正常に動作しています！",
        "version": "Simple Version",
        "python_version": f"Python {os.sys.version}"
    })

@app.route("/result", methods=["POST"])
def result():
    try:
        # 基本的な計算による給与予測（機械学習モデルなし）
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

        # 簡易給与計算式（機械学習モデルの代替）
        base_salary = 200 + (position * 50) + (age * 3) + (education * 20)
        experience_bonus = service_length * 5
        study_bonus = study_time * 10
        area_adjustment = area * 20
        family_penalty = (partner * 10) + (num_child * 5)
        overtime_bonus = overtime * 2
        
        salary = max(150, base_salary + experience_bonus + study_bonus + 
                    area_adjustment - family_penalty + overtime_bonus)
        
        return render_template("result.html", 
                             position=position, age=age, area=area,
                             salary=round(salary), result="success", 
                             model_info="Simple Formula (Demo)")
        
    except Exception as e:
        error_message = f"計算エラー: {str(e)}"
        return render_template("result.html", 
                             position=0, age=0, area=0,
                             salary=error_message, result="error", 
                             model_info="Error")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 