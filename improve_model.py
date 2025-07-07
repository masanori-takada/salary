import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from lightgbm import LGBMRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("🚀 精度改善版モデルを作成します...")

# データ読み込み
print("📂 データを読み込み中...")
df = pd.read_csv('./data/dataset/train_data.csv')
print(f"✅ データ読み込み完了: {df.shape}")

# データクリーニング
print("🧹 データクリーニング中...")
df = df.dropna()
df = df.reset_index(drop=True)

# エリアのラベルエンコーディング
le = LabelEncoder()
df['area'] = le.fit_transform(df['area'])

print("🔧 特徴量エンジニアリング中...")

# 新しい特徴量を作成
df['age_position_ratio'] = df['age'] / (df['position'] + 1)  # 年齢と職位の比率
df['experience_ratio'] = df['service_length'] / df['age']    # 勤続年数と年齢の比率
df['study_overtime_ratio'] = df['study_time'] / (df['overtime'] + 1)  # 勉強時間と残業時間の比率
df['total_time'] = df['commute'] + df['overtime']            # 総労働関連時間
df['family_burden'] = df['partner'] + df['num_child']        # 家族負担

# 年齢カテゴリ
df['age_category'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], labels=[1, 2, 3, 4, 5])
df['age_category'] = df['age_category'].astype(int)

print("✅ 特徴量エンジニアリング完了")

# 特徴量とターゲットを分離
feature_columns = [col for col in df.columns if col not in ['salary']]
X = df[feature_columns].values
y = df['salary'].values

print(f"✅ 改良版特徴量: {X.shape}, ターゲット: {y.shape}")

# データを訓練・検証用に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 複数モデルで性能比較中...")

# モデル一覧
models = {
    'LightGBM': LGBMRegressor(
        n_estimators=1000,
        max_depth=20,
        num_leaves=31,
        min_child_weight=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    ),
    'XGBoost': xgb.XGBRegressor(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
}

best_model = None
best_score = float('inf')
best_model_name = ""

for name, model in models.items():
    print(f"🔄 {name} を訓練中...")
    
    # 訓練
    model.fit(X_train, y_train)
    
    # 予測
    y_pred = model.predict(X_test)
    
    # 評価
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"📊 {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}")
    
    if mae < best_score:
        best_score = mae
        best_model = model
        best_model_name = name

print(f"🏆 最良モデル: {best_model_name} (MAE: {best_score:.2f})")

# アンサンブルモデル作成
print("🔮 アンサンブルモデルを作成中...")
ensemble_models = [models['LightGBM'], models['XGBoost'], models['RandomForest']]

# 各モデルの予測を取得
predictions = []
for model in ensemble_models:
    pred = model.predict(X_test)
    predictions.append(pred)

# 平均を取る（シンプルアンサンブル）
ensemble_pred = np.mean(predictions, axis=0)

# アンサンブルの評価
ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
ensemble_r2 = r2_score(y_test, ensemble_pred)

print(f"🎯 アンサンブル: MAE={ensemble_mae:.2f}, RMSE={ensemble_rmse:.2f}, R²={ensemble_r2:.4f}")

# 最良モデルをフルデータで再訓練
print(f"🔄 {best_model_name} をフルデータで再訓練中...")
final_model = type(best_model)(**best_model.get_params())
final_model.fit(X, y)

# サンプル予測
sample_pred = final_model.predict(X[:5])
print(f"✅ サンプル予測: {sample_pred}")

# 改良版データフレームを保存（新しい特徴量含む）
print("💾 改良版データフレームを保存中...")
df.to_pickle('./data/df_improved.pkl')

# 最良モデルを保存
print("💾 改良版モデルを保存中...")
with open("./data/model_improved.pkl", "wb") as f:
    pickle.dump(final_model, f)

# 特徴量名も保存
with open("./data/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("🎉 精度改善版モデル作成完了！")
print(f"📈 精度向上: {best_model_name}を採用")
print("🔄 Flaskアプリを改良版に更新してください") 