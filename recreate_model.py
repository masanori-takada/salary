import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

print("🚀 モデル再作成を開始します...")

# データ読み込み
print("📂 データを読み込み中...")
df = pd.read_csv('./data/dataset/train_data.csv')
print(f"✅ データ読み込み完了: {df.shape}")

# データクリーニング
print("🧹 データクリーニング中...")
df = df.dropna()
df = df.reset_index(drop=True)
print(f"✅ 欠損値除去後: {df.shape}")

# エリアのラベルエンコーディング
print("🔢 ラベルエンコーディング中...")
le = LabelEncoder()
df['area'] = le.fit_transform(df['area'])
print("✅ ラベルエンコーディング完了")

# データフレームを保存
print("💾 データフレームを保存中...")
df.to_pickle('./data/df.pkl')
print("✅ df.pkl保存完了")

# 特徴量とターゲットを分離
print("🎯 特徴量とターゲットを分離中...")
X = df.iloc[:, 1:-1].values  # 最初の列を除く、最後の列を除く
y = df.iloc[:, -1].values    # 最後の列（給与）
print(f"✅ 特徴量: {X.shape}, ターゲット: {y.shape}")

# 最適パラメータでモデル作成（Notebookから取得）
print("🤖 LightGBMモデルを訓練中...")
model = LGBMRegressor(
    n_estimators=848,
    max_depth=16,
    num_leaves=10,
    min_child_weight=19,
    subsample=0.7,
    colsample_bytree=0.9,
    random_state=0,
    verbose=-1  # ログを抑制
)

# モデル訓練
model.fit(X, y)
print("✅ モデル訓練完了")

# モデル検証
print("🔍 モデル検証中...")
sample_prediction = model.predict(X[:5])
print(f"✅ サンプル予測: {sample_prediction}")

# モデル保存
print("💾 モデルを保存中...")
with open("./data/model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ model.pkl保存完了")

print("🎉 モデル再作成が完了しました！")
print("🚀 Flaskアプリを再起動してください") 