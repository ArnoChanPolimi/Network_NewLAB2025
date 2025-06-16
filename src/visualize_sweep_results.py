import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取结果
csv_path = 'results/sweep_window_results.csv'
df = pd.read_csv(csv_path)

sns.set(style="whitegrid")

# F1 分数热力图
pivot_f1 = df.pivot(index='N', columns='X', values='f1')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_f1, annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("F1 Score Heatmap")
plt.ylabel("Lookback Window N")
plt.xlabel("Prediction Window X")
plt.tight_layout()
plt.savefig('results/sweep_f1_heatmap.png')
plt.show()

# 召回率热力图
pivot_recall = df.pivot(index='N', columns='X', values='recall')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_recall, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Recall Heatmap")
plt.ylabel("Lookback Window N")
plt.xlabel("Prediction Window X")
plt.tight_layout()
plt.savefig('results/sweep_recall_heatmap.png')
plt.show()

# 精确率热力图
pivot_precision = df.pivot(index='N', columns='X', values='precision')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_precision, annot=True, fmt=".2f", cmap="Blues")
plt.title("Precision Heatmap")
plt.ylabel("Lookback Window N")
plt.xlabel("Prediction Window X")
plt.tight_layout()
plt.savefig('results/sweep_precision_heatmap.png')
plt.show()

# ROC AUC 热力图
pivot_auc = df.pivot(index='N', columns='X', values='roc_auc')
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_auc, annot=True, fmt=".2f", cmap="Greens")
plt.title("ROC AUC Heatmap")
plt.ylabel("Lookback Window N")
plt.xlabel("Prediction Window X")
plt.tight_layout()
plt.savefig('results/sweep_rocauc_heatmap.png')
plt.show() 