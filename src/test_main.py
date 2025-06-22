import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from preprocessing import load_data, clean_data, create_sliding_windows, extract_statistical_features
from feature_engineering import create_feature_matrix
from model_training import ModelTrainer, train_neural_network
from visualization import Visualizer
from feature_engineering import extract_time_domain_features, extract_advanced_features


def test_main():
    # 设置随机种子以确保结果可重现!!!
    np.random.seed(42)
    
    # 创建必要的目录
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # 初始化组件
    trainer = ModelTrainer()
    visualizer = Visualizer(results_dir)
    
    print("开始测试主程序...")
    
    # 1. 加载单个CSV文件
    print("\n1. 加载数据...")
    csv_path = '../dataset/1st_capture/cpe_a-cpe_b-fiber.csv'  # 只使用第一个文件
    df = load_data(csv_path)
    print(f"加载数据完成，数据形状: {df.shape}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    df = clean_data(df)
    print("预处理完成")
    
    # 3. 特征工程
    print("\n3. 特征工程...")
    features_dict = create_feature_matrix(df['delay_ms'].values)
    X = features_dict['feature_matrix']
    y = features_dict['labels']
    # feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    # 得到真实特征名字
    feature_names = list(extract_time_domain_features(np.zeros(10)).keys()) + list(extract_advanced_features(np.zeros(10)).keys())

    print(f"特征工程完成，特征数量: {X.shape[1]}")
    
    # 4. 数据集分割
    print("\n4. 数据集分割...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")
    
    print(f"全数据集丢包样本数: {np.sum(y)} / {len(y)}")  
    print(f"训练集丢包样本数: {np.sum(y_train)} / {len(y_train)}")  
    print(f"测试集丢包样本数: {np.sum(y_test)} / {len(y_test)}")  

    # 5. 训练随机森林模型
    print("\n5. 训练随机森林模型...")
    trainer.train_random_forest(X_train, y_train, feature_names)
    rf_model = trainer.rf_model
    rf_predictions = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # 计算随机森林模型指标
    rf_metrics = {
        'accuracy': accuracy_score(y_test, rf_predictions),
        'precision': precision_score(y_test, rf_predictions),
        'recall': recall_score(y_test, rf_predictions),
        'f1': f1_score(y_test, rf_predictions)
    }

    print("-------------------")
    print("随机森林模型指标:")
    for metric, value in rf_metrics.items():
        print(f"{metric}: {value:.4f}")

    cm = confusion_matrix(y_test, rf_predictions)
    tn, fp, fn, tp = cm.ravel()

    print("随机森林模型预测结果详细统计：")
    print(f"总丢包数(测试集真实正样本): {tp + fn}")
    print(f"成功预测丢包数(True Positives): {tp}")
    print(f"遗漏丢包数(False Negatives): {fn}")
    print(f"误报正常包数(False Positives): {fp}")
    print(f"正确预测正常包数(True Negatives): {tn}")
    print("------------------------------------")
    
    # 训练完模型 rf_model 后
    importances = rf_model.feature_importances_

    print("每个特征的权重(重要性), 共计14种特征:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance:.4f}")
    
    # 6. 训练神经网络模型
    print("\n6. 训练神经网络模型...")
    nn_model, history = train_neural_network(
        X_train, y_train, X_test, y_test,
        epochs=50,  # 增加训练轮数
        batch_size=128,  # 减小批量大小
        lr=0.001  # 调整学习率
    )
    
    # 使用PyTorch模型进行预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    nn_model.eval()
    with torch.no_grad():
        nn_proba = nn_model(X_test_tensor).cpu().numpy()
        nn_predictions = (nn_proba > 0.5).astype(int)
    
    # 计算神经网络模型指标
    nn_metrics = {
        'accuracy': accuracy_score(y_test, nn_predictions),
        'precision': precision_score(y_test, nn_predictions),
        'recall': recall_score(y_test, nn_predictions),
        'f1': f1_score(y_test, nn_predictions)
    }
    print("神经网络模型指标:")
    for metric, value in nn_metrics.items():
        print(f"{metric}: {value:.4f}")

    cm_nn = confusion_matrix(y_test, nn_predictions)
    tn, fp, fn, tp = cm_nn.ravel()

    print("神经网络模型预测结果详细统计：")
    print(f"总丢包数(测试集真实正样本): {tp + fn}")
    print(f"成功预测丢包数 (True Positives): {tp}")
    print(f"遗漏丢包数(False Negatives): {fn}")
    print(f"误报正常包数(False Positives): {fp}")
    print(f"正确预测正常包数(True Negatives): {tn}")

    
    # 7. 生成可视化
    print("\n7. 生成可视化...")
    
    # 数据分布可视化
    visualizer.plot_data_distribution(df['delay_ms'].values)
    print("已生成数据分布图")
    
    # 特征相关性矩阵
    visualizer.plot_correlation_matrix(X, feature_names)
    print("已生成特征相关性矩阵")
    
    # 绘制训练历史
    visualizer.plot_training_history(history)
    
    # 绘制特征重要性
    visualizer.plot_feature_importance(rf_model.feature_importances_, feature_names)
    
    # 绘制随机森林混淆矩阵
    y_pred_rf = rf_model.predict(X_test)
    visualizer.plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest')
    
    # 绘制神经网络混淆矩阵
    nn_model.eval()
    with torch.no_grad():
        y_pred_nn = (torch.sigmoid(nn_model(torch.tensor(X_test, dtype=torch.float32))) > 0.5).numpy()
    visualizer.plot_confusion_matrix(y_test, y_pred_nn, 'Neural Network')
    
    # 绘制ROC曲线
    visualizer.plot_roc_curve(y_test, y_pred_rf, 'Random Forest')
    visualizer.plot_roc_curve(y_test, y_pred_nn, 'Neural Network')
    
    # 绘制预测分布
    visualizer.plot_prediction_distribution(y_pred_rf, 'Random Forest')
    visualizer.plot_prediction_distribution(y_pred_nn, 'Neural Network')
    
    print("\n程序运行完成！所有可视化结果已保存到 results 目录")

if __name__ == "__main__":
    test_main() 