# 网络数据分析实验室项目 2024-2025

## 项目概述
本项目专注于使用机器学习技术预测网络数据中的数据包丢失事件。项目实现了随机森林和神经网络两种模型，用于分析网络延迟模式并预测潜在的数据包丢失事件。

## 功能特点
- **数据预处理**
  - 数据加载和清洗
  - 处理缺失值和数据包丢失事件
  - 滑动窗口特征创建
  - 统计特征提取

- **特征工程**
  - 时域特征
  - 高级统计特征
  - 特征相关性分析
  - 特征重要性可视化

- **模型训练**
  - 随机森林分类器
  - 神经网络（多层感知器）
  - 模型评估和比较
  - 超参数优化

- **可视化**
  - 数据分布图
  - 特征重要性可视化
  - 混淆矩阵
  - ROC曲线
  - 模型比较图
  - 训练历史图
  - 特征相关性矩阵
  - 预测分布图

## 项目结构
```
├── src/
│   ├── preprocessing.py      # 数据预处理函数
│   ├── feature_engineering.py # 特征创建和工程
│   ├── model_training.py     # 模型训练和评估
│   ├── visualization.py      # 可视化工具
│   ├── main.py              # 主执行脚本
│   └── test_*.py            # 各组件测试脚本
├── dataset/
│   ├── 1st_capture/         # 第一次捕获数据集
│   └── 2nd_capture/         # 第二次捕获数据集
├── results/                 # 结果输出目录
├── config.yaml             # 配置文件
└── requirements.txt        # 项目依赖
```

## 安装说明
1. 克隆仓库：
```bash
git clone [仓库地址]
cd network-data-analysis-lab-project
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明
1. 准备数据：
   - 将网络数据CSV文件放入 `dataset/1st_capture` 和 `dataset/2nd_capture` 目录
   - 确保数据文件包含必要的列（time, delay_ms）

2. 配置项目：
   - 查看并修改 `config.yaml` 以满足特定需求
   - 调整窗口大小、模型参数等

3. 运行分析：
```bash
python src/main.py
```

4. 查看结果：
   - 检查 `results` 目录中的：
     - 模型性能指标
     - 可视化图表
     - 特征重要性分析
     - 模型比较结果

## 模型详情

### 随机森林模型
- 针对数据包丢失预测优化
- 特点：
  - 平衡类别权重
  - 200个决策树
  - 最大深度为10
  - 最小分裂样本数为5
  - 最小叶节点样本数为2

### 神经网络模型
- 简单多层感知器架构
- 特点：
  - 两个隐藏层（64, 32个神经元）
  - ReLU激活函数
  - Sigmoid输出
  - Adam优化器
  - 二元交叉熵损失函数

## 可视化输出
项目在 `results` 目录下生成以下可视化图表：
- `delay_distribution.png`：网络延迟分布
- `feature_correlation_matrix.png`：特征相关性矩阵
- `feature_importance.png`：特征重要性
- `confusion_matrix.png`：模型预测准确度
- `training_history.png`：神经网络训练进度
- `model_comparison.png`：模型性能比较
- `roc_curves.png`：两个模型的ROC曲线
- `prediction_distribution.png`：预测分布

## 测试说明
项目包含以下测试脚本：
- `test_rf.py`：随机森林模型测试
- `test_nn.py`：神经网络模型测试
- `test_sweep_window.py`：窗口大小参数优化
- `test_all_rf.py`：随机森林综合测试

## 配置说明
`config.yaml` 文件包含以下可配置参数：
- 数据路径
- 模型参数
- 训练设置
- 可视化选项

## 结果说明
分析结果保存在 `results` 目录中：
- 模型指标（准确率、精确率、召回率、F1分数）
- 特征重要性排名
- 模型比较表格
- 各种可视化图表

## 贡献指南
1. Fork 本仓库
2. 创建您的特性分支
3. 提交您的更改
4. 推送到分支
5. 创建 Pull Request

## 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件
