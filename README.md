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

## 许可证
本项目采用 MIT 许可证 - 详见 LICENSE 文件


## 今日工作总结 (2025-06-18)

### 随机森林模型优化
1. 特征工程改进
   - 增加了尾部敏感特征(rolling_std_last_5, rolling_mean_diff_last_3)
   - 通过重复尾部特征来增加其权重(重复3次)
   - 保留了原始时序窗口数据作为基础特征
   - 添加了统计特征:
     - 基础统计量(均值、标准差、最大最小值、中位数等)
     - 趋势特征(斜率、最后一个值与前值均值的差值)
     - 变异系数(std/mean)

2. 模型参数调优
   - 使用balanced_subsample类权重处理样本不平衡
   - 增加决策树数量至300(从100提升)
   - 调整最大深度为12
   - 设置最小叶节点样本数为3
   - 使用随机状态42保证实验可重复性

3. 实验结果
   - 在单个数据集(cpe_a-cpe_b-mobile)上表现优异
     - 准确率显著提升
     - False Negative大幅减少
     - 特征重要性分布更合理,尾部特征权重提升明显
   - 跨数据集泛化性能
     - 在部分数据集(如cpe_c-cpe_a)上表现欠佳
     - 显示出模型对数据集特征分布敏感
   - 多数据集联合训练(random_forest_o4_all.py)
     - 合并了两次采集的数据集进行训练
     - 相比单数据集训练有一定泛化性提升
     - 但仍存在过拟合现象

4. 不同配置对比
   - 仅使用统计特征(random_forest_with_only_stat_feature.py)
     - 模型更简单,训练更快
     - 但预测性能几乎没有改变
   - 使用完整特征集(random_forest_o4.py)
     - 在单数据集上达到最佳效果
     - 特征重要性分析显示时序窗口原始数据贡献较大

### 待改进方向
1. 提升模型泛化能力
   - 考虑使用更多数据集联合训练
   - 探索更具普适性的特征
   - 研究不同场景下的特征重要性变化
2. 优化特征选择
   - 分析表现差异大的数据集特征
   - 寻找更稳定的预测指标
   - 考虑特征选择算法
3. 模型结构优化
   - 尝试集成学习方法
   - 研究不同的类别权重方案
   - 进一步调优超参数


4. 神经网络与LSTM实验
   - 神经网络实现(neural_network.py)
     - 采用与随机森林相同的特征工程方案
     - 在大多数数据集上达到与随机森林相近的召回率
     - 训练时间较长但推理速度快
   - LSTM模型尝试(lstm_model.py) 
     - 专门针对随机森林和神经网络表现欠佳的数据集
     - 对loss类别的检测准确率更高
     - 存在一定的假阳性(将正常样本误判为loss)
     - 优势:
       - 能更好地捕捉时序特征中的长期依赖
       - 对特定类型的loss模式识别效果好
     - 局限:
       - 计算开销较大
       - 需要更多训练数据
       - 存在少量误报

5. 多模型集成效果
   - 三种模型各有优势:
     - 随机森林: 稳定性好,训练快
     - 神经网络: 整体表现均衡,推理快
     - LSTM: 对特定loss检测准确
   - 建议根据具体场景选择:
     - 对实时性要求高的场景使用神经网络
     - 对准确率要求高的场景使用LSTM
     - 需要快速迭代的场景使用随机森林

6. 待讨论问题
   - 数据集训练策略
     - 是否应该合并所有数据集一起训练?
     - 对于往返线路,往向和返向是否应分开训练?
   - 窗口大小选择
     - 不同链路和方向是否需要不同的lookback/predict window大小?
     - 目前发现predict window=1时准确率最高
     - 但1秒预测窗口是否能满足实际需求?
     - 是否需要更大的预测窗口来满足实际应用?
   - 模型泛化性能
     - 如何平衡准确率和预测窗口大小?
     - 不同场景下模型性能如何权衡?
     - 如何提升模型在更大预测窗口下的表现?
   - 特征选择策略
     - 使用lookback window的统计特征是否意味着必须同时使用原始时序数据?
     - 是否可以只用统计特征而不使用原始数据?
     - 两种特征的组合是否能带来更好的效果?

6. Let's Talk About These Questions
   - About Training Our Datasets
     - Should we throw all our data together for training, or keep them separate?
     - When we've got two-way routes, do we need to split up the forward and backward data?
   - About Picking Window Sizes
     - Do we need to play around with different window sizes for different routes?
     - We're getting our best results with a 1-second prediction window right now
     - But is that really enough for what we need in the real world?
     - Maybe we should look at bigger prediction windows?
   - About Making Our Model Work Better
     - How do we get the sweet spot between accuracy and window size?
     - What's the best way to handle different scenarios?
     - Any ideas on making the model work better with bigger prediction windows?
   - About Choosing Our Features
     - If we're using stats from our lookback window, do we really need all the raw data too?
     - Could we get away with just using the stats?
     - Would mixing both types of data give us better results?

