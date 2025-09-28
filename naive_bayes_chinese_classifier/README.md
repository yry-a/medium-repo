# 基于朴素贝叶斯的中文文本处理系统

这个项目实现了基于朴素贝叶斯分类器的中文文本处理系统，包含中文人名性别判定、文本分类和选词填空三个核心功能。

## 功能特性

### 1. 中文人名性别判定
- 基于朴素贝叶斯分类器的中文姓名性别预测
- 智能特征提取（姓氏、名字字符、性别倾向分析）
- 支持模型保存和加载
- 提供详细的性能评估

### 2. 文本分类
- 支持多类别中文文本分类
- 自动中文分词和停用词过滤
- 可配置的特征提取（TF-IDF、词袋模型）
- 超参数自动优化
- 特征重要性分析

### 3. 选词填空
- 基于上下文的词汇预测
- 多候选词概率计算
- 支持Top-K预测评估
- 上下文相似度计算

## 项目结构

```
naive_bayes_chinese_classifier/
├── naive_bayes_classifier.py      # 核心朴素贝叶斯分类器
├── data_preprocessing.py          # 数据预处理工具
├── gender_prediction.py           # 中文人名性别判定
├── text_classification.py         # 文本分类功能
├── word_filling.py                # 选词填空核心功能
├── requirements.txt               # 依赖包列表
├── README.md                      # 项目说明文档
└── examples/                      # 示例数据和测试用例
    ├── sample_names.csv           # 示例姓名数据
    ├── sample_texts.csv           # 示例文本数据
    └── test_word_filling.py       # 选词填空测试脚本
```

## 安装说明

### 1. 克隆项目

```bash
git clone https://github.com/yry-a/medium-repo.git
cd medium-repo/naive_bayes_chinese_classifier
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装中文分词工具

项目使用`jieba`进行中文分词，第一次运行时会自动下载词典。

## 使用方法

### 1. 中文人名性别判定

```python
from gender_prediction import ChineseNameGenderPredictor

# 创建预测器
predictor = ChineseNameGenderPredictor()

# 准备训练数据
names = ["张伟", "李娜", "王小明", "刘美丽"]
genders = ["male", "female", "male", "female"]

# 训练模型
predictor.fit(names, genders)

# 预测单个姓名
predicted_gender, probability = predictor.predict_single("陈静雅")
print(f"预测性别: {predicted_gender}, 概率: {probability:.4f}")

# 批量预测
test_names = ["张强", "李美", "王建国"]
predictions = predictor.predict(test_names)
print(f"批量预测结果: {predictions}")
```

### 2. 文本分类

```python
from text_classification import ChineseTextClassifier

# 创建分类器
classifier = ChineseTextClassifier(model_type='multinomial')

# 准备训练数据
texts = [
    "这个产品质量很好，我很满意",
    "服务态度很差，不推荐",
    "价格合理，功能基本够用"
]
labels = ["positive", "negative", "neutral"]

# 训练模型
classifier.fit(texts, labels)

# 预测单个文本
predicted_class, probability, all_probs = classifier.predict_single("这个东西真的很棒")
print(f"预测类别: {predicted_class}")
print(f"预测概率: {probability:.4f}")
print(f"所有类别概率: {all_probs}")
```

### 3. 选词填空

```python
from word_filling import WordFillingClassifier

# 创建选词填空分类器
classifier = WordFillingClassifier()

# 准备训练数据
texts = [
    "今天天气很好，我决定去公园散步",
    "这本书的内容很有趣，我读得很认真"
]
target_words = [
    ["天气", "公园", "散步"],
    ["书", "内容", "认真"]
]

# 训练模型
classifier.fit(texts, target_words)

# 选词填空
context = "今天___很好，我决定去公园散步"
candidates = ["天气", "心情", "运气", "身体"]
predictions = classifier.predict_word(context, candidates, top_k=3)

print("选词填空结果:")
for i, (word, prob) in enumerate(predictions):
    print(f"{i+1}. {word} (概率: {prob:.4f})")
```

## 示例和测试

### 运行基本测试

```bash
# 测试核心朴素贝叶斯分类器
python naive_bayes_classifier.py

# 测试数据预处理
python data_preprocessing.py

# 测试人名性别判定
python gender_prediction.py

# 测试文本分类
python text_classification.py

# 测试选词填空
python word_filling.py
```

### 运行示例脚本

```bash
cd examples
python test_word_filling.py
```

## 技术特点

### 朴素贝叶斯算法支持
- **MultinomialNB**: 适用于文本分类和选词填空
- **GaussianNB**: 适用于连续特征的性别判定
- **ComplementNB**: 适用于不平衡数据集的文本分类
- **BernoulliNB**: 适用于二进制特征

### 中文文本处理
- **智能分词**: 使用jieba进行精确中文分词
- **停用词过滤**: 内置中文停用词表
- **特征提取**: 支持TF-IDF和词袋模型特征提取
- **N-gram支持**: 可配置的N-gram特征

### 性能优化
- **特征选择**: 可配置的最大特征数量
- **超参数优化**: 网格搜索自动优化参数
- **交叉验证**: 内置交叉验证评估
- **模型持久化**: 支持模型保存和加载

## 性能评估

### 评估指标
- **准确率 (Accuracy)**: 整体分类准确性
- **精确率 (Precision)**: 每个类别的预测精确度
- **召回率 (Recall)**: 每个类别的识别完整性
- **F1分数**: 精确率和召回率的调和平均
- **混淆矩阵**: 详细的分类结果分析

### 选词填空特有指标
- **Top-1准确率**: 首选答案正确率
- **Top-3准确率**: 前三个答案的正确率
- **Top-5准确率**: 前五个答案的正确率

## 扩展功能

### 自定义特征
项目提供了灵活的特征扩展接口，可以轻松添加：
- 自定义文本预处理逻辑
- 新的特征提取方法
- 特定领域的词典支持

### 模型集成
支持与其他机器学习模型结合：
- 可以作为特征提取器使用
- 支持模型融合和集成学习
- 提供标准的sklearn接口

## 注意事项

1. **中文支持**: 确保您的Python环境支持UTF-8编码
2. **内存使用**: 大规模文本处理时注意内存使用情况
3. **模型选择**: 根据数据特性选择合适的朴素贝叶斯变体
4. **特征工程**: 合理的特征工程对模型性能至关重要

## 常见问题

### Q: 如何处理新词或未见过的词？
A: 系统会为未见过的词分配很小的概率，建议定期更新训练数据。

### Q: 如何提高模型准确率？
A: 可以尝试：
- 增加训练数据量
- 调整特征提取参数
- 使用超参数优化
- 进行特征工程

### Q: 支持哪些文件格式？
A: 目前支持CSV和Excel格式的数据文件。

## 贡献

欢迎提交问题和改进建议：
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证，详情请参见 [LICENSE](../LICENSE) 文件。

## 作者

Var Group Data Science Team

## 更新日志

### v1.0.0
- 初始版本发布
- 实现中文人名性别判定
- 实现文本分类功能
- 实现选词填空功能
- 提供完整的示例和文档