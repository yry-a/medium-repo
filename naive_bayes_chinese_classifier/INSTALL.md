# 安装指南 / Installation Guide

## 快速开始 / Quick Start

### 1. 无依赖演示 / Demo Without Dependencies

如果您想快速体验系统功能，可以直接运行演示脚本：

```bash
cd naive_bayes_chinese_classifier
python demo_without_dependencies.py
```

这个演示脚本使用纯Python实现，展示了：
- 中文文本分类
- 中文姓名性别判定  
- 选词填空功能

### 2. 完整功能安装 / Full Feature Installation

#### 环境要求 / Requirements
- Python 3.7+
- pip (Python包管理器)

#### 安装步骤 / Installation Steps

1. **克隆项目 / Clone Repository**
```bash
git clone https://github.com/yry-a/medium-repo.git
cd medium-repo/naive_bayes_chinese_classifier
```

2. **创建虚拟环境（推荐）/ Create Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 Windows: venv\Scripts\activate
```

3. **安装依赖包 / Install Dependencies**
```bash
pip install -r requirements.txt
```

如果网络连接有问题，可以尝试：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

#### 依赖包说明 / Dependencies Description

- **scikit-learn**: 机器学习库，提供朴素贝叶斯算法
- **jieba**: 中文分词库
- **pandas**: 数据处理库
- **numpy**: 数值计算库
- **matplotlib**: 数据可视化库
- **seaborn**: 统计图表库

### 3. 验证安装 / Verify Installation

运行以下命令验证安装是否成功：

```bash
# 测试核心模块
python -c "from naive_bayes_classifier import NaiveBayesClassifier; print('Core module OK')"

# 测试性别预测
python -c "from gender_prediction import ChineseNameGenderPredictor; print('Gender prediction OK')"

# 测试文本分类
python -c "from text_classification import ChineseTextClassifier; print('Text classification OK')"

# 测试选词填空
python -c "from word_filling import WordFillingClassifier; print('Word filling OK')"
```

### 4. 运行示例 / Run Examples

#### 基本功能测试 / Basic Function Tests

```bash
# 测试数据预处理
python data_preprocessing.py

# 测试性别预测
python gender_prediction.py

# 测试文本分类
python text_classification.py

# 测试选词填空
python word_filling.py
```

#### 高级示例 / Advanced Examples

```bash
cd examples
python test_word_filling.py
```

## 常见问题 / FAQ

### Q1: 安装依赖时出现网络超时错误
**A1**: 尝试使用国内镜像源：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### Q2: 中文显示乱码
**A2**: 确保您的终端支持UTF-8编码：
```bash
export LANG=zh_CN.UTF-8
export LC_ALL=zh_CN.UTF-8
```

### Q3: jieba分词库安装失败
**A3**: 可以尝试：
```bash
pip install jieba --user
```

### Q4: 内存不足错误
**A4**: 在大数据集上训练时，可以：
- 减少max_features参数
- 分批处理数据
- 使用数据采样

### Q5: 模型准确率较低
**A5**: 可以尝试：
- 增加训练数据量
- 调整特征提取参数
- 使用超参数优化
- 进行特征工程

## 开发环境设置 / Development Setup

如果您想参与开发或修改代码：

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install pytest jupyter notebook

# 运行测试
python -m pytest tests/

# 启动Jupyter Notebook
jupyter notebook
```

## 性能优化建议 / Performance Tips

1. **大数据集处理**：使用批处理和内存映射
2. **特征选择**：限制max_features参数
3. **模型选择**：根据数据特性选择合适的朴素贝叶斯变体
4. **并行处理**：使用n_jobs参数启用并行计算

## 支持与反馈 / Support & Feedback

如果您遇到问题或有改进建议，请：

1. 查看 [README.md](README.md) 了解详细使用方法
2. 检查 [常见问题](#常见问题--faq) 部分
3. 在GitHub上提交Issue
4. 联系项目维护者

---

**注意**: 本项目专为中文文本处理优化，在其他语言上的效果可能不佳。