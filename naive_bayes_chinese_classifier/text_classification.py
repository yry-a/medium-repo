"""
中文文本分类模块
Chinese Text Classification Module

基于朴素贝叶斯分类器实现中文文本的多类别分类功能，
包括文本预处理、特征提取、模型训练和分类预测。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import logging
import pickle
import os

from data_preprocessing import ChineseTextPreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseTextClassifier:
    """中文文本分类器"""
    
    def __init__(self, model_type: str = 'multinomial', vectorizer_type: str = 'tfidf',
                 max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        初始化文本分类器
        
        Args:
            model_type: 模型类型 ('multinomial', 'complement')
            vectorizer_type: 向量化器类型 ('tfidf', 'count')
            max_features: 最大特征数
            ngram_range: N-gram范围
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.preprocessor = ChineseTextPreprocessor()
        self.pipeline = None
        self.classes_ = None
        self.is_fitted = False
        
        # 创建管道
        self._create_pipeline()
    
    def _create_pipeline(self):
        """创建文本分类管道"""
        # 选择向量化器
        if self.vectorizer_type == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=None,  # 已在预处理中处理停用词
                lowercase=False,  # 中文不需要转小写
                token_pattern=r'(?u)\b\w+\b'  # 适合中文的词汇模式
            )
        elif self.vectorizer_type == 'count':
            vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=None,
                lowercase=False,
                token_pattern=r'(?u)\b\w+\b'
            )
        else:
            raise ValueError(f"不支持的向量化器类型: {self.vectorizer_type}")
        
        # 选择分类器
        if self.model_type == 'multinomial':
            classifier = MultinomialNB()
        elif self.model_type == 'complement':
            classifier = ComplementNB()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 创建管道
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        预处理文本
        
        Args:
            texts: 原始文本列表
            
        Returns:
            预处理后的文本列表
        """
        processed_texts = []
        for text in texts:
            # 分词并连接
            words = self.preprocessor.segment_text(text)
            processed_text = ' '.join(words)
            processed_texts.append(processed_text)
        
        return processed_texts
    
    def fit(self, texts: List[str], labels: List[str]) -> 'ChineseTextClassifier':
        """
        训练文本分类模型
        
        Args:
            texts: 训练文本列表
            labels: 训练标签列表
            
        Returns:
            self: 返回自身以支持链式调用
        """
        try:
            logger.info(f"开始训练文本分类模型，数据量: {len(texts)}")
            
            # 预处理文本
            processed_texts = self._preprocess_texts(texts)
            
            # 训练管道
            self.pipeline.fit(processed_texts, labels)
            self.classes_ = self.pipeline.named_steps['classifier'].classes_
            self.is_fitted = True
            
            logger.info(f"文本分类模型训练完成，类别数: {len(self.classes_)}")
            return self
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        预测文本类别
        
        Args:
            texts: 待预测的文本列表
            
        Returns:
            预测的类别列表
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 预处理文本
            processed_texts = self._preprocess_texts(texts)
            
            # 预测
            predictions = self.pipeline.predict(processed_texts)
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"文本分类预测失败: {str(e)}")
            raise
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        预测文本类别概率
        
        Args:
            texts: 待预测的文本列表
            
        Returns:
            预测概率矩阵
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 预处理文本
            processed_texts = self._preprocess_texts(texts)
            
            # 预测概率
            probabilities = self.pipeline.predict_proba(processed_texts)
            return probabilities
            
        except Exception as e:
            logger.error(f"概率预测失败: {str(e)}")
            raise
    
    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """
        预测单个文本的类别
        
        Args:
            text: 待预测的文本
            
        Returns:
            (预测类别, 预测概率, 所有类别概率字典)
        """
        predictions = self.predict([text])
        probabilities = self.predict_proba([text])[0]
        
        predicted_class = predictions[0]
        predicted_prob = max(probabilities)
        
        # 创建所有类别的概率字典
        all_probs = {}
        for i, class_name in enumerate(self.classes_):
            all_probs[class_name] = float(probabilities[i])
        
        return predicted_class, predicted_prob, all_probs
    
    def evaluate(self, texts: List[str], true_labels: List[str]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            texts: 测试文本列表
            true_labels: 真实标签列表
            
        Returns:
            评估结果字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 预测
            predicted_labels = self.predict(texts)
            probabilities = self.predict_proba(texts)
            
            # 计算指标
            accuracy = accuracy_score(true_labels, predicted_labels)
            report = classification_report(true_labels, predicted_labels, output_dict=True)
            conf_matrix = confusion_matrix(true_labels, predicted_labels)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'predictions': predicted_labels,
                'probabilities': probabilities,
                'true_labels': true_labels,
                'classes': self.classes_.tolist()
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        if not self.is_fitted:
            return []
        
        vectorizer = self.pipeline.named_steps['vectorizer']
        return vectorizer.get_feature_names_out().tolist()
    
    def get_top_features_per_class(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        获取每个类别的重要特征
        
        Args:
            top_n: 每个类别返回的顶部特征数
            
        Returns:
            每个类别的重要特征字典
        """
        if not self.is_fitted:
            return {}
        
        try:
            # 获取分类器和特征名
            classifier = self.pipeline.named_steps['classifier']
            feature_names = self.get_feature_names()
            
            # 获取特征权重
            if hasattr(classifier, 'feature_log_prob_'):
                feature_weights = classifier.feature_log_prob_
            elif hasattr(classifier, 'coef_'):
                feature_weights = classifier.coef_
            else:
                logger.warning("无法获取特征权重")
                return {}
            
            class_features = {}
            
            for i, class_name in enumerate(self.classes_):
                # 获取该类别的特征权重
                weights = feature_weights[i]
                
                # 获取权重最高的特征
                top_indices = np.argsort(weights)[-top_n:][::-1]
                
                top_features = []
                for idx in top_indices:
                    if idx < len(feature_names):
                        feature_name = feature_names[idx]
                        weight = float(weights[idx])
                        top_features.append((feature_name, weight))
                
                class_features[class_name] = top_features
            
            return class_features
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {str(e)}")
            return {}
    
    def optimize_hyperparameters(self, texts: List[str], labels: List[str], 
                                cv: int = 5) -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            texts: 训练文本
            labels: 训练标签
            cv: 交叉验证折数
            
        Returns:
            最优参数和评分
        """
        try:
            logger.info("开始超参数优化...")
            
            # 预处理文本
            processed_texts = self._preprocess_texts(texts)
            
            # 定义参数网格
            param_grid = {
                'vectorizer__max_features': [5000, 10000, 15000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'classifier__alpha': [0.1, 0.5, 1.0, 2.0]
            }
            
            # 网格搜索
            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=cv, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(processed_texts, labels)
            
            # 更新管道为最优参数
            self.pipeline = grid_search.best_estimator_
            self.classes_ = self.pipeline.named_steps['classifier'].classes_
            self.is_fitted = True
            
            logger.info(f"超参数优化完成，最优得分: {grid_search.best_score_:.4f}")
            
            return {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            logger.error(f"超参数优化失败: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，无法保存")
        
        try:
            model_data = {
                'pipeline': self.pipeline,
                'classes_': self.classes_,
                'model_type': self.model_type,
                'vectorizer_type': self.vectorizer_type,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'is_fitted': self.is_fitted
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"模型已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"模型保存失败: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """
        从文件加载模型
        
        Args:
            filepath: 模型文件路径
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"模型文件不存在: {filepath}")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.pipeline = model_data['pipeline']
            self.classes_ = model_data['classes_']
            self.model_type = model_data['model_type']
            self.vectorizer_type = model_data['vectorizer_type']
            self.max_features = model_data['max_features']
            self.ngram_range = model_data['ngram_range']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"模型已从{filepath}加载")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise


class SentimentClassifier(ChineseTextClassifier):
    """情感分类器（继承自文本分类器）"""
    
    def __init__(self, **kwargs):
        """初始化情感分类器"""
        super().__init__(**kwargs)
    
    def create_sample_sentiment_data(self, n_samples: int = 1000) -> Tuple[List[str], List[str]]:
        """
        创建示例情感分析数据
        
        Args:
            n_samples: 样本数量
            
        Returns:
            (文本列表, 情感标签列表)
        """
        positive_texts = [
            "这个产品质量很好，我很满意",
            "服务态度非常棒，推荐大家购买",
            "物流很快，包装也很精美",
            "性价比很高，值得购买",
            "客服很耐心，解答了我所有问题",
            "产品功能强大，使用很方便",
            "质量超出预期，非常喜欢",
            "商家服务很好，会再次购买"
        ]
        
        negative_texts = [
            "产品质量很差，不建议购买",
            "服务态度恶劣，很不满意",
            "物流太慢了，包装也破损",
            "性价比不高，浪费钱",
            "客服态度很差，问题没有解决",
            "产品功能有问题，不好用",
            "质量不如预期，很失望",
            "商家不负责任，不会再买"
        ]
        
        neutral_texts = [
            "产品还可以吧，没什么特别的",
            "服务一般般，中规中矩",
            "物流正常，包装普通",
            "价格合理，功能基本够用",
            "客服回复及时，态度正常",
            "产品符合描述，没有惊喜",
            "质量还行，凑合能用",
            "整体体验平平，不好不坏"
        ]
        
        texts = []
        labels = []
        
        np.random.seed(42)
        
        for _ in range(n_samples // 3):
            # 正面评论
            text = np.random.choice(positive_texts)
            texts.append(text)
            labels.append('positive')
            
            # 负面评论
            text = np.random.choice(negative_texts)
            texts.append(text)
            labels.append('negative')
            
            # 中性评论
            text = np.random.choice(neutral_texts)
            texts.append(text)
            labels.append('neutral')
        
        return texts, labels


def create_news_classification_data(n_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """
    创建示例新闻分类数据
    
    Args:
        n_samples: 样本数量
        
    Returns:
        (新闻文本列表, 类别标签列表)
    """
    categories = {
        'technology': [
            "人工智能技术在医疗领域取得重大突破",
            "5G网络建设加速推进，覆盖率大幅提升",
            "新型芯片技术发布，性能提升显著",
            "云计算服务市场持续增长",
            "自动驾驶技术测试取得新进展"
        ],
        'sports': [
            "足球世界杯比赛激烈进行",
            "篮球联赛总决赛即将开始",
            "奥运会筹备工作有序推进",
            "网球公开赛精彩对决",
            "游泳世锦赛创造新纪录"
        ],
        'finance': [
            "股市今日大幅上涨，投资者信心增强",
            "央行宣布调整货币政策",
            "房地产市场出现新变化",
            "外汇汇率波动引起关注",
            "银行业绩报告显示良好增长"
        ],
        'entertainment': [
            "新电影上映获得好评",
            "音乐会门票一票难求",
            "电视剧收视率创新高",
            "明星参加慈善活动",
            "综艺节目引起热议"
        ]
    }
    
    texts = []
    labels = []
    
    np.random.seed(42)
    
    for _ in range(n_samples):
        category = np.random.choice(list(categories.keys()))
        text = np.random.choice(categories[category])
        
        texts.append(text)
        labels.append(category)
    
    return texts, labels


def train_and_evaluate_text_classifier(texts: List[str], labels: List[str], 
                                     test_size: float = 0.2, 
                                     model_type: str = 'multinomial') -> Dict[str, Any]:
    """
    训练和评估文本分类模型的便捷函数
    
    Args:
        texts: 文本列表
        labels: 标签列表
        test_size: 测试集比例
        model_type: 模型类型
        
    Returns:
        评估结果字典
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # 创建和训练模型
    classifier = ChineseTextClassifier(model_type=model_type)
    classifier.fit(X_train, y_train)
    
    # 评估模型
    evaluation_results = classifier.evaluate(X_test, y_test)
    
    # 添加模型实例到结果中
    evaluation_results['model'] = classifier
    evaluation_results['train_size'] = len(X_train)
    evaluation_results['test_size'] = len(X_test)
    
    return evaluation_results


if __name__ == "__main__":
    print("中文文本分类测试")
    
    # 测试情感分类
    print("\n=== 情感分类测试 ===")
    sentiment_classifier = SentimentClassifier()
    texts, labels = sentiment_classifier.create_sample_sentiment_data(600)
    
    print(f"情感数据样本: {len(texts)}")
    print("前5个样本:")
    for i in range(5):
        print(f"  {texts[i]} - {labels[i]}")
    
    # 训练情感分类模型
    results = train_and_evaluate_text_classifier(texts, labels)
    print(f"情感分类准确率: {results['accuracy']:.4f}")
    
    # 测试单个预测
    model = results['model']
    test_texts = [
        "这个产品真的很棒，强烈推荐",
        "质量太差了，完全不值这个价格",
        "还可以吧，没什么特别的感觉"
    ]
    
    print("\n单个情感预测:")
    for text in test_texts:
        pred_class, pred_prob, all_probs = model.predict_single(text)
        print(f"文本: {text}")
        print(f"预测: {pred_class} (概率: {pred_prob:.4f})")
        print(f"所有概率: {all_probs}")
        print()
    
    # 测试新闻分类
    print("\n=== 新闻分类测试 ===")
    news_texts, news_labels = create_news_classification_data(800)
    
    print(f"新闻数据样本: {len(news_texts)}")
    print("前5个样本:")
    for i in range(5):
        print(f"  {news_texts[i]} - {news_labels[i]}")
    
    # 训练新闻分类模型
    news_results = train_and_evaluate_text_classifier(news_texts, news_labels)
    print(f"新闻分类准确率: {news_results['accuracy']:.4f}")
    
    # 获取特征重要性
    print("\n新闻分类 - 每个类别的重要特征:")
    news_model = news_results['model']
    top_features = news_model.get_top_features_per_class(top_n=5)
    
    for class_name, features in top_features.items():
        print(f"{class_name}:")
        for feature, weight in features:
            print(f"  {feature}: {weight:.4f}")
        print()