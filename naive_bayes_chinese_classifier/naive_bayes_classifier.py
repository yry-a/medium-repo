"""
朴素贝叶斯分类器核心实现
Naive Bayes Classifier Core Implementation

这个模块提供了基于scikit-learn的朴素贝叶斯分类器的封装实现，
支持多种朴素贝叶斯算法变体，包括文本分类和概率预测功能。
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from typing import List, Tuple, Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaiveBayesClassifier:
    """
    朴素贝叶斯分类器类
    
    支持多种朴素贝叶斯算法:
    - MultinomialNB: 适用于文本分类
    - GaussianNB: 适用于连续特征
    - BernoulliNB: 适用于二进制特征
    """
    
    def __init__(self, model_type: str = 'multinomial', vectorizer_type: str = 'tfidf'):
        """
        初始化朴素贝叶斯分类器
        
        Args:
            model_type: 模型类型 ('multinomial', 'gaussian', 'bernoulli')
            vectorizer_type: 向量化器类型 ('tfidf', 'count')
        """
        self.model_type = model_type
        self.vectorizer_type = vectorizer_type
        self.model = None
        self.vectorizer = None
        self.is_fitted = False
        
        # 初始化模型
        self._initialize_model()
        self._initialize_vectorizer()
    
    def _initialize_model(self):
        """初始化贝叶斯模型"""
        if self.model_type == 'multinomial':
            self.model = MultinomialNB()
        elif self.model_type == 'gaussian':
            self.model = GaussianNB()
        elif self.model_type == 'bernoulli':
            self.model = BernoulliNB()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def _initialize_vectorizer(self):
        """初始化特征向量化器"""
        if self.vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words=None,  # 中文停用词需要单独处理
                ngram_range=(1, 2)
            )
        elif self.vectorizer_type == 'count':
            self.vectorizer = CountVectorizer(
                max_features=10000,
                stop_words=None,
                ngram_range=(1, 2)
            )
        else:
            raise ValueError(f"不支持的向量化器类型: {self.vectorizer_type}")
    
    def fit(self, X: List[str], y: List[str]) -> 'NaiveBayesClassifier':
        """
        训练模型
        
        Args:
            X: 训练文本数据
            y: 训练标签
            
        Returns:
            self: 返回自身以支持链式调用
        """
        try:
            logger.info(f"开始训练模型，数据量: {len(X)}")
            
            # 特征向量化
            if self.model_type in ['multinomial', 'bernoulli']:
                X_vectorized = self.vectorizer.fit_transform(X)
            else:  # gaussian
                # 对于高斯朴素贝叶斯，我们使用TF-IDF特征的稠密表示
                X_vectorized = self.vectorizer.fit_transform(X).toarray()
            
            # 训练模型
            self.model.fit(X_vectorized, y)
            self.is_fitted = True
            
            logger.info("模型训练完成")
            return self
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def predict(self, X: List[str]) -> List[str]:
        """
        预测类别
        
        Args:
            X: 待预测的文本数据
            
        Returns:
            预测的类别列表
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 特征向量化
            if self.model_type in ['multinomial', 'bernoulli']:
                X_vectorized = self.vectorizer.transform(X)
            else:  # gaussian
                X_vectorized = self.vectorizer.transform(X).toarray()
            
            predictions = self.model.predict(X_vectorized)
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        预测类别概率
        
        Args:
            X: 待预测的文本数据
            
        Returns:
            预测概率矩阵，每行对应一个样本，每列对应一个类别
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 特征向量化
            if self.model_type in ['multinomial', 'bernoulli']:
                X_vectorized = self.vectorizer.transform(X)
            else:  # gaussian
                X_vectorized = self.vectorizer.transform(X).toarray()
            
            probabilities = self.model.predict_proba(X_vectorized)
            return probabilities
            
        except Exception as e:
            logger.error(f"概率预测失败: {str(e)}")
            raise
    
    def get_feature_names(self) -> List[str]:
        """获取特征名称"""
        if self.vectorizer is None:
            return []
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_classes(self) -> List[str]:
        """获取类别名称"""
        if not self.is_fitted:
            return []
        return self.model.classes_.tolist()
    
    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            X_test: 测试文本数据
            y_test: 测试标签
            
        Returns:
            包含各种评估指标的字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 预测
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'predictions': y_pred,
                'probabilities': y_proba
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def cross_validate(self, X: List[str], y: List[str], cv: int = 5) -> Dict[str, float]:
        """
        交叉验证
        
        Args:
            X: 文本数据
            y: 标签
            cv: 交叉验证折数
            
        Returns:
            交叉验证结果
        """
        try:
            # 特征向量化
            if self.model_type in ['multinomial', 'bernoulli']:
                X_vectorized = self.vectorizer.fit_transform(X)
            else:  # gaussian
                X_vectorized = self.vectorizer.fit_transform(X).toarray()
            
            # 交叉验证
            scores = cross_val_score(self.model, X_vectorized, y, cv=cv, scoring='accuracy')
            
            return {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"交叉验证失败: {str(e)}")
            raise


def create_classifier(model_type: str = 'multinomial', 
                     vectorizer_type: str = 'tfidf') -> NaiveBayesClassifier:
    """
    创建朴素贝叶斯分类器的便捷函数
    
    Args:
        model_type: 模型类型
        vectorizer_type: 向量化器类型
        
    Returns:
        朴素贝叶斯分类器实例
    """
    return NaiveBayesClassifier(model_type=model_type, vectorizer_type=vectorizer_type)


if __name__ == "__main__":
    # 简单测试
    print("朴素贝叶斯分类器测试")
    
    # 创建测试数据
    X_train = [
        "这是一个正面的评论",
        "这个产品很好",
        "我很喜欢这个",
        "这是一个负面的评论",
        "这个产品很差",
        "我不喜欢这个"
    ]
    y_train = ["positive", "positive", "positive", "negative", "negative", "negative"]
    
    X_test = ["这个很好", "这个很差"]
    
    # 创建并训练分类器
    classifier = create_classifier()
    classifier.fit(X_train, y_train)
    
    # 预测
    predictions = classifier.predict(X_test)
    probabilities = classifier.predict_proba(X_test)
    
    print(f"预测结果: {predictions}")
    print(f"预测概率: {probabilities}")
    
    # 获取类别和特征
    print(f"类别: {classifier.get_classes()}")
    print(f"特征数量: {len(classifier.get_feature_names())}")