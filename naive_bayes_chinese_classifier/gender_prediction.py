"""
中文人名性别判定模块
Chinese Name Gender Prediction Module

基于朴素贝叶斯分类器实现中文姓名的性别预测功能，
包括特征提取、模型训练、性别预测和性能评估。
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import logging
import pickle
import os

from data_preprocessing import ChineseNamePreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseNameGenderPredictor:
    """中文姓名性别预测器"""
    
    def __init__(self, model_type: str = 'gaussian'):
        """
        初始化性别预测器
        
        Args:
            model_type: 模型类型 ('gaussian', 'multinomial')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.preprocessor = ChineseNamePreprocessor()
        self.feature_columns = None
        self.is_fitted = False
        
        # 初始化模型
        if model_type == 'gaussian':
            self.model = GaussianNB()
        elif model_type == 'multinomial':
            self.model = MultinomialNB()
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    def _prepare_features(self, names: List[str]) -> np.ndarray:
        """
        准备特征矩阵
        
        Args:
            names: 姓名列表
            
        Returns:
            特征矩阵
        """
        # 使用预处理器提取特征
        feature_df = self.preprocessor.create_feature_vector(names)
        
        # 如果是训练阶段，保存特征列名
        if self.feature_columns is None:
            self.feature_columns = feature_df.columns.tolist()
        else:
            # 确保测试数据与训练数据的特征列一致
            missing_cols = set(self.feature_columns) - set(feature_df.columns)
            for col in missing_cols:
                feature_df[col] = 0
            
            # 按训练时的列顺序排列
            feature_df = feature_df[self.feature_columns]
        
        return feature_df.fillna(0).values
    
    def fit(self, names: List[str], genders: List[str]) -> 'ChineseNameGenderPredictor':
        """
        训练性别预测模型
        
        Args:
            names: 姓名列表
            genders: 性别标签列表 ('male', 'female' 或 '男', '女')
            
        Returns:
            self: 返回自身以支持链式调用
        """
        try:
            logger.info(f"开始训练性别预测模型，数据量: {len(names)}")
            
            # 标准化性别标签
            standardized_genders = self._standardize_gender_labels(genders)
            
            # 准备特征
            X = self._prepare_features(names)
            
            # 对于高斯朴素贝叶斯，进行特征标准化
            if self.model_type == 'gaussian':
                X = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X, standardized_genders)
            self.is_fitted = True
            
            logger.info("性别预测模型训练完成")
            return self
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def predict(self, names: List[str]) -> List[str]:
        """
        预测姓名性别
        
        Args:
            names: 待预测的姓名列表
            
        Returns:
            预测的性别列表
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 准备特征
            X = self._prepare_features(names)
            
            # 特征标准化（如果使用高斯朴素贝叶斯）
            if self.model_type == 'gaussian':
                X = self.scaler.transform(X)
            
            # 预测
            predictions = self.model.predict(X)
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"性别预测失败: {str(e)}")
            raise
    
    def predict_proba(self, names: List[str]) -> np.ndarray:
        """
        预测姓名性别概率
        
        Args:
            names: 待预测的姓名列表
            
        Returns:
            预测概率矩阵
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 准备特征
            X = self._prepare_features(names)
            
            # 特征标准化（如果使用高斯朴素贝叶斯）
            if self.model_type == 'gaussian':
                X = self.scaler.transform(X)
            
            # 预测概率
            probabilities = self.model.predict_proba(X)
            return probabilities
            
        except Exception as e:
            logger.error(f"概率预测失败: {str(e)}")
            raise
    
    def predict_single(self, name: str) -> Tuple[str, float]:
        """
        预测单个姓名的性别
        
        Args:
            name: 姓名
            
        Returns:
            (预测性别, 预测概率)
        """
        prediction = self.predict([name])[0]
        probabilities = self.predict_proba([name])[0]
        
        # 获取预测类别的概率
        classes = self.model.classes_
        prediction_idx = np.where(classes == prediction)[0][0]
        probability = probabilities[prediction_idx]
        
        return prediction, probability
    
    def evaluate(self, names: List[str], true_genders: List[str]) -> Dict[str, any]:
        """
        评估模型性能
        
        Args:
            names: 测试姓名列表
            true_genders: 真实性别标签
            
        Returns:
            评估结果字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 标准化真实标签
            standardized_true_genders = self._standardize_gender_labels(true_genders)
            
            # 预测
            predicted_genders = self.predict(names)
            probabilities = self.predict_proba(names)
            
            # 计算指标
            accuracy = accuracy_score(standardized_true_genders, predicted_genders)
            report = classification_report(standardized_true_genders, predicted_genders, output_dict=True)
            conf_matrix = confusion_matrix(standardized_true_genders, predicted_genders)
            
            return {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'predictions': predicted_genders,
                'probabilities': probabilities,
                'true_labels': standardized_true_genders
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def _standardize_gender_labels(self, genders: List[str]) -> List[str]:
        """
        标准化性别标签
        
        Args:
            genders: 原始性别标签
            
        Returns:
            标准化后的性别标签
        """
        standardized = []
        for gender in genders:
            if gender.lower() in ['male', '男', 'm', '1']:
                standardized.append('male')
            elif gender.lower() in ['female', '女', 'f', '0']:
                standardized.append('female')
            else:
                # 保持原样
                standardized.append(gender)
        return standardized
    
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
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_type': self.model_type,
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
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_type = model_data['model_type']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"模型已从{filepath}加载")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def get_feature_importance(self, names: List[str]) -> Dict[str, float]:
        """
        获取特征重要性（基于特征权重的近似）
        
        Args:
            names: 姓名样本
            
        Returns:
            特征重要性字典
        """
        if not self.is_fitted or self.feature_columns is None:
            return {}
        
        try:
            # 准备特征
            X = self._prepare_features(names)
            
            # 计算特征的方差作为重要性的代理
            feature_variance = np.var(X, axis=0)
            
            # 归一化
            total_variance = np.sum(feature_variance)
            if total_variance > 0:
                feature_importance = feature_variance / total_variance
            else:
                feature_importance = np.zeros_like(feature_variance)
            
            # 创建特征重要性字典
            importance_dict = {}
            for i, feature_name in enumerate(self.feature_columns):
                importance_dict[feature_name] = float(feature_importance[i])
            
            # 按重要性排序
            sorted_importance = dict(sorted(importance_dict.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            logger.error(f"特征重要性计算失败: {str(e)}")
            return {}


def create_sample_data(n_samples: int = 1000) -> Tuple[List[str], List[str]]:
    """
    创建示例中文姓名数据
    
    Args:
        n_samples: 样本数量
        
    Returns:
        (姓名列表, 性别标签列表)
    """
    # 常见姓氏
    surnames = ['李', '王', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴', '徐', '孙', '胡', '朱', '高', '林']
    
    # 男性名字
    male_names = ['伟', '强', '军', '杰', '华', '明', '建', '国', '文', '志', '勇', '刚', '鹏', '涛', '磊', '雄', '峰', '超']
    
    # 女性名字
    female_names = ['丽', '红', '燕', '霞', '玲', '娜', '美', '静', '雅', '芳', '兰', '梅', '花', '萍', '琳', '莉', '婷', '欣']
    
    names = []
    genders = []
    
    np.random.seed(42)  # 设置随机种子以确保可重现性
    
    for _ in range(n_samples):
        surname = np.random.choice(surnames)
        
        if np.random.rand() < 0.5:  # 50% 概率生成男性姓名
            given_name = np.random.choice(male_names)
            gender = 'male'
        else:  # 50% 概率生成女性姓名
            given_name = np.random.choice(female_names)
            gender = 'female'
        
        # 可能添加第二个字
        if np.random.rand() < 0.3:  # 30% 概率是三字姓名
            if gender == 'male':
                second_char = np.random.choice(male_names)
            else:
                second_char = np.random.choice(female_names)
            full_name = surname + given_name + second_char
        else:
            full_name = surname + given_name
        
        names.append(full_name)
        genders.append(gender)
    
    return names, genders


def train_and_evaluate_model(names: List[str], genders: List[str], 
                           test_size: float = 0.2, model_type: str = 'gaussian') -> Dict[str, any]:
    """
    训练和评估性别预测模型的便捷函数
    
    Args:
        names: 姓名列表
        genders: 性别标签列表
        test_size: 测试集比例
        model_type: 模型类型
        
    Returns:
        评估结果字典
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        names, genders, test_size=test_size, random_state=42, stratify=genders
    )
    
    # 创建和训练模型
    predictor = ChineseNameGenderPredictor(model_type=model_type)
    predictor.fit(X_train, y_train)
    
    # 评估模型
    evaluation_results = predictor.evaluate(X_test, y_test)
    
    # 添加模型实例到结果中
    evaluation_results['model'] = predictor
    evaluation_results['train_size'] = len(X_train)
    evaluation_results['test_size'] = len(X_test)
    
    return evaluation_results


if __name__ == "__main__":
    print("中文姓名性别预测测试")
    
    # 创建示例数据
    print("创建示例数据...")
    names, genders = create_sample_data(1000)
    
    print(f"数据样本: {len(names)}")
    print(f"前10个样本:")
    for i in range(10):
        print(f"  {names[i]} - {genders[i]}")
    
    # 训练和评估模型
    print("\n训练和评估模型...")
    results = train_and_evaluate_model(names, genders)
    
    print(f"准确率: {results['accuracy']:.4f}")
    print(f"训练集大小: {results['train_size']}")
    print(f"测试集大小: {results['test_size']}")
    
    # 测试单个预测
    print("\n单个预测测试:")
    model = results['model']
    test_names = ["张伟", "李娜", "王小明", "刘美丽", "陈建国", "赵静雅"]
    
    for name in test_names:
        predicted_gender, probability = model.predict_single(name)
        print(f"姓名: {name}, 预测性别: {predicted_gender}, 概率: {probability:.4f}")
    
    # 获取特征重要性
    print("\n特征重要性:")
    importance = model.get_feature_importance(names[:100])
    for feature, score in list(importance.items())[:10]:
        print(f"  {feature}: {score:.4f}")