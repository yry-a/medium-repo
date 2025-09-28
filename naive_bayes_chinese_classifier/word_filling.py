"""
选词填空模块
Word Filling Module

基于朴素贝叶斯分类器实现选词填空功能，
通过上下文分析选择最合适的词汇填入空白处。
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import logging
import pickle
import os
import jieba
from collections import Counter

from data_preprocessing import ContextPreprocessor, ChineseTextPreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WordFillingClassifier:
    """选词填空分类器"""
    
    def __init__(self, window_size: int = 5, max_features: int = 10000, 
                 ngram_range: Tuple[int, int] = (1, 2)):
        """
        初始化选词填空分类器
        
        Args:
            window_size: 上下文窗口大小
            max_features: 最大特征数
            ngram_range: N-gram范围
        """
        self.window_size = window_size
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        self.context_preprocessor = ContextPreprocessor(window_size)
        self.text_preprocessor = ChineseTextPreprocessor()
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,
            lowercase=False,
            token_pattern=r'(?u)\b\w+\b'
        )
        
        self.model = MultinomialNB()
        self.word_vocab = set()  # 词汇表
        self.is_fitted = False
    
    def _extract_context_features(self, contexts: List[str]) -> np.ndarray:
        """
        提取上下文特征
        
        Args:
            contexts: 上下文文本列表
            
        Returns:
            特征矩阵
        """
        # 预处理上下文
        processed_contexts = []
        for context in contexts:
            words = self.text_preprocessor.segment_text(context)
            processed_context = ' '.join(words)
            processed_contexts.append(processed_context)
        
        # 向量化
        if not self.is_fitted:
            features = self.vectorizer.fit_transform(processed_contexts)
        else:
            features = self.vectorizer.transform(processed_contexts)
        
        return features
    
    def prepare_training_data(self, texts: List[str], 
                            target_words_per_text: List[List[str]]) -> Tuple[List[str], List[str]]:
        """
        准备训练数据，从完整文本中创建上下文-目标词对
        
        Args:
            texts: 完整文本列表
            target_words_per_text: 每个文本对应的目标词列表
            
        Returns:
            (上下文列表, 目标词列表)
        """
        contexts = []
        target_words = []
        
        for text, words in zip(texts, target_words_per_text):
            # 分词
            text_words = self.text_preprocessor.segment_text(text)
            
            # 为每个目标词创建训练样本
            for target_word in words:
                if target_word in text_words:
                    # 找到目标词的位置
                    word_positions = [i for i, w in enumerate(text_words) if w == target_word]
                    
                    for pos in word_positions:
                        # 创建带空白的文本
                        text_with_blank = text_words.copy()
                        text_with_blank[pos] = '___'
                        text_with_blank_str = ''.join(text_with_blank)
                        
                        # 提取上下文特征
                        context_features = self.context_preprocessor.extract_context_features(
                            text_with_blank_str, pos
                        )
                        
                        contexts.append(context_features['full_context_text'])
                        target_words.append(target_word)
                        
                        # 添加到词汇表
                        self.word_vocab.add(target_word)
        
        return contexts, target_words
    
    def fit(self, texts: List[str], target_words_per_text: List[List[str]]) -> 'WordFillingClassifier':
        """
        训练选词填空模型
        
        Args:
            texts: 训练文本列表
            target_words_per_text: 每个文本对应的目标词列表
            
        Returns:
            self: 返回自身以支持链式调用
        """
        try:
            logger.info(f"开始训练选词填空模型，文本数量: {len(texts)}")
            
            # 准备训练数据
            contexts, target_words = self.prepare_training_data(texts, target_words_per_text)
            
            logger.info(f"生成训练样本: {len(contexts)}, 词汇表大小: {len(self.word_vocab)}")
            
            # 提取特征
            X = self._extract_context_features(contexts)
            
            # 训练模型
            self.model.fit(X, target_words)
            self.is_fitted = True
            
            logger.info("选词填空模型训练完成")
            return self
            
        except Exception as e:
            logger.error(f"模型训练失败: {str(e)}")
            raise
    
    def predict_word(self, context_text: str, candidate_words: List[str], 
                    top_k: int = 5) -> List[Tuple[str, float]]:
        """
        预测最适合填入空白的词
        
        Args:
            context_text: 包含空白(___)的上下文
            candidate_words: 候选词列表
            top_k: 返回前k个预测结果
            
        Returns:
            (词, 概率)列表，按概率降序排列
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            # 提取上下文特征
            context_features = self.context_preprocessor.extract_context_features(context_text, 0)
            context = context_features['full_context_text']
            
            # 向量化上下文
            X = self._extract_context_features([context])
            
            # 获取所有类别的概率
            all_probabilities = self.model.predict_proba(X)[0]
            all_classes = self.model.classes_
            
            # 创建类别-概率字典
            class_prob_dict = {}
            for i, class_name in enumerate(all_classes):
                class_prob_dict[class_name] = all_probabilities[i]
            
            # 计算候选词的概率
            candidate_probabilities = []
            for word in candidate_words:
                if word in class_prob_dict:
                    prob = class_prob_dict[word]
                else:
                    # 如果词不在训练集中，给一个很小的概率
                    prob = 1e-10
                
                candidate_probabilities.append((word, float(prob)))
            
            # 按概率排序
            candidate_probabilities.sort(key=lambda x: x[1], reverse=True)
            
            return candidate_probabilities[:top_k]
            
        except Exception as e:
            logger.error(f"词预测失败: {str(e)}")
            raise
    
    def fill_blank(self, context_text: str, candidate_words: List[str]) -> Tuple[str, float]:
        """
        选择最佳词汇填入空白
        
        Args:
            context_text: 包含空白的上下文
            candidate_words: 候选词列表
            
        Returns:
            (最佳词汇, 预测概率)
        """
        predictions = self.predict_word(context_text, candidate_words, top_k=1)
        if predictions:
            return predictions[0]
        else:
            return candidate_words[0] if candidate_words else ("", 0.0)
    
    def evaluate(self, test_contexts: List[str], true_words: List[str], 
                candidate_words_per_context: List[List[str]]) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            test_contexts: 测试上下文列表
            true_words: 真实答案列表
            candidate_words_per_context: 每个上下文的候选词列表
            
        Returns:
            评估指标字典
        """
        if not self.is_fitted:
            raise ValueError("模型未训练，请先调用fit方法")
        
        try:
            predictions = []
            top3_predictions = []
            top5_predictions = []
            
            for context, true_word, candidates in zip(test_contexts, true_words, 
                                                    candidate_words_per_context):
                # 预测
                word_probs = self.predict_word(context, candidates, top_k=5)
                
                if word_probs:
                    # Top-1 预测
                    pred_word = word_probs[0][0]
                    predictions.append(pred_word)
                    
                    # Top-3 和 Top-5 预测
                    top3_words = [w for w, _ in word_probs[:3]]
                    top5_words = [w for w, _ in word_probs[:5]]
                    
                    top3_predictions.append(true_word in top3_words)
                    top5_predictions.append(true_word in top5_words)
                else:
                    predictions.append("")
                    top3_predictions.append(False)
                    top5_predictions.append(False)
            
            # 计算准确率
            top1_accuracy = accuracy_score(true_words, predictions)
            top3_accuracy = np.mean(top3_predictions)
            top5_accuracy = np.mean(top5_predictions)
            
            return {
                'top1_accuracy': top1_accuracy,
                'top3_accuracy': top3_accuracy,
                'top5_accuracy': top5_accuracy,
                'predictions': predictions,
                'true_labels': true_words
            }
            
        except Exception as e:
            logger.error(f"模型评估失败: {str(e)}")
            raise
    
    def get_vocabulary(self) -> List[str]:
        """获取词汇表"""
        return list(self.word_vocab)
    
    def get_context_similarity(self, context1: str, context2: str) -> float:
        """
        计算两个上下文的相似度
        
        Args:
            context1: 第一个上下文
            context2: 第二个上下文
            
        Returns:
            相似度分数
        """
        if not self.is_fitted:
            return 0.0
        
        try:
            # 向量化上下文
            X = self._extract_context_features([context1, context2])
            
            # 计算余弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(X[0:1], X[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {str(e)}")
            return 0.0
    
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
                'vectorizer': self.vectorizer,
                'word_vocab': self.word_vocab,
                'window_size': self.window_size,
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
            
            self.model = model_data['model']
            self.vectorizer = model_data['vectorizer']
            self.word_vocab = model_data['word_vocab']
            self.window_size = model_data['window_size']
            self.max_features = model_data['max_features']
            self.ngram_range = model_data['ngram_range']
            self.is_fitted = model_data['is_fitted']
            
            logger.info(f"模型已从{filepath}加载")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise


class WordFillingDataGenerator:
    """选词填空数据生成器"""
    
    def __init__(self):
        """初始化数据生成器"""
        self.text_preprocessor = ChineseTextPreprocessor()
    
    def create_sample_data(self, n_samples: int = 100) -> Tuple[List[str], List[List[str]]]:
        """
        创建示例选词填空数据
        
        Args:
            n_samples: 样本数量
            
        Returns:
            (文本列表, 目标词列表)
        """
        sample_texts = [
            "今天天气很好，我决定去公园散步",
            "这本书的内容很有趣，我读得很认真",
            "妈妈在厨房里做饭，香味飘满了整个房子",
            "学生们在教室里认真听老师讲课",
            "春天到了，花园里的花都开了",
            "他每天早上都去跑步锻炼身体",
            "图书馆里很安静，适合学习和阅读",
            "这家餐厅的菜品味道很不错",
            "孩子们在操场上快乐地玩耍",
            "电影院里正在播放一部精彩的电影",
            "医生建议我们要保持健康的生活方式",
            "老师表扬了那些努力学习的学生",
            "商店里有各种各样的商品可以选择",
            "朋友们约好了周末一起去旅游",
            "音乐会上演奏家们的表演非常精彩"
        ]
        
        target_words_lists = [
            ["天气", "公园", "散步"],
            ["书", "内容", "认真"],
            ["妈妈", "厨房", "做饭"],
            ["学生", "教室", "老师"],
            ["春天", "花园", "花"],
            ["跑步", "锻炼", "身体"],
            ["图书馆", "安静", "学习"],
            ["餐厅", "菜品", "味道"],
            ["孩子", "操场", "玩耍"],
            ["电影院", "播放", "电影"],
            ["医生", "健康", "生活"],
            ["老师", "学习", "学生"],
            ["商店", "商品", "选择"],
            ["朋友", "周末", "旅游"],
            ["音乐会", "演奏家", "表演"]
        ]
        
        # 扩展数据到所需数量
        texts = []
        target_words = []
        
        for i in range(n_samples):
            idx = i % len(sample_texts)
            texts.append(sample_texts[idx])
            target_words.append(target_words_lists[idx])
        
        return texts, target_words
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """
        创建测试用例
        
        Returns:
            测试用例列表
        """
        test_cases = [
            {
                'context': '今天___很好，我决定去公园散步',
                'candidates': ['天气', '心情', '运气', '身体'],
                'correct_answer': '天气'
            },
            {
                'context': '这本___的内容很有趣，我读得很认真',
                'candidates': ['书', '杂志', '报纸', '小说'],
                'correct_answer': '书'
            },
            {
                'context': '妈妈在___里做饭，香味飘满了整个房子',
                'candidates': ['厨房', '客厅', '卧室', '书房'],
                'correct_answer': '厨房'
            },
            {
                'context': '学生们在教室里认真听老师___',
                'candidates': ['讲课', '唱歌', '跳舞', '画画'],
                'correct_answer': '讲课'
            },
            {
                'context': '春天到了，花园里的___都开了',
                'candidates': ['花', '树', '草', '叶'],
                'correct_answer': '花'
            },
            {
                'context': '他每天早上都去___锻炼身体',
                'candidates': ['跑步', '游泳', '打球', '骑车'],
                'correct_answer': '跑步'
            },
            {
                'context': '___里很安静，适合学习和阅读',
                'candidates': ['图书馆', '咖啡厅', '公园', '商场'],
                'correct_answer': '图书馆'
            },
            {
                'context': '这家餐厅的菜品___很不错',
                'candidates': ['味道', '价格', '分量', '样式'],
                'correct_answer': '味道'
            },
            {
                'context': '孩子们在___上快乐地玩耍',
                'candidates': ['操场', '沙滩', '草地', '公园'],
                'correct_answer': '操场'
            },
            {
                'context': '___院里正在播放一部精彩的电影',
                'candidates': ['电影', '剧', '音乐', '体育'],
                'correct_answer': '电影'
            }
        ]
        
        return test_cases


def train_and_evaluate_word_filling(texts: List[str], target_words_per_text: List[List[str]], 
                                  test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    训练和评估选词填空模型的便捷函数
    
    Args:
        texts: 训练文本列表
        target_words_per_text: 每个文本的目标词列表
        test_cases: 测试用例
        
    Returns:
        评估结果字典
    """
    # 创建和训练模型
    classifier = WordFillingClassifier()
    classifier.fit(texts, target_words_per_text)
    
    # 准备测试数据
    test_contexts = [case['context'] for case in test_cases]
    true_words = [case['correct_answer'] for case in test_cases]
    candidate_words_per_context = [case['candidates'] for case in test_cases]
    
    # 评估模型
    evaluation_results = classifier.evaluate(test_contexts, true_words, candidate_words_per_context)
    
    # 添加模型和其他信息
    evaluation_results['model'] = classifier
    evaluation_results['vocabulary_size'] = len(classifier.get_vocabulary())
    evaluation_results['test_cases'] = test_cases
    
    return evaluation_results


if __name__ == "__main__":
    print("选词填空测试")
    
    # 创建数据生成器
    data_generator = WordFillingDataGenerator()
    
    # 生成训练数据
    print("生成训练数据...")
    texts, target_words = data_generator.create_sample_data(300)
    
    print(f"训练文本数量: {len(texts)}")
    print("前5个训练样本:")
    for i in range(5):
        print(f"  文本: {texts[i]}")
        print(f"  目标词: {target_words[i]}")
        print()
    
    # 创建测试用例
    test_cases = data_generator.create_test_cases()
    
    print(f"测试用例数量: {len(test_cases)}")
    print("前3个测试用例:")
    for i in range(3):
        case = test_cases[i]
        print(f"  上下文: {case['context']}")
        print(f"  候选词: {case['candidates']}")
        print(f"  正确答案: {case['correct_answer']}")
        print()
    
    # 训练和评估模型
    print("训练和评估模型...")
    results = train_and_evaluate_word_filling(texts, target_words, test_cases)
    
    print(f"Top-1 准确率: {results['top1_accuracy']:.4f}")
    print(f"Top-3 准确率: {results['top3_accuracy']:.4f}")
    print(f"Top-5 准确率: {results['top5_accuracy']:.4f}")
    print(f"词汇表大小: {results['vocabulary_size']}")
    
    # 详细测试结果
    print("\n详细测试结果:")
    model = results['model']
    
    for i, case in enumerate(test_cases[:5]):
        context = case['context']
        candidates = case['candidates']
        correct = case['correct_answer']
        
        predictions = model.predict_word(context, candidates, top_k=3)
        
        print(f"测试 {i+1}:")
        print(f"  上下文: {context}")
        print(f"  正确答案: {correct}")
        print(f"  预测结果:")
        for j, (word, prob) in enumerate(predictions):
            print(f"    {j+1}. {word} (概率: {prob:.4f})")
        print()
    
    # 测试上下文相似度
    print("上下文相似度测试:")
    context1 = "今天天气很好，我想去___"
    context2 = "今天阳光明媚，我想去___"
    context3 = "昨天下雨了，我不想出___"
    
    sim12 = model.get_context_similarity(context1, context2)
    sim13 = model.get_context_similarity(context1, context3)
    
    print(f"'{context1}' 与 '{context2}' 的相似度: {sim12:.4f}")
    print(f"'{context1}' 与 '{context3}' 的相似度: {sim13:.4f}")