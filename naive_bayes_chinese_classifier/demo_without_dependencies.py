#!/usr/bin/env python3
"""
朴素贝叶斯分类器演示脚本（无外部依赖版本）
Naive Bayes Classifier Demo Script (No External Dependencies)

这个脚本演示了朴素贝叶斯分类器的核心原理和实现，
不依赖外部库，使用纯Python实现基本功能。
"""

import math
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Any
import os


class SimpleChineseTokenizer:
    """简单的中文分词器（基于规则）"""
    
    def __init__(self):
        # 常见的中文停用词
        self.stopwords = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '上', '也', '很', '到', '说', '要', '去',
            '会', '着', '没有', '看', '好', '自己', '这', '那', '它', '他', '她', '们', '个', '来', '对', '还', '而'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """简单的中文分词（基于字符）"""
        # 清理文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        
        # 提取中文字符和词汇
        tokens = []
        i = 0
        while i < len(text):
            if '\u4e00' <= text[i] <= '\u9fa5':  # 中文字符
                # 尝试构建词汇（最多3个字符）
                word = text[i]
                if i + 1 < len(text) and '\u4e00' <= text[i + 1] <= '\u9fa5':
                    word += text[i + 1]
                    if i + 2 < len(text) and '\u4e00' <= text[i + 2] <= '\u9fa5':
                        word += text[i + 2]
                        i += 3
                    else:
                        i += 2
                else:
                    i += 1
                
                if word not in self.stopwords and len(word) >= 1:
                    tokens.append(word)
            else:
                i += 1
        
        return tokens


class SimpleNaiveBayesClassifier:
    """简单的朴素贝叶斯分类器实现"""
    
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.total_samples = 0
        self.classes = set()
        self.tokenizer = SimpleChineseTokenizer()
    
    def _extract_features(self, text: str) -> Dict[str, int]:
        """提取文本特征"""
        tokens = self.tokenizer.tokenize(text)
        return Counter(tokens)
    
    def fit(self, texts: List[str], labels: List[str]):
        """训练分类器"""
        print(f"Training classifier with {len(texts)} samples...")
        
        self.total_samples = len(texts)
        self.classes = set(labels)
        
        for text, label in zip(texts, labels):
            self.class_counts[label] += 1
            features = self._extract_features(text)
            
            for feature, count in features.items():
                self.feature_counts[label][feature] += count
                self.vocabulary.add(feature)
        
        print(f"Training completed. Vocabulary size: {len(self.vocabulary)}, Classes: {len(self.classes)}")
        return self
    
    def _calculate_class_probability(self, label: str) -> float:
        """计算类别先验概率"""
        return self.class_counts[label] / self.total_samples
    
    def _calculate_feature_probability(self, feature: str, label: str) -> float:
        """计算特征在给定类别下的概率（使用拉普拉斯平滑）"""
        feature_count = self.feature_counts[label][feature]
        total_features = sum(self.feature_counts[label].values())
        vocab_size = len(self.vocabulary)
        
        # 拉普拉斯平滑
        return (feature_count + 1) / (total_features + vocab_size)
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """预测类别概率"""
        features = self._extract_features(text)
        class_probabilities = {}
        
        for label in self.classes:
            # 计算类别先验概率
            prob = math.log(self._calculate_class_probability(label))
            
            # 计算特征概率
            for feature, count in features.items():
                feature_prob = self._calculate_feature_probability(feature, label)
                prob += count * math.log(feature_prob)
            
            class_probabilities[label] = prob
        
        # 转换为概率（从对数概率）
        max_prob = max(class_probabilities.values())
        for label in class_probabilities:
            class_probabilities[label] = math.exp(class_probabilities[label] - max_prob)
        
        # 归一化
        total = sum(class_probabilities.values())
        for label in class_probabilities:
            class_probabilities[label] /= total
        
        return class_probabilities
    
    def predict(self, text: str) -> str:
        """预测类别"""
        probabilities = self.predict_proba(text)
        return max(probabilities, key=probabilities.get)


class SimpleNameGenderClassifier:
    """简单的中文姓名性别分类器"""
    
    def __init__(self):
        self.classifier = SimpleNaiveBayesClassifier()
        
        # 预定义的性别特征字符
        self.male_chars = {
            '强', '伟', '军', '杰', '华', '明', '建', '国', '文', '志', '勇', '刚', '鹏', '涛', '磊', '雄', '峰', '超',
            '龙', '虎', '豪', '威', '凯', '健', '俊', '浩', '阳', '斌', '博', '宇', '东', '南', '北', '西', '昊', '轩'
        }
        
        self.female_chars = {
            '丽', '红', '燕', '霞', '玲', '娜', '美', '静', '雅', '芳', '兰', '梅', '花', '萍', '琳', '莉', '婷', '欣',
            '怡', '慧', '敏', '洁', '雯', '琴', '秀', '娟', '瑶', '蕾', '薇', '菲', '妍', '颖', '晶', '珍', '艳', '凤'
        }
    
    def _extract_name_features(self, name: str) -> str:
        """提取姓名特征"""
        if len(name) < 2:
            return name
        
        features = []
        
        # 姓名长度特征
        features.append(f"length_{len(name)}")
        
        # 名字部分（去掉姓氏）
        given_name = name[1:]
        
        # 性别倾向字符特征
        male_count = sum(1 for char in given_name if char in self.male_chars)
        female_count = sum(1 for char in given_name if char in self.female_chars)
        
        features.append(f"male_chars_{male_count}")
        features.append(f"female_chars_{female_count}")
        
        # 最后一个字特征
        if given_name:
            last_char = given_name[-1]
            features.append(f"last_char_{last_char}")
            
            if last_char in self.male_chars:
                features.append("last_char_type_male")
            elif last_char in self.female_chars:
                features.append("last_char_type_female")
            else:
                features.append("last_char_type_neutral")
        
        # 每个字符特征
        for char in given_name:
            features.append(f"char_{char}")
        
        return " ".join(features)
    
    def fit(self, names: List[str], genders: List[str]):
        """训练性别分类器"""
        feature_texts = [self._extract_name_features(name) for name in names]
        self.classifier.fit(feature_texts, genders)
        return self
    
    def predict(self, name: str) -> str:
        """预测姓名性别"""
        feature_text = self._extract_name_features(name)
        return self.classifier.predict(feature_text)
    
    def predict_proba(self, name: str) -> Dict[str, float]:
        """预测姓名性别概率"""
        feature_text = self._extract_name_features(name)
        return self.classifier.predict_proba(feature_text)


class SimpleWordFillingClassifier:
    """简单的选词填空分类器"""
    
    def __init__(self, context_window: int = 3):
        self.context_window = context_window
        self.word_context_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        self.tokenizer = SimpleChineseTokenizer()
    
    def _extract_context(self, words: List[str], position: int) -> str:
        """提取上下文特征"""
        start = max(0, position - self.context_window)
        end = min(len(words), position + self.context_window + 1)
        
        context_words = []
        for i in range(start, end):
            if i != position:
                context_words.append(words[i])
        
        return " ".join(context_words)
    
    def fit(self, texts: List[str], target_words_lists: List[List[str]]):
        """训练选词填空模型"""
        print(f"Training word filling classifier with {len(texts)} texts...")
        
        for text, target_words in zip(texts, target_words_lists):
            words = self.tokenizer.tokenize(text)
            
            for target_word in target_words:
                # 找到目标词在文本中的位置
                for i, word in enumerate(words):
                    if word == target_word:
                        context = self._extract_context(words, i)
                        self.word_context_counts[target_word][context] += 1
                        self.context_counts[context] += 1
                        self.vocabulary.add(target_word)
        
        print(f"Training completed. Vocabulary size: {len(self.vocabulary)}")
        return self
    
    def predict_word(self, context_text: str, candidate_words: List[str]) -> List[Tuple[str, float]]:
        """预测最适合的词"""
        context = " ".join(self.tokenizer.tokenize(context_text.replace('___', '')))
        
        word_scores = []
        for word in candidate_words:
            if word in self.vocabulary:
                # 计算词在此上下文中的概率
                word_context_count = self.word_context_counts[word][context]
                context_count = self.context_counts[context]
                
                if context_count > 0:
                    score = word_context_count / context_count
                else:
                    score = 1e-10  # 很小的概率
            else:
                score = 1e-10
            
            word_scores.append((word, score))
        
        # 按分数排序
        word_scores.sort(key=lambda x: x[1], reverse=True)
        return word_scores


def demo_text_classification():
    """演示文本分类功能"""
    print("\n" + "="*50)
    print("文本分类演示")
    print("="*50)
    
    # 准备训练数据
    texts = [
        "这个产品质量很好我很满意推荐大家购买",
        "服务态度非常棒客服很耐心解答问题",
        "物流很快包装也很精美没有损坏",
        "产品质量很差不建议购买浪费钱",
        "服务态度恶劣客服不耐烦很不满意",
        "物流太慢了包装破损商品有问题",
        "产品还可以吧没什么特别的感觉",
        "服务一般般中规中矩没有惊喜",
        "物流正常包装普通符合预期"
    ]
    
    labels = [
        "positive", "positive", "positive",
        "negative", "negative", "negative",
        "neutral", "neutral", "neutral"
    ]
    
    # 训练分类器
    classifier = SimpleNaiveBayesClassifier()
    classifier.fit(texts, labels)
    
    # 测试预测
    test_texts = [
        "这个东西真的很棒我非常喜欢",
        "质量太差了完全不值这个价格",
        "还行吧没什么特别的"
    ]
    
    print("\n测试结果:")
    for text in test_texts:
        prediction = classifier.predict(text)
        probabilities = classifier.predict_proba(text)
        
        print(f"\n文本: {text}")
        print(f"预测类别: {prediction}")
        print("类别概率:")
        for label, prob in probabilities.items():
            print(f"  {label}: {prob:.4f}")


def demo_name_gender_prediction():
    """演示姓名性别预测功能"""
    print("\n" + "="*50)
    print("中文姓名性别预测演示")
    print("="*50)
    
    # 准备训练数据
    names = [
        "张伟", "李强", "王建国", "刘志明", "陈浩", "杨超", "赵峰", "黄刚",
        "李娜", "王美丽", "张静", "刘雅", "陈丽", "杨红", "赵燕", "黄琳"
    ]
    
    genders = [
        "male", "male", "male", "male", "male", "male", "male", "male",
        "female", "female", "female", "female", "female", "female", "female", "female"
    ]
    
    # 训练分类器
    classifier = SimpleNameGenderClassifier()
    classifier.fit(names, genders)
    
    # 测试预测
    test_names = ["张强", "李美", "王建华", "刘静雅", "陈威", "杨婷"]
    
    print("\n测试结果:")
    for name in test_names:
        prediction = classifier.predict(name)
        probabilities = classifier.predict_proba(name)
        
        print(f"\n姓名: {name}")
        print(f"预测性别: {prediction}")
        print("性别概率:")
        for gender, prob in probabilities.items():
            print(f"  {gender}: {prob:.4f}")


def demo_word_filling():
    """演示选词填空功能"""
    print("\n" + "="*50)
    print("选词填空演示")
    print("="*50)
    
    # 准备训练数据
    texts = [
        "今天天气很好我决定去公园散步",
        "这本书的内容很有趣我读得很认真",
        "妈妈在厨房里做饭香味飘满了整个房子",
        "学生们在教室里认真听老师讲课",
        "春天到了花园里的花都开了"
    ]
    
    target_words_lists = [
        ["天气", "公园", "散步"],
        ["书", "内容", "认真"],
        ["妈妈", "厨房", "做饭"],
        ["学生", "教室", "老师", "讲课"],
        ["春天", "花园", "花"]
    ]
    
    # 训练分类器
    classifier = SimpleWordFillingClassifier()
    classifier.fit(texts, target_words_lists)
    
    # 测试用例
    test_cases = [
        {
            'context': '今天___很好我决定去公园散步',
            'candidates': ['天气', '心情', '运气', '阳光'],
        },
        {
            'context': '这本___的内容很有趣我读得很认真',
            'candidates': ['书', '杂志', '报纸', '小说'],
        },
        {
            'context': '妈妈在___里做饭香味飘满了整个房子',
            'candidates': ['厨房', '客厅', '卧室', '书房'],
        }
    ]
    
    print("\n测试结果:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}:")
        print(f"上下文: {case['context']}")
        print(f"候选词: {case['candidates']}")
        
        predictions = classifier.predict_word(case['context'], case['candidates'])
        
        print("预测结果:")
        for j, (word, score) in enumerate(predictions[:3], 1):
            print(f"  {j}. {word} (分数: {score:.4f})")


def main():
    """主函数"""
    print("朴素贝叶斯分类器演示程序")
    print("基于纯Python实现，无需外部依赖")
    print("="*50)
    
    try:
        # 演示文本分类
        demo_text_classification()
        
        # 演示姓名性别预测
        demo_name_gender_prediction()
        
        # 演示选词填空
        demo_word_filling()
        
        print("\n" + "="*50)
        print("所有演示完成！")
        print("\n注意：这是简化版本的演示。")
        print("完整功能需要安装 requirements.txt 中的依赖包。")
        print("="*50)
        
    except Exception as e:
        print(f"演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()