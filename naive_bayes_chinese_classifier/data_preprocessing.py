"""
数据预处理工具模块
Data Preprocessing Utilities Module

提供中文文本预处理、特征提取和数据清洗功能，
专门针对中文姓名、文本分类和选词填空任务优化。
"""

import re
import jieba
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 中文停用词列表
CHINESE_STOPWORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
    '会', '着', '没有', '看', '好', '自己', '这', '那', '它', '他', '她', '们', '个', '来', '对', '还', '而', '能', '下',
    '以', '及', '其', '或', '等', '被', '将', '已', '与', '可以', '可', '但', '却', '只', '把', '从', '向', '用', '比',
    '由', '为', '因为', '所以', '如果', '虽然', '但是', '然而', '因此', '于是', '然后', '接着', '最后', '总之'
}


class ChineseTextPreprocessor:
    """中文文本预处理器"""
    
    def __init__(self, use_stopwords: bool = True, custom_stopwords: Optional[Set[str]] = None):
        """
        初始化预处理器
        
        Args:
            use_stopwords: 是否使用停用词过滤
            custom_stopwords: 自定义停用词集合
        """
        self.use_stopwords = use_stopwords
        self.stopwords = CHINESE_STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            return ""
        
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 去除邮箱
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # 去除多余的空格和换行符
        text = re.sub(r'\s+', ' ', text)
        
        # 去除特殊字符（保留中文、英文、数字）
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        
        return text.strip()
    
    def segment_text(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 使用jieba进行分词
        words = jieba.lcut(cleaned_text)
        
        # 过滤停用词和单字符词
        if self.use_stopwords:
            words = [word for word in words if word not in self.stopwords and len(word) > 1]
        
        return words
    
    def extract_features_from_text(self, texts: List[str]) -> Dict[str, any]:
        """
        从文本中提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            包含各种特征的字典
        """
        features = {
            'text_lengths': [],
            'word_counts': [],
            'unique_word_counts': [],
            'avg_word_length': [],
            'segmented_texts': []
        }
        
        for text in texts:
            # 分词
            words = self.segment_text(text)
            features['segmented_texts'].append(' '.join(words))
            
            # 文本长度
            features['text_lengths'].append(len(text))
            
            # 词数量
            features['word_counts'].append(len(words))
            
            # 唯一词数量
            features['unique_word_counts'].append(len(set(words)))
            
            # 平均词长
            if words:
                avg_len = sum(len(word) for word in words) / len(words)
            else:
                avg_len = 0
            features['avg_word_length'].append(avg_len)
        
        return features


class ChineseNamePreprocessor:
    """中文姓名预处理器"""
    
    def __init__(self):
        """初始化姓名预处理器"""
        # 常见姓氏
        self.common_surnames = {
            '李', '王', '张', '刘', '陈', '杨', '赵', '黄', '周', '吴', '徐', '孙', '胡', '朱', '高', '林', '何', '郭', '马',
            '罗', '梁', '宋', '郑', '谢', '韩', '唐', '冯', '于', '董', '萧', '程', '曹', '袁', '邓', '许', '傅', '沈',
            '曾', '彭', '吕', '苏', '卢', '蒋', '蔡', '贾', '丁', '魏', '薛', '叶', '阎', '余', '潘', '杜', '戴', '夏',
            '钟', '汪', '田', '任', '姜', '范', '方', '石', '姚', '谭', '廖', '邹', '熊', '金', '陆', '郝', '孔', '白'
        }
        
        # 常见名字字符（按性别倾向分类）
        self.male_name_chars = {
            '强', '伟', '军', '杰', '华', '明', '建', '国', '文', '志', '勇', '刚', '鹏', '涛', '磊', '雄', '峰', '超',
            '龙', '虎', '豪', '威', '凯', '健', '俊', '浩', '阳', '斌', '博', '宇', '东', '南', '北', '西', '昊', '轩'
        }
        
        self.female_name_chars = {
            '丽', '红', '燕', '霞', '玲', '娜', '美', '静', '雅', '芳', '兰', '梅', '花', '萍', '琳', '莉', '婷', '欣',
            '怡', '慧', '敏', '洁', '雯', '琴', '秀', '娟', '瑶', '蕾', '薇', '菲', '妍', '颖', '晶', '珍', '艳', '凤'
        }
    
    def extract_name_features(self, name: str) -> Dict[str, any]:
        """
        从姓名中提取特征
        
        Args:
            name: 姓名
            
        Returns:
            包含姓名特征的字典
        """
        if not isinstance(name, str) or len(name) < 2:
            return self._get_empty_features()
        
        features = {}
        
        # 姓名长度
        features['name_length'] = len(name)
        
        # 姓氏特征
        surname = name[0]
        features['is_common_surname'] = 1 if surname in self.common_surnames else 0
        features['surname'] = surname
        
        # 名字部分
        given_name = name[1:]
        features['given_name_length'] = len(given_name)
        
        # 性别倾向特征
        male_char_count = sum(1 for char in given_name if char in self.male_name_chars)
        female_char_count = sum(1 for char in given_name if char in self.female_name_chars)
        
        features['male_char_count'] = male_char_count
        features['female_char_count'] = female_char_count
        features['male_char_ratio'] = male_char_count / len(given_name) if given_name else 0
        features['female_char_ratio'] = female_char_count / len(given_name) if given_name else 0
        
        # 字符特征
        features['all_chars'] = list(name)
        features['given_name_chars'] = list(given_name)
        
        # 最后一个字的特征（通常对性别判断很重要）
        if given_name:
            last_char = given_name[-1]
            features['last_char'] = last_char
            features['last_char_is_male'] = 1 if last_char in self.male_name_chars else 0
            features['last_char_is_female'] = 1 if last_char in self.female_name_chars else 0
        
        return features
    
    def _get_empty_features(self) -> Dict[str, any]:
        """返回空特征字典"""
        return {
            'name_length': 0,
            'is_common_surname': 0,
            'surname': '',
            'given_name_length': 0,
            'male_char_count': 0,
            'female_char_count': 0,
            'male_char_ratio': 0,
            'female_char_ratio': 0,
            'all_chars': [],
            'given_name_chars': [],
            'last_char': '',
            'last_char_is_male': 0,
            'last_char_is_female': 0
        }
    
    def create_feature_vector(self, names: List[str]) -> pd.DataFrame:
        """
        为姓名列表创建特征向量
        
        Args:
            names: 姓名列表
            
        Returns:
            特征DataFrame
        """
        features_list = []
        
        for name in names:
            features = self.extract_name_features(name)
            features_list.append(features)
        
        # 转换为DataFrame
        df = pd.DataFrame(features_list)
        
        # 处理分类特征（姓氏、字符）
        surname_dummies = pd.get_dummies(df['surname'], prefix='surname')
        last_char_dummies = pd.get_dummies(df['last_char'], prefix='last_char')
        
        # 合并特征
        numeric_features = df.select_dtypes(include=[np.number])
        result_df = pd.concat([numeric_features, surname_dummies, last_char_dummies], axis=1)
        
        return result_df


class ContextPreprocessor:
    """上下文预处理器，用于选词填空任务"""
    
    def __init__(self, window_size: int = 5):
        """
        初始化上下文预处理器
        
        Args:
            window_size: 上下文窗口大小
        """
        self.window_size = window_size
        self.text_preprocessor = ChineseTextPreprocessor()
    
    def extract_context_features(self, text: str, blank_position: int) -> Dict[str, any]:
        """
        提取空白位置周围的上下文特征
        
        Args:
            text: 包含空白的文本
            blank_position: 空白位置索引
            
        Returns:
            上下文特征字典
        """
        # 分词
        words = self.text_preprocessor.segment_text(text.replace('___', ' ___'))
        
        # 找到空白位置
        try:
            blank_index = words.index('___')
        except ValueError:
            blank_index = blank_position
        
        features = {}
        
        # 左上下文
        left_start = max(0, blank_index - self.window_size)
        left_context = words[left_start:blank_index]
        features['left_context'] = left_context
        features['left_context_text'] = ' '.join(left_context)
        
        # 右上下文
        right_end = min(len(words), blank_index + self.window_size + 1)
        right_context = words[blank_index + 1:right_end]
        features['right_context'] = right_context
        features['right_context_text'] = ' '.join(right_context)
        
        # 完整上下文
        features['full_context'] = left_context + right_context
        features['full_context_text'] = ' '.join(features['full_context'])
        
        # 位置特征
        features['blank_position'] = blank_index
        features['relative_position'] = blank_index / len(words) if words else 0
        
        return features
    
    def create_context_pairs(self, texts: List[str], target_words: List[str]) -> List[Tuple[str, str]]:
        """
        创建上下文-目标词对
        
        Args:
            texts: 包含空白的文本列表
            target_words: 目标词列表
            
        Returns:
            (上下文, 目标词)对列表
        """
        pairs = []
        
        for text, target_word in zip(texts, target_words):
            # 提取上下文特征
            features = self.extract_context_features(text, 0)
            context = features['full_context_text']
            pairs.append((context, target_word))
        
        return pairs


def load_and_preprocess_data(file_path: str, text_column: str, 
                           label_column: str, preprocessor_type: str = 'text') -> Tuple[List[str], List[str]]:
    """
    加载和预处理数据的便捷函数
    
    Args:
        file_path: 数据文件路径
        text_column: 文本列名
        label_column: 标签列名
        preprocessor_type: 预处理器类型 ('text', 'name', 'context')
        
    Returns:
        (处理后的文本列表, 标签列表)
    """
    try:
        # 读取数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("不支持的文件格式")
        
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].astype(str).tolist()
        
        # 选择预处理器
        if preprocessor_type == 'text':
            preprocessor = ChineseTextPreprocessor()
            processed_texts = []
            for text in texts:
                words = preprocessor.segment_text(text)
                processed_texts.append(' '.join(words))
            return processed_texts, labels
        
        elif preprocessor_type == 'name':
            # 姓名不需要分词处理
            return texts, labels
        
        elif preprocessor_type == 'context':
            preprocessor = ContextPreprocessor()
            # 上下文处理需要特殊逻辑
            return texts, labels
        
        else:
            raise ValueError(f"不支持的预处理器类型: {preprocessor_type}")
    
    except Exception as e:
        logger.error(f"数据加载和预处理失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 测试文本预处理器
    print("测试中文文本预处理器")
    text_preprocessor = ChineseTextPreprocessor()
    
    test_text = "这是一个很好的产品，我非常喜欢！！！"
    cleaned = text_preprocessor.clean_text(test_text)
    segmented = text_preprocessor.segment_text(test_text)
    
    print(f"原文: {test_text}")
    print(f"清洗后: {cleaned}")
    print(f"分词结果: {segmented}")
    
    # 测试姓名预处理器
    print("\n测试中文姓名预处理器")
    name_preprocessor = ChineseNamePreprocessor()
    
    test_names = ["张伟", "李娜", "王小明", "刘美丽"]
    for name in test_names:
        features = name_preprocessor.extract_name_features(name)
        print(f"姓名: {name}, 特征: {features}")
    
    # 测试上下文预处理器
    print("\n测试上下文预处理器")
    context_preprocessor = ContextPreprocessor()
    
    test_context = "今天天气很好，我想去___公园散步"
    features = context_preprocessor.extract_context_features(test_context, 0)
    print(f"上下文: {test_context}")
    print(f"上下文特征: {features}")