#!/usr/bin/env python3
"""
选词填空测试脚本
Word Filling Test Script

展示如何使用选词填空功能进行中文文本的空白填充
"""

import sys
import os

# 添加父目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from word_filling import WordFillingClassifier, WordFillingDataGenerator
from data_preprocessing import ChineseTextPreprocessor
import pandas as pd


def test_basic_word_filling():
    """测试基本选词填空功能"""
    print("=== 基本选词填空测试 ===")
    
    # 创建数据生成器
    data_generator = WordFillingDataGenerator()
    
    # 生成训练数据
    print("生成训练数据...")
    texts, target_words = data_generator.create_sample_data(200)
    
    # 创建和训练分类器
    classifier = WordFillingClassifier(window_size=3)
    classifier.fit(texts, target_words)
    
    print(f"训练完成，词汇表大小: {len(classifier.get_vocabulary())}")
    
    # 测试用例
    test_cases = [
        {
            'context': '今天___很好，我决定去公园散步',
            'candidates': ['天气', '心情', '运气', '身体', '阳光'],
            'description': '天气相关的选词填空'
        },
        {
            'context': '这本___的内容很有趣，我读得很认真',
            'candidates': ['书', '杂志', '报纸', '小说', '文章'],
            'description': '阅读材料的选词填空'
        },
        {
            'context': '妈妈在___里做饭，香味飘满了整个房子',
            'candidates': ['厨房', '客厅', '卧室', '阳台', '书房'],
            'description': '地点相关的选词填空'
        },
        {
            'context': '学生们在教室里认真听老师___',
            'candidates': ['讲课', '唱歌', '讲故事', '读书', '画画'],
            'description': '动作相关的选词填空'
        }
    ]
    
    print("\n测试结果:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {case['description']}")
        print(f"上下文: {case['context']}")
        print(f"候选词: {case['candidates']}")
        
        # 进行预测
        predictions = classifier.predict_word(case['context'], case['candidates'], top_k=3)
        
        print("预测结果 (Top-3):")
        for j, (word, prob) in enumerate(predictions, 1):
            print(f"  {j}. {word} (概率: {prob:.4f})")
    
    return classifier


def test_context_similarity():
    """测试上下文相似度计算"""
    print("\n=== 上下文相似度测试 ===")
    
    # 使用之前训练的分类器
    data_generator = WordFillingDataGenerator()
    texts, target_words = data_generator.create_sample_data(100)
    
    classifier = WordFillingClassifier()
    classifier.fit(texts, target_words)
    
    # 相似度测试用例
    similarity_tests = [
        {
            'context1': '今天天气很好，我想去___',
            'context2': '今天阳光明媚，我想去___',
            'expected': '高相似度'
        },
        {
            'context1': '今天天气很好，我想去___',
            'context2': '昨天下雨了，我不想___',
            'expected': '低相似度'
        },
        {
            'context1': '这本书很有趣，我喜欢___',
            'context2': '这本小说很精彩，我喜欢___',
            'expected': '中等相似度'
        }
    ]
    
    print("上下文相似度计算:")
    for i, test in enumerate(similarity_tests, 1):
        similarity = classifier.get_context_similarity(test['context1'], test['context2'])
        print(f"\n测试 {i} ({test['expected']}):")
        print(f"  上下文1: {test['context1']}")
        print(f"  上下文2: {test['context2']}")
        print(f"  相似度: {similarity:.4f}")


def test_with_real_data():
    """使用真实数据测试"""
    print("\n=== 真实数据测试 ===")
    
    # 创建更复杂的训练数据
    training_texts = [
        "春天来了，花园里的花都开了",
        "夏天很热，我们去游泳池游泳",
        "秋天到了，树叶变黄了",
        "冬天下雪，孩子们堆雪人",
        "早上起床后，我刷牙洗脸",
        "中午吃完饭，我去公园散步",
        "晚上写完作业，我看电视休息",
        "周末的时候，全家去郊游",
        "图书馆里很安静，适合读书学习",
        "医院里医生给病人看病",
        "学校里老师教学生知识",
        "商店里有各种商品可以购买",
        "餐厅里厨师在厨房做菜",
        "电影院里观众在看电影",
        "公园里有很多人在锻炼身体"
    ]
    
    training_target_words = [
        ["春天", "花园", "花"],
        ["夏天", "游泳池", "游泳"],
        ["秋天", "树叶"],
        ["冬天", "雪", "孩子", "雪人"],
        ["早上", "刷牙", "洗脸"],
        ["中午", "公园", "散步"],
        ["晚上", "作业", "电视"],
        ["周末", "全家", "郊游"],
        ["图书馆", "安静", "读书", "学习"],
        ["医院", "医生", "病人"],
        ["学校", "老师", "学生", "知识"],
        ["商店", "商品", "购买"],
        ["餐厅", "厨师", "厨房"],
        ["电影院", "观众", "电影"],
        ["公园", "锻炼", "身体"]
    ]
    
    # 训练模型
    classifier = WordFillingClassifier(window_size=4, max_features=5000)
    classifier.fit(training_texts, training_target_words)
    
    # 复杂测试用例
    complex_test_cases = [
        {
            'context': '春天来了，___里的花都开了',
            'candidates': ['花园', '公园', '院子', '田野', '山上'],
            'expected': '花园'
        },
        {
            'context': '图书馆里很___，适合读书学习',
            'candidates': ['热闹', '安静', '明亮', '宽敞', '温暖'],
            'expected': '安静'
        },
        {
            'context': '早上起床后，我刷牙___',
            'candidates': ['洗脸', '吃饭', '跑步', '睡觉', '工作'],
            'expected': '洗脸'
        },
        {
            'context': '医院里___给病人看病',
            'candidates': ['护士', '医生', '保安', '清洁工', '家属'],
            'expected': '医生'
        }
    ]
    
    print("复杂测试用例结果:")
    correct_predictions = 0
    
    for i, case in enumerate(complex_test_cases, 1):
        predictions = classifier.predict_word(case['context'], case['candidates'], top_k=3)
        best_prediction = predictions[0][0] if predictions else ""
        
        is_correct = best_prediction == case['expected']
        if is_correct:
            correct_predictions += 1
        
        print(f"\n测试 {i}:")
        print(f"  上下文: {case['context']}")
        print(f"  期望答案: {case['expected']}")
        print(f"  最佳预测: {best_prediction} ({'✓' if is_correct else '✗'})")
        print(f"  Top-3预测:")
        for j, (word, prob) in enumerate(predictions[:3], 1):
            print(f"    {j}. {word} (概率: {prob:.4f})")
    
    accuracy = correct_predictions / len(complex_test_cases)
    print(f"\n整体准确率: {accuracy:.2%} ({correct_predictions}/{len(complex_test_cases)})")


def test_model_persistence():
    """测试模型保存和加载"""
    print("\n=== 模型持久化测试 ===")
    
    # 训练一个模型
    data_generator = WordFillingDataGenerator()
    texts, target_words = data_generator.create_sample_data(100)
    
    classifier = WordFillingClassifier()
    classifier.fit(texts, target_words)
    
    # 测试预测
    test_context = "今天___很好，我决定去公园散步"
    test_candidates = ["天气", "心情", "运气"]
    
    original_predictions = classifier.predict_word(test_context, test_candidates)
    print("原始模型预测:")
    for word, prob in original_predictions:
        print(f"  {word}: {prob:.4f}")
    
    # 保存模型
    model_path = "/tmp/word_filling_model.pkl"
    classifier.save_model(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    # 加载模型
    new_classifier = WordFillingClassifier()
    new_classifier.load_model(model_path)
    print("模型已加载")
    
    # 测试加载后的预测
    loaded_predictions = new_classifier.predict_word(test_context, test_candidates)
    print("\n加载后模型预测:")
    for word, prob in loaded_predictions:
        print(f"  {word}: {prob:.4f}")
    
    # 验证预测结果一致性
    predictions_match = all(
        abs(orig[1] - loaded[1]) < 1e-10 
        for orig, loaded in zip(original_predictions, loaded_predictions)
    )
    
    print(f"\n预测结果一致性: {'✓' if predictions_match else '✗'}")
    
    # 清理临时文件
    if os.path.exists(model_path):
        os.remove(model_path)
        print("临时文件已清理")


def main():
    """主函数"""
    print("选词填空功能测试")
    print("=" * 50)
    
    try:
        # 基本功能测试
        classifier = test_basic_word_filling()
        
        # 上下文相似度测试
        test_context_similarity()
        
        # 真实数据测试
        test_with_real_data()
        
        # 模型持久化测试
        test_model_persistence()
        
        print("\n" + "=" * 50)
        print("所有测试完成！")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()