#!/usr/bin/env python3
"""
IMDB电影评论情感标签自动生成 -
"""

import ollama
import pandas as pd
import yaml
import time
import logging
import sys
import os
import re
from typing import List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IMDBAutoLabeler:
    """IMDB数据集自动标签生成器"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config['model']['name']
        self.temperature = self.config['model']['temperature']
        self.max_tokens = self.config['model']['max_tokens']

        logger.info(f"初始化完成，使用模型: {self.model_name}")

    def load_data(self, filepath: str) -> pd.DataFrame:
        """加载CSV数据"""
        logger.info(f"开始加载数据: {filepath}")

        # 使用多种方法尝试读取CSV
        try:
            # 方法1: 跳过错误行
            df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
        except Exception as e:
            logger.warning(f"方法1失败: {e}，尝试方法2")
            # 方法2: 指定引号
            df = pd.read_csv(filepath, quotechar='"', engine='python')

        logger.info(f"原始数据: {len(df)} 行, 列: {list(df.columns)}")

        # 检查必要列
        text_col = 'review'
        if text_col not in df.columns:
            raise ValueError(f"数据中缺少 '{text_col}' 列")

        # 数据清理
        df = df.dropna(subset=[text_col])
        df[text_col] = df[text_col].astype(str).str.strip()
        df = df[df[text_col] != '']

        # 采样
        sample_size = self.config['data'].get('sample_size')
        if sample_size and sample_size < len(df):
            df = df.sample(sample_size, random_state=42)
            logger.info(f"采样 {len(df)} 条记录")

        return df

    def create_prompt(self, review: str) -> str:
        """创建提示词"""
        prompt = f"""
请分析以下电影评论的情感倾向。
只输出一个单词：POSITIVE 或 NEGATIVE。
不要输出任何解释、说明或其他内容。

电影评论：{review}

情感：
"""
        return prompt.strip()

    def generate_label(self, review: str) -> str:
        """为单条评论生成情感标签"""
        try:
            # 创建提示词
            prompt = self.create_prompt(review)

            # 调用Ollama模型
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )

            # 提取标签
            label = self.extract_label(response['response'])
            return label

        except Exception as e:
            logger.error(f"生成标签失败: {e}")
            return "error"

    def extract_label(self, response_text: str) -> str:
        """从模型响应中提取情感标签"""
        text = response_text.strip().upper()

        if "POSITIVE" in text:
            return "positive"
        elif "NEGATIVE" in text:
            return "negative"
        else:
            # 简单关键词匹配
            text_lower = text.lower()
            if any(word in text_lower for word in ['good', 'great', 'excellent', 'love', 'like']):
                return "positive"
            elif any(word in text_lower for word in ['bad', 'terrible', 'awful', 'hate', 'dislike']):
                return "negative"
            return "uncertain"

    def process_reviews(self, df: pd.DataFrame) -> List[str]:
        """处理所有评论"""
        reviews = df['review'].tolist()
        labels = []

        logger.info(f"开始生成情感标签，共 {len(reviews)} 条...")

        for i, review in enumerate(reviews, 1):
            # 进度显示
            if i % 10 == 0 or i == 1 or i == len(reviews):
                logger.info(f"处理进度: {i}/{len(reviews)}")

            # 截断过长的评论
            if len(review) > 2000:
                review = review[:2000] + "..."

            # 生成标签
            label = self.generate_label(review)
            labels.append(label)

            # 避免请求过快
            time.sleep(0.1)

        return labels

    def evaluate_results(self, df: pd.DataFrame) -> None:
        """评估结果"""
        if 'sentiment' not in df.columns or 'predicted_sentiment' not in df.columns:
            logger.warning("缺少必要列，无法评估")
            return

        # 只评估有效预测
        valid_mask = df['predicted_sentiment'].isin(['positive', 'negative'])
        valid_df = df[valid_mask]

        if len(valid_df) == 0:
            logger.warning("没有有效的预测结果")
            return

        # 计算准确率
        correct = (valid_df['sentiment'] == valid_df['predicted_sentiment']).sum()
        accuracy = correct / len(valid_df)

        logger.info("\n" + "=" * 50)
        logger.info("评估结果:")
        logger.info(f"有效预测数: {len(valid_df)}/{len(df)}")
        logger.info(f"准确率: {accuracy:.2%}")

        # 按类别统计
        for true_label in ['positive', 'negative']:
            mask = valid_df['sentiment'] == true_label
            if mask.any():
                class_correct = (valid_df[mask]['predicted_sentiment'] == true_label).sum()
                class_accuracy = class_correct / mask.sum()
                logger.info(f"{true_label}: {class_accuracy:.2%} ({int(class_correct)}/{int(mask.sum())})")

    def run(self):
        """运行自动打标"""
        logger.info("=" * 50)
        logger.info("IMDB情感自动标注系统")
        logger.info("=" * 50)

        # 加载数据
        input_path = self.config['data']['input_path']
        df = self.load_data(input_path)

        if len(df) == 0:
            logger.error("没有数据可处理")
            return

        # 处理评论
        predicted_labels = self.process_reviews(df)
        df['predicted_sentiment'] = predicted_labels

        # 保存结果
        output_path = self.config['data']['output_path']
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"结果已保存: {output_path}")

        # 评估
        self.evaluate_results(df)

        # 显示前5条结果
        self.display_preview(df)

        logger.info("处理完成!")

    def display_preview(self, df: pd.DataFrame, num_samples: int = 5):
        """显示结果预览"""
        logger.info("\n" + "=" * 50)
        logger.info(f"前{min(num_samples, len(df))}条结果预览:")
        logger.info("=" * 50)

        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            review = str(row.get('review', ''))

            # 清理和截断
            clean_review = re.sub(r'<[^>]+>', '', review)
            if len(clean_review) > 100:
                preview = clean_review[:100] + "..."
            else:
                preview = clean_review

            true_label = row.get('sentiment', 'N/A')
            pred_label = row.get('predicted_sentiment', 'N/A')

            logger.info(f"\n行 {i + 1}:")
            logger.info(f"  评论: {preview}")
            logger.info(f"  预测: {pred_label}, 真实: {true_label}")

            if true_label != 'N/A' and pred_label != 'N/A':
                if true_label == pred_label:
                    logger.info(f"  ✓ 正确")
                else:
                    logger.info(f"  ✗ 错误")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="IMDB情感自动标注")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--input", help="输入文件路径（覆盖配置）")
    parser.add_argument("--output", help="输出文件路径（覆盖配置）")
    parser.add_argument("--sample", type=int, help="采样数量")

    args = parser.parse_args()

    # 检查Ollama服务
    try:
        import ollama
        ollama.list()  # 测试连接
        logger.info("✓ Ollama服务正常")
    except Exception as e:
        logger.error(f"✗ 无法连接到Ollama: {e}")
        logger.info("请先运行: ollama serve")
        return

    # 创建标签生成器
    labeler = IMDBAutoLabeler(args.config)

    # 覆盖配置
    if args.input:
        labeler.config['data']['input_path'] = args.input
    if args.output:
        labeler.config['data']['output_path'] = args.output
    if args.sample:
        labeler.config['data']['sample_size'] = args.sample

    # 运行
    labeler.run()


if __name__ == "__main__":
    main()