#!/usr/bin/env python3
"""
数据加载模块 - 专为IMDB电影评论数据设计
提供简单的数据预处理功能
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, List
import os
import csv


class IMDBDataLoader:
    """IMDB数据加载器 """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化数据加载器

        Args:
            logger: 日志记录器
        """
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)

    def load_csv(self,
                 file_path: str,
                 text_column: str = 'review',
                 label_column: Optional[str] = 'sentiment',
                 sample_size: Optional[int] = None,
                 random_seed: int = 42) -> pd.DataFrame:
        """
        加载CSV格式的IMDB数据

        Args:
            file_path: CSV文件路径
            text_column: 评论文本列名
            label_column: 标签列名（可选）
            sample_size: 采样数量，None表示全部
            random_seed: 随机种子

        Returns:
            处理后的DataFrame
        """
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        self.logger.info(f"开始加载数据: {file_path}")

        # 1. 读取CSV文件
        df = self._safe_read_csv(file_path)
        self.logger.info(f"原始数据: {len(df)} 行, 列: {list(df.columns)}")

        # 2. 数据预处理
        df_clean = self._preprocess_data(df, text_column, label_column)

        # 3. 数据采样
        if sample_size and sample_size < len(df_clean):
            df_clean = df_clean.sample(
                n=sample_size,
                random_state=random_seed
            ).reset_index(drop=True)
            self.logger.info(f"采样 {sample_size} 条数据")

        # 4. 打印统计信息
        self._print_stats(df_clean, text_column, label_column)

        return df_clean

    def _safe_read_csv(self, file_path: str) -> pd.DataFrame:
        """读取CSV文件"""
        try:
            return pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        except Exception as e:
            self.logger.warning(f"标准读取失败: {e}，尝试其他方法")

            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) < 2:
                raise ValueError(f"CSV文件至少需要2行，实际{len(rows)}行")

            headers = rows[0]
            data_rows = rows[1:]

            # 处理字段数不一致的行
            processed_data = []
            for i, row in enumerate(data_rows, 1):
                if len(row) == len(headers):
                    processed_data.append(row)
                elif len(row) > len(headers):
                    # 如果字段数多于列数，合并多余的字段
                    if len(headers) == 2:
                        # 假设格式是 review,sentiment
                        merged_review = ','.join(row[:-1])
                        sentiment = row[-1]
                        processed_data.append([merged_review, sentiment])
                    else:
                        # 只取前len(headers)个字段
                        processed_data.append(row[:len(headers)])

            return pd.DataFrame(processed_data, columns=headers)

    def _preprocess_data(self,
                         df: pd.DataFrame,
                         text_column: str,
                         label_column: Optional[str]) -> pd.DataFrame:
        """数据预处理"""
        df_clean = df.copy()

        # 1. 检查文本列
        if text_column not in df_clean.columns:
            # 尝试自动识别
            for col in df_clean.columns:
                if 'review' in col.lower() or 'text' in col.lower():
                    self.logger.warning(f"自动识别文本列为: {col}")
                    text_column = col
                    break
            if text_column not in df_clean.columns:
                raise ValueError(f"未找到文本列，可用列: {list(df_clean.columns)}")

        # 2. 清理文本数据
        initial_count = len(df_clean)
        df_clean[text_column] = df_clean[text_column].fillna('').astype(str).str.strip()
        df_clean = df_clean[df_clean[text_column] != '']

        removed_count = initial_count - len(df_clean)
        if removed_count > 0:
            self.logger.info(f"移除 {removed_count} 条无效文本记录")

        # 3. 清理标签数据（如果存在）
        if label_column and label_column in df_clean.columns:
            df_clean[label_column] = df_clean[label_column].fillna('').astype(str).str.lower().str.strip()

            # 只保留有效的标签
            valid_labels = ['positive', 'negative', 'pos', 'neg']
            mask = df_clean[label_column].isin(valid_labels)

            invalid_count = len(df_clean) - mask.sum()
            if invalid_count > 0:
                self.logger.warning(f"发现 {invalid_count} 条无效标签，已过滤")
                df_clean = df_clean[mask]

            # 标准化标签
            label_map = {'pos': 'positive', 'neg': 'negative'}
            df_clean[label_column] = df_clean[label_column].replace(label_map)

        return df_clean.reset_index(drop=True)

    def _print_stats(self,
                     df: pd.DataFrame,
                     text_column: str,
                     label_column: Optional[str]):
        """打印数据统计信息"""
        self.logger.info("=" * 50)
        self.logger.info("数据统计信息:")
        self.logger.info(f"总样本数: {len(df)}")

        # 文本长度统计
        df['text_length'] = df[text_column].apply(len)
        self.logger.info(f"平均文本长度: {df['text_length'].mean():.1f} 字符")

        # 标签分布
        if label_column and label_column in df.columns:
            label_dist = df[label_column].value_counts()
            self.logger.info("标签分布:")
            for label, count in label_dist.items():
                percentage = count / len(df) * 100
                self.logger.info(f"  {label}: {count} 条 ({percentage:.1f}%)")

        self.logger.info("=" * 50)

    def save_data(self,
                  df: pd.DataFrame,
                  file_path: str):
        """
        保存数据到CSV文件

        Args:
            df: 要保存的DataFrame
            file_path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 保存为CSV
        df.to_csv(file_path, index=False, encoding='utf-8')
        self.logger.info(f"数据已保存: {file_path} ({len(df)} 条)")