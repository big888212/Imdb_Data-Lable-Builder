IMDB电影评论情感自动标注系统

基于大型语言模型的电影评论情感自动标注工具，为情感分析任务提供高效的自动化数据标注解决方案。

📁 项目结构

imdb-auto-labeler/
├── data_loader.py      # 数据加载与预处理
├── main.py            # 主程序 - 自动标注逻辑
├── config.yaml        # 配置文件
├── requirements.txt   # 依赖包列表
├── data/              # 数据目录
│   ├── input/         # 输入数据
│   └── output/        # 输出结果
└── README.md         # 说明文档

1. 环境安装


# 克隆项目
git clone https://github.com/yourusername/imdb-auto-labeler.git

# 安装Python依赖
pip install -r requirements.txt

# 安装Ollama并下载模型
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:8b

2. 配置模型

编辑 config.yaml文件：

yaml
复制
model:
  name: "llama3:8b"     # Ollama模型名称
  temperature: 0.1      # 低随机性确保一致性
  max_tokens: 10        # 最大输出token数

data:
  input_path: "data/IMDB_Dataset.csv"    # 输入文件
  output_path: "data/IMDB_Dataset_labeled.csv"  # 输出文件
  sample_size: 100      # 采样数量，null表示全部
3. 运行标注
bash
复制
# 基本使用
python main.py

 数据格式
输入数据:

CSV文件，至少包含 review列：

review
"This movie was absolutely fantastic!"
"I was very disappointed with this film."
"A decent movie with some good moments."
输出结果

自动添加 predicted_sentiment列：

review,predicted_sentiment
"This movie was absolutely fantastic!",positive
"I was very disappointed with this film.",negative
"A decent movie with some good moments.",negative

	

成功生成标签比例

 技术栈

Python 3.8​ - 主要编程语言

Pandas​ - 数据处理

Ollama​ - 本地LLM推理

Llama3 8B​ - 基础语言模型

PyYAML​ - 配置管理

 设计思路
提示词设计

prompt = """
请分析以下电影评论的情感倾向。
只输出一个单词：POSITIVE 或 NEGATIVE。
不要输出任何解释、说明或其他内容。

电影评论：{review}

情感：
"""
关键设计

简洁明确：明确任务，避免歧义

格式约束：强制单单词输出

低随机性：temperature=0.1确保一致性

容错处理：多层异常处理和结果解析



