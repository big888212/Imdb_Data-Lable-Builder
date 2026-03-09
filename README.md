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
前往Ollama安装Ollama程序
ollama pull llama3:8b

2. 配置模型

编辑 config.yaml文件：

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

操作系统    Windows 11 专业版 (64位)
CPU    超威半导体 AMD Ryzen 7 9700X 8-Core Processor 八核
主板    铭瑄 MS-Terminator B850M
内存    32GB(5600 MHz / 5600 MHz)
主硬盘    1000 GB (金士顿 SNV3S1000G)
显卡    英伟达 NVIDIA GeForce RTX 5070 Ti (16303 MB)
显示器    EXK1743 AMZ G25F6B-1 (24.5英寸 / 32位真彩色 / 280Hz)
声卡    Microsoft Misiom
网卡    Realtek Semiconductor Corp. Realtek 8852BE Wireless LAN WiFi 6 PCI-E NIC
在为100个评论打标签示例运行中耗时25秒


