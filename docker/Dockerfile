# 使用基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量，确保 Python 输出直接打印到控制台，不进行缓存
ENV PYTHONUNBUFFERED=1

# 安装系统依赖（如果某些库需要编译，可以在此添加，安装完后清理以减小镜像体积）
# 例如：netifaces 和一些图像处理库可能需要 gcc
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 升级 pip 并安装 Python 依赖
# 注意：PIL 的安装包名为 Pillow，docx 的安装包名为 python-docx
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    nltk \
    pandas \
    netifaces \
    beautifulsoup4 \
    Pillow \
    python-docx \
    openpyxl \
    numpy \
    gensim \
    PyPDF2 \
    PyJWT

# (可选) 如果你需要使用 NLTK 的数据包，可以提前下载
# RUN python -m nltk.downloader punkt

# 将当前目录内容复制到镜像中
COPY . .

# 运行你的程序
# CMD ["python", "main.py"]