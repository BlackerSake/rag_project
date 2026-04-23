import logging
import os
from logging.handlers import RotatingFileHandler

# 创建日志目录
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'  # 设置UTF-8编码
        ),
        logging.StreamHandler()
    ]
)

# 创建日志记录器
def get_logger(name):
    return logging.getLogger(name)
