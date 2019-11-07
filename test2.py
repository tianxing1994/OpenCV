import sys
import logging

# 获取 logger 实例, 如果参数为空则返回 root logger
logger = logging.getLogger("hello world")

# 指定logger输出格式
formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

# 文件日志
# file_handler = logging.FileHandler("test.log")
# file_handler.setFormatter(formatter)  # 可以通过 setFormatter 指定输出格式

# 控制台日志
console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter  # 也可以直接给 formatter 赋值

# 为 logger 添加的日志处理器
# logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 指定日志的最低输出级别，默认为 WARN 级别
logger.setLevel(logging.INFO)


logger.info("hello world")
