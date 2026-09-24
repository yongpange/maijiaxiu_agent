import os

# image_generator 在导入时创建 Moonshot 客户端，没有密钥会直接报错。
os.environ.setdefault("MOONSHOT_API_KEY", "test-key")
