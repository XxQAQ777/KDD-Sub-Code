import configparser

# 测试配置文件读取
config_file = './METRLA_DGCRN.conf'
config = configparser.ConfigParser()
config.read(config_file)

print("配置文件路径:", config_file)
print("配置文件是否存在:", config_file)
print("读取的sections:", config.sections())
print("data section 是否存在:", 'data' in config)
if 'data' in config:
    print("data section 内容:", dict(config['data']))
else:
    print("data section 不存在")
