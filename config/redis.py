import os
from config import env

if env.ENV == env.DEV:
    redis_conf = {'host': 'localhost', 'port': '6379'}
    # redis_conf = {'host': 'mesoor.f3322.net', 'port': '31012'}
elif env.ENV == env.PRE_TEST:
    redis_conf = {'host': '10.10.10.202', 'port': '1012'}
else:
    host = os.getenv('REDIS_HOST')
    port = os.getenv('REDIS_PORT')
    redis_conf = {'host': host if host else 'text2sql-redis', 'port': port if port else '6379'}
