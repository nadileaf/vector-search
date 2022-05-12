import os
import sys
import json
import redis
from queue import Queue
import threading
from typing import Union, List, Any

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
_root_dir = os.path.split(_cur_dir)[0]
sys.path.append(_root_dir)

from config.redis import redis_conf

_o_redis = redis.Redis(**redis_conf)


def redis_save(k: Union[str, int], v, table_name: str = ''):
    if isinstance(v, dict) or isinstance(v, list) or isinstance(v, tuple):
        v = json.dumps(v)
    _o_redis.set(f'{table_name}____{k}', v)


def redis_get(k: Union[str, int], table_name: str = '') -> Union[dict, list, str, int, float]:
    v = _o_redis.get(f'{table_name}____{k}')
    if not v or isinstance(v, int) or isinstance(v, float):
        return v

    try:
        if isinstance(v, bytes):
            v = v.decode('utf-8')
        v = json.loads(v)
    except:
        pass
    return v


def redis_exist(key: Union[str, int], table_name: str = '') -> bool:
    return _o_redis.exists(f'{table_name}____{key}') > 0


def _redis_exist_thread(_queue: Queue, table_name, d_result: dict):
    while not _queue.empty():
        k = _queue.get()
        d_result[k] = redis_exist(k, table_name)


def redis_batch_exist(keys: List[Union[str, int]], table_name: str = '', num_thread=10) -> List[bool]:
    _queue = Queue()
    for k in keys:
        _queue.put(k)

    d_result = {}

    pool = []
    for i in range(min(num_thread, _queue.qsize())):
        t = threading.Thread(target=_redis_exist_thread, args=(_queue, table_name, d_result))
        t.start()
        pool.append(t)

    for t in pool:
        t.join()

    return list(map(lambda x: d_result[x], keys))


def _redis_get_thread(_queue: Queue, d_result: dict):
    while not _queue.empty():
        k, table_name = _queue.get()
        d_result[f'{table_name}____{k}'] = redis_get(k, table_name)


def redis_batch_get(keys: List[Union[str, int]], table_name: Union[str, List[str]] = '',
                    return_dict=False, num_thread=10) -> Union[List[Any], dict]:
    if isinstance(table_name, str):
        table_name = [table_name] * len(keys)

    key_tables = list(zip(keys, table_name))

    _queue = Queue()
    for k, t in key_tables:
        _queue.put([k, t])

    d_table_key_2_result = {}

    pool = []
    for i in range(min(num_thread, _queue.qsize())):
        t = threading.Thread(target=_redis_get_thread, args=(_queue, d_table_key_2_result))
        t.start()
        pool.append(t)

    for t in pool:
        t.join()

    if return_dict:
        return d_table_key_2_result
    else:
        return list(map(lambda x: d_table_key_2_result[f'{x[1]}____{x[0]}'], key_tables))


def _redis_save_thread(_queue: Queue, table_name):
    while not _queue.empty():
        tmp_data = _queue.get()
        redis_save(tmp_data['key'], tmp_data['value'], table_name)


def redis_batch_save(keys: List[Union[str, int]], info: List[dict], table_name: str = '', num_thread=10):
    _queue = Queue()
    for i, k in enumerate(keys):
        _queue.put({'key': k, 'value': info[i]})

    pool = []
    for i in range(min(num_thread, _queue.qsize())):
        t = threading.Thread(target=_redis_save_thread, args=(_queue, table_name))
        t.start()
        pool.append(t)

    for t in pool:
        t.join()


def redis_keys(prefix) -> List[str]:
    return list(map(lambda x: x.decode('utf-8'), _o_redis.keys(prefix)))


def redis_drop(db_name):
    keys = _o_redis.keys(f'{db_name}____*')
    if keys:
        _o_redis.delete(*keys)


def redis_del(keys: list, table_name=''):
    keys = list(map(lambda k: f'{table_name}____{k}', keys))
    if keys:
        _o_redis.delete(*keys)
