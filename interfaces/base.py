import time
from functools import wraps
from pydantic import BaseModel
from fastapi import FastAPI
from lib import logs

app = FastAPI()


def _process_arg(arg):
    if isinstance(arg, BaseModel):
        arg = arg.__dict__

    if isinstance(arg, dict):
        new_arg = {}
        for k, v in arg.items():
            if k == 'vectors' and v is not None:
                new_arg[k] = len(v)
            elif isinstance(v, list):
                new_arg[k] = v[:10]
            elif isinstance(v, dict):
                new_arg[k] = {}
                for kk, vv in v.items():
                    if kk == 'vectors' and vv is not None:
                        new_arg[k][kk] = len(vv)
                    elif isinstance(vv, list):
                        new_arg[k][kk] = vv[:10]
                    else:
                        new_arg[k][kk] = vv
            else:
                new_arg[k] = v
        arg = new_arg

    return arg


def log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 生成请求的唯一id
        log_id = logs.uid()

        # 记录请求的参数
        log_args = [_process_arg(arg) for arg in args]
        log_kwargs = {k: _process_arg(v) for k, v in kwargs.items()}
        logs.add(log_id, f'Request {func.__name__}', f'args: {log_args}, kwargs: {log_kwargs}')
        s_time = time.time()

        # 执行 函数本身
        kwargs['log_id'] = log_id
        _ret = func(*args, **kwargs)

        e_time = time.time()

        # 修改日志的等级
        _level = logs.LEVEL_MSG
        if isinstance(_ret, dict):
            _ret['log_id'] = log_id
            if 'code' not in _ret or not _ret['code']:
                _level = logs.LEVEL_ERROR

        # 记录返回结果
        logs.add(log_id, f'Response {func.__name__}', f'(use time: {e_time - s_time:.4f}s) '
                                                      f'return {_process_arg(_ret)}', _level=_level)
        return _ret

    return wrapper
