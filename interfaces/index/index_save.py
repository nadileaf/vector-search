from fastapi import Query, Header
from typing import Optional
from interfaces.base import app, log
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss


@app.get('/v1/index/save',
         name="v1 index save",
         response_model=Response,
         description="保存索引到文件")
@log
def index_save(
        index_name: str = Query('', description='索引的名称; 若为 * , 则保存所有索引'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Header('_test'),
        log_id: int = None
):
    tenant = tenant if isinstance(tenant, str) else tenant.default
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'index_name 不能为空'}, logs.LEVEL_ERROR)

    _ret = 1
    if index_name == '*':
        o_faiss.save(tenant, log_id)
    else:
        _ret = o_faiss.save_one(tenant, index_name, partition, log_id)

    msg = 'Successfully' if _ret else 'Fail'
    msg = f'{msg} save index "{index_name}({partition})" (tenant: "{tenant}")'

    return {'code': _ret, 'msg': msg}


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    from interfaces.index.index_create import index_create
    from interfaces.index.index_train import index_train
    from interfaces.index.index_add_vectors import index_add_vectors, VectorInput

    index_create('test', 384, '')

    index_train(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(200)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(200)],
    ))

    index_save('test')
