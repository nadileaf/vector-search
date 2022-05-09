from fastapi import Query
from typing import Optional
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss


@app.get('/v1/index/save',
         name="v1 index save",
         response_model=Response,
         description="保存索引到文件")
def index_save(
        index_name: str = Query('', description='索引的名称; 若为 * , 则保存所有索引'),
        partition: Optional[str] = Query('', description='索引的分区'),
):
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default

    log_id = logs.uid()
    params = {'index_name': index_name, 'partition': partition}
    logs.add(log_id, f'GET {logs.fn_name()}', f'params: {params}')

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'index_name 不能为空'}, logs.LEVEL_ERROR)

    ret = 1
    if index_name == '*':
        o_faiss.save(log_id)
    else:
        ret = o_faiss.save_one(index_name, partition, log_id)
        if not ret:
            return logs.ret(log_id, logs.fn_name(), 'POST', {
                'code': ret, 'msg': f'Fail saving index "{index_name}({partition})"'}, logs.LEVEL_ERROR)

    return logs.ret(log_id, logs.fn_name(), 'POST', {
        'code': ret, 'msg': f'Successfully save index "{index_name}({partition})"'})


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
