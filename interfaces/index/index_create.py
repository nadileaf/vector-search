from fastapi import Query
from typing import Optional
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss, get_index


@app.get('/v1/index/create',
         name="v1 index create",
         response_model=Response,
         description="创建索引")
def index_create(
        index_name: str = Query('', description='索引的名称'),
        dim_size: int = Query(0, description='向量的维度大小, 如 384, 1024 等'),
        partition: Optional[str] = Query('', description='索引的分区'),
        count: Optional[int] = Query(10000, description='预估的数据量'),
):
    index_name = index_name if isinstance(index_name, str) else index_name.default
    dim_size = int(dim_size) if isinstance(dim_size, int) else dim_size.default
    partition = partition if isinstance(partition, str) else partition.default
    count = int(count) if isinstance(count, int) else count.default

    log_id = logs.uid()
    params = {'index_name': index_name, 'partition': partition, 'count': count, 'dim_size': dim_size}
    logs.add(log_id, f'GET {logs.fn_name()}', f'params: {params}')

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'index_name 不能为空'})
    if not dim_size:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'dim_size 不能为空 或 0'})

    partition = partition if partition else o_faiss.DEFAULT

    if index_name not in o_faiss.indices:
        o_faiss.indices[index_name] = {}
    if partition in o_faiss.indices[index_name]:
        return logs.ret(log_id, logs.fn_name(), 'GET', {
            'code': 0, 'msg': f'index_name "{index_name}({partition})" 已存在，请先删除索引'})

    o_faiss.indices[index_name][partition] = get_index(count, dim_size)

    return logs.ret(log_id, logs.fn_name(), 'POST', {
        'code': 1, 'msg': f'Successfully create index "{index_name}({partition})"'})


if __name__ == '__main__':
    # 测试代码
    index_create('test', 384, '')
