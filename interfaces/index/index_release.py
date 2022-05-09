from fastapi import Query
from typing import Optional
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss


@app.get('/v1/index/release',
         name="v1 index release",
         response_model=Response,
         description="释放索引，从内存中删除索引")
def index_release(
        index_name: str = Query('', description='索引的名称; 若为 * , 则加载索引索引'),
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
        for _index_name in o_faiss.indices.keys():
            o_faiss.release(_index_name, partition, log_id=log_id)
    else:
        ret = o_faiss.release(index_name, partition, log_id)
        if not ret:
            return logs.ret(log_id, logs.fn_name(), 'POST', {
                'code': ret, 'msg': f'Fail releasing index "{index_name}({partition})"'}, logs.LEVEL_ERROR)

    return logs.ret(log_id, logs.fn_name(), 'POST', {
        'code': ret, 'msg': f'Successfully release index "{index_name}({partition})"'})


if __name__ == '__main__':
    # 测试代码
    from interfaces.index.index_create import index_create

    index_create('test', 384)
    index_release('test', '')
