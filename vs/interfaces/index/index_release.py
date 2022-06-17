from fastapi import Query, Header
from typing import Optional, Union
from vs.interfaces.base import app, log
from vs.interfaces.definitions.common import Response
from vs.core.db import o_faiss


@app.get('/v1/index/release',
         name="v1 index release",
         response_model=Response,
         description="释放索引，从内存中删除索引")
@log
def index_release(
        index_name: str = Query('', description='索引的名称; 若为 * , 则加载索引索引'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Header('_test'),
        log_id: Union[int, str] = None,
):
    tenant = tenant if isinstance(tenant, str) else tenant.default
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    _ret = 1
    if index_name == '*':
        for _index_name in o_faiss.indices.keys():
            o_faiss.release(tenant, _index_name, partition, log_id=log_id)
    else:
        _ret = o_faiss.release(tenant, index_name, partition, log_id=log_id)

    msg = 'Successfully' if _ret else 'Fail'
    msg = f'{msg} releasing index "{index_name}({partition})" (tenant: "{tenant}")'

    return {'code': _ret, 'msg': msg}


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    from vs.interfaces.index.index_load import index_load
    from vs.interfaces.index.index_search import index_search, SearchInput

    index_load('test', '')

    ret = index_search(SearchInput(
        index_names=['test'],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(3, 384))),
        top_k=3,
    ))

    for v_list in ret['data']:
        print('\n--------------------------------------')
        for v in v_list:
            print(v)

    index_release('test', '')

    ret = index_search(SearchInput(
        index_names=['test'],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(3, 384))),
        top_k=3,
    ))
