from pydantic import Field
from fastapi import Header, Query
from typing import Optional, Union
from vs.interfaces.base import app, log
from vs.interfaces.definitions.common import Response
from vs.core.db import o_faiss


class IsTrainResponse(Response):
    data: Optional[bool] = Field(description='是否存在索引')


@app.get('/v1/index/is_train',
         name="v1 index is train",
         response_model=IsTrainResponse,
         description="检查索引是否存在")
@log
def index_is_train(
        index_name: str = Query('', description='索引的名称'),
        partition: Optional[str] = Query('', description='索引的分区'),
        tenant: Optional[str] = Header('_test'),
        log_id: Union[int, str] = None,
):
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default
    tenant = tenant if isinstance(tenant, str) else tenant.default

    partition = partition if partition else o_faiss.DEFAULT

    index = o_faiss.index(tenant, index_name, partition)
    if index is None:
        return {'code': 0, 'msg': f'index {index_name}({partition}) 不存在'}
    return {'code': 1, 'data': index.is_trained}


if __name__ == '__main__':
    # 测试代码
    from vs.interfaces.index.index_create import index_create

    index_create('test', 384, '')
    index_is_train('test', '')
