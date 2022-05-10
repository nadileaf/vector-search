from pydantic import Field
from fastapi import Query
from typing import Optional
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss


class ExistResponse(Response):
    data: Optional[bool] = Field(description='是否存在索引')


@app.get('/v1/index/exist',
         name="v1 index exist",
         response_model=ExistResponse,
         description="检查索引是否存在")
def index_exist(
        index_name: str = Query('', description='索引的名称'),
        partition: Optional[str] = Query('', description='索引的分区'),
):
    index_name = index_name if isinstance(index_name, str) else index_name.default
    partition = partition if isinstance(partition, str) else partition.default

    log_id = logs.uid()
    params = {'index_name': index_name, 'partition': partition}
    logs.add(log_id, f'GET {logs.fn_name()}', f'params: {params}')

    partition = partition if partition else o_faiss.DEFAULT

    exist = False if not index_name or index_name not in o_faiss.indices or partition not in o_faiss.indices[
        index_name] else True
    return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 1, 'data': exist})


if __name__ == '__main__':
    # 测试代码
    index_exist('test', '')
