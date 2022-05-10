import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss


class VectorInput(BaseModel):
    index_name: str = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    partition: Optional[str] = Field('', description='索引的分区')


class Result(BaseModel):
    count: int = Field(0, description='插入成功的数量')
    exist_count: int = Field(0, description='重复的数量')
    ids: List[int] = Field([], description='插入的数据的 id 数组')


@app.post('/v1/index/train',
          name="v1 index train",
          response_model=Response,
          description="训练索引")
def index_train(_input: VectorInput):
    log_id = logs.uid()
    log_params = {k: v for k, v in _input.__dict__.items() if k != 'vectors'}
    logs.add(log_id, f'POST {logs.fn_name()}', f'payload: {log_params}')

    index_name = _input.index_name
    vectors = _input.vectors
    partition = _input.partition

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'index_name 不能为空'})

    if not vectors:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'vectors 不能为空'})

    index = o_faiss.index(index_name, partition)
    if index is None:
        return logs.ret(log_id, logs.fn_name(), 'POST', {
            'code': 0, 'msg': f'index "{index_name}({partition})" 不存在，请先创建索引'})

    vectors = np.array(vectors).astype(np.float32)
    o_faiss.train(index_name, vectors, partition, log_id)
    return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 1})


if __name__ == '__main__':
    # 测试代码
    from interfaces.index.index_create import index_create

    index_create('test', 384, '')
    index_train(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))
