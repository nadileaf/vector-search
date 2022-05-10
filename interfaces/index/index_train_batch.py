import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss

# 用于缓存数据
d_key_2_vectors = {}


class VectorInput(BaseModel):
    index_name: str = Field(description='索引的名称')
    vectors: Optional[List[List[float]]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    partition: Optional[str] = Field('', description='索引的分区')
    start_train: Optional[bool] = Field(False, description='是否开始训练；false 则先缓存数据，等数据齐了再train；true 则开始训练')


class Result(BaseModel):
    count: int = Field(0, description='插入成功的数量')
    exist_count: int = Field(0, description='重复的数量')
    ids: List[int] = Field([], description='插入的数据的 id 数组')


@app.post('/v1/index/train_batch',
          name="v1 index train batch",
          response_model=Response,
          description="训练索引；避免数据过大，没法经过网络请求传输")
def index_train_batch(_input: VectorInput):
    log_id = logs.uid()
    log_params = {k: v for k, v in _input.__dict__.items() if k != 'vectors'}
    logs.add(log_id, f'POST {logs.fn_name()}', f'payload: {log_params}')

    index_name = _input.index_name
    vectors = _input.vectors
    partition = _input.partition
    start_train = _input.start_train

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'index_name 不能为空'})

    if not start_train and not vectors:
        return logs.ret(log_id, logs.fn_name(), 'GET', {'code': 0, 'msg': f'vectors 不能为空'})

    key = f'{index_name}____{partition}'
    if key not in d_key_2_vectors:
        d_key_2_vectors[key] = []

    if not start_train:
        d_key_2_vectors[key].extend(vectors)

    else:
        index = o_faiss.index(index_name, partition)
        if index is None:
            return logs.ret(log_id, logs.fn_name(), 'POST', {
                'code': 0, 'msg': f'index "{index_name}({partition})" 不存在，请先创建索引'})

        vectors = np.array(d_key_2_vectors[key]).astype(np.float32)
        o_faiss.train(index_name, vectors, partition, log_id)

        del d_key_2_vectors[key]

    return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 1})


if __name__ == '__main__':
    # 测试代码
    from interfaces.index.index_create import index_create

    index_create('test', 384, '')
    index_train_batch(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        start_train=False,
    ))

    index_train_batch(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.random.rand(20, 384))),
        start_train=False,
    ))

    index_train_batch(VectorInput(
        index_name='test',
        vectors=None,
        start_train=True,
    ))
