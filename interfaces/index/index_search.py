import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss


class SearchInput(BaseModel):
    index_name: str = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    partition: Optional[str] = Field('', description='索引的分区')
    nprobe: Optional[int] = Field(10, description='查找索引中最近的 nprobe 个中心点')
    top_k: Optional[int] = Field(20, description='返回 top_k 个结果')


class Result(BaseModel):
    data: Any = Field(description='插入成功的数量')
    score: float = Field(0, description='该结果的 相似度 或 score')


class SearchResponse(Response):
    data: List[List[Result]] = Field(description='插入数据的结果')


@app.post('/v1/index/search',
          name="v1 index search",
          response_model=SearchResponse,
          description="搜索数据")
def index_search(_input: SearchInput):
    log_id = logs.uid()
    log_params = {k: v for k, v in _input.__dict__.items() if k != 'vectors'}
    logs.add(log_id, f'POST {logs.fn_name()}', f'payload: {log_params}')

    index_name = _input.index_name
    vectors = _input.vectors
    partition = _input.partition
    nprobe = _input.nprobe
    top_k = _input.top_k

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 0, 'msg': f'index_name 不能为空'})

    if not vectors:
        return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 0, 'msg': f'vectors 不能为空'})

    index = o_faiss.index(index_name, partition)
    if index is None:
        return logs.ret(log_id, logs.fn_name(), 'POST', {
            'code': 0, 'msg': f'index "{index_name}({partition})" 不存在，请先创建索引'})

    try:
        vectors = np.array(vectors).astype(np.float32)
        ret = o_faiss.search(index_name, vectors, partition, nprobe, top_k, log_id)
        return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 1, 'data': ret})
    except:
        return logs.ret_error(log_id, logs.fn_name(), 'POST', {
            'code': 0, 'msg': f'Error happen during searching in index "{index_name}({partition})"'})


if __name__ == '__main__':
    # 测试代码
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

    ret = index_search(SearchInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(3, 384))),
        top_k=3,
    ))

    for v_list in ret['data']:
        print('\n--------------------------------------')
        for v in v_list:
            print(v)
