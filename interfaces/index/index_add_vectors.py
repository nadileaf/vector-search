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
    texts: Optional[List[Any]] = Field(description='向量数据对应的文本或描述；与 vector 有一一对应关系，用于生成 id; '
                                                   '若不提供，则以 vector 本身进行去重，但耗时更久')
    info: List[Any] = Field([], description='结构化数据 或 文本数据 等; len(info) = 数据量')
    partition: Optional[str] = Field('', description='索引的分区')
    filter_exist: Optional[bool] = Field(False, description='是否去重插入向量到索引')
    add_default_partition: Optional[bool] = Field(False, description='是否将当前分区的内容也添加到 默认分区')


class Result(BaseModel):
    count: int = Field(0, description='插入成功的数量')
    exist_count: int = Field(0, description='重复的数量')
    ids: List[int] = Field([], description='插入的数据的 id 数组')


class InsertResponse(Response):
    data: Result = Field(description='插入数据的结果')


@app.post('/v1/index/add_vectors',
          name="v1 index add vectors",
          response_model=InsertResponse,
          description="添加向量到索引 (内存；需要手动调用 save 接口才会将 索引数据保存到磁盘)")
def index_add_vectors(_input: VectorInput):
    log_id = logs.uid()
    logs.add(log_id, f'POST {logs.fn_name()}', f'payload: {_input}')

    index_name = _input.index_name
    vectors = _input.vectors
    texts = _input.texts
    info = _input.info
    partition = _input.partition
    filter_exist = _input.filter_exist
    add_default_partition = _input.add_default_partition

    if not index_name:
        return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 0, 'msg': f'index_name 不能为空'})

    if not vectors:
        return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 0, 'msg': f'vectors 不能为空'})

    index = o_faiss.index(index_name, partition)
    if index is None:
        return logs.ret(log_id, logs.fn_name(), 'POST', {
            'code': 0, 'msg': f'index "{index_name}({partition})" 不存在，请先创建索引'})

    if not index.is_trained:
        return logs.ret(log_id, logs.fn_name(), 'POST', {
            'code': 0, 'msg': f'index "{index_name}({partition})" 还没 train，需要先调用 train 接口'}, logs.LEVEL_ERROR)

    try:
        vectors = np.array(vectors).astype(np.float32)
        ret = o_faiss.add(index_name, vectors, texts, info, partition, filter_exist, add_default=add_default_partition)
        return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 1, 'data': ret})
    except:
        return logs.ret_error(log_id, logs.fn_name(), 'POST', {'code': 0, 'msg': f'插入数据到 index error'})


if __name__ == '__main__':
    # 测试代码
    from interfaces.index.index_create import index_create
    from interfaces.index.index_train import index_train

    index_create('test', 384, '')

    index_train(VectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(600, 800)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(200, 400)],
    ))
