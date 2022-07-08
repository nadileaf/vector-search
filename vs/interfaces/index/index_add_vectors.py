import numpy as np
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Union
from vs.interfaces.base import app, log
from vs.interfaces.definitions.common import Response
from vs.core.db import o_faiss
from vs.lib.utils import check_tenant


class VectorInput(BaseModel):
    tenant: Optional[str] = Field('_test', description='租户名称')
    index_name: str = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    texts: Optional[List[Any]] = Field(description='向量数据对应的文本或描述；与 vector 有一一对应关系，用于生成 id; '
                                                   '若不提供，则以 vector 本身进行去重，但耗时更久')
    info: List[Any] = Field([], description='结构化数据 或 文本数据 等; len(info) = 数据量')
    partition: Optional[str] = Field('', description='索引的分区')
    filter_exist: Optional[bool] = Field(False, description='是否去重插入向量到索引')
    add_default_partition: Optional[bool] = Field(False, description='是否将当前分区的内容也添加到 默认分区')
    ret_id: Optional[bool] = Field(False, description='是否返回数据的id')


class Result(BaseModel):
    count: int = Field(0, description='插入成功的数量')
    exist_count: int = Field(0, description='重复的数量')
    ids: Optional[List[int]] = Field([], description='插入的数据的 id 数组')


class InsertResponse(Response):
    data: Optional[Result] = Field(description='插入数据的结果')


@app.post('/v1/index/add_vectors',
          name="v1 index add vectors",
          response_model=InsertResponse,
          description="添加向量到索引 (内存；需要手动调用 save 接口才会将 索引数据保存到磁盘)")
@log
def index_add_vectors(_input: VectorInput, log_id: Union[int, str] = None):
    tenant = _input.tenant
    index_name = _input.index_name
    vectors = _input.vectors
    texts = _input.texts
    info = _input.info
    partition = _input.partition
    filter_exist = _input.filter_exist
    add_default_partition = _input.add_default_partition
    ret_id = _input.ret_id

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not vectors:
        return {'code': 0, 'msg': f'vectors 不能为空'}

    index = o_faiss.index(tenant, index_name, partition)
    if index is None:
        return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: "{tenant}") 不存在，请先创建索引'}

    if not index.is_trained:
        return {'code': 0, 'msg': f'index "{index_name}({partition})" (tenant: {tenant}) 还没 train，需要先调用 train 接口'}

    vectors = np.array(vectors).astype(np.float32)
    ret = o_faiss.add(tenant, index_name, vectors, texts, info, partition, filter_exist,
                      add_default=add_default_partition, log_id=log_id)

    # 根据用户输入参数，决定是否返回 ids，提高响应速度
    if not ret_id and 'ids' in ret:
        del ret['ids']
    return {'code': 1, 'data': ret}


if __name__ == '__main__':
    # 测试代码
    from vs.interfaces.index.index_create import index_create
    from vs.interfaces.index.index_train import index_train, TrainVectorInput

    index_create('test', 384, '', 200)

    index_train(TrainVectorInput(
        index_name='test',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    index_add_vectors(VectorInput(
        index_name='test',
        texts=[f'{i}' for i in range(600, 800)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(200, 400)],
    ))
