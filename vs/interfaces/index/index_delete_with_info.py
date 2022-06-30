import numpy as np
from pydantic import BaseModel, Field
from fastapi import Header
from typing import Optional, List, Any, Union
from vs.interfaces.base import app, log
from vs.interfaces.definitions.common import Response
from vs.core.db import o_faiss
from vs.lib.utils import check_tenant


class InfoInput(BaseModel):
    index_name: str = Field(description='索引的名称')
    vectors: List[List[float]] = Field(description='向量数据; shape = (数据量，数据的维度) ')
    texts: Optional[List[Any]] = Field(description='向量数据对应的文本或描述；与 vector 有一一对应关系，用于生成 id; '
                                                   '若不提供，则以 vector 本身进行去重，但耗时更久')
    info: List[Any] = Field([], description='结构化数据 或 文本数据 等; len(info) = 数据量')
    partition: Optional[str] = Field('', description='索引的分区')


@app.post('/v1/index/delete_with_info',
          name="v1 index delete with info",
          response_model=Response,
          description="删除数据，根据 info 删除")
@log
def index_delete_with_info(_input: InfoInput, tenant: Optional[str] = Header('_test'), log_id: Union[int, str] = None):
    tenant = tenant if isinstance(tenant, str) else tenant.default
    index_name = _input.index_name
    vectors = _input.vectors
    texts = _input.texts
    info = _input.info
    partition = _input.partition

    tenant = check_tenant(tenant)

    if not index_name:
        return {'code': 0, 'msg': f'index_name 不能为空'}

    if not texts and not vectors:
        return {'code': 0, 'msg': f'texts, vectors 不能为空'}

    vectors = np.array(vectors).astype(np.float32)
    o_faiss.delete_with_info(tenant, index_name, vectors, texts, info, partition, log_id=log_id)
    return {'code': 1}


if __name__ == '__main__':
    # 测试代码
    from vs.interfaces.index.index_create import index_create
    from vs.interfaces.index.index_train import index_train, TrainVectorInput
    from vs.interfaces.index.index_add_vectors import index_add_vectors, VectorInput
    from vs.interfaces.index.index_search import index_search, SearchInput

    index_create('test11', 384, '', 2000)

    index_train(TrainVectorInput(
        index_name='test11',
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
    ))

    ret_add = index_add_vectors(VectorInput(
        index_name='test11',
        texts=[f'{i}' for i in range(200)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(200, 384))),
        info=[{'value': i} for i in range(0, 200)],
    ))

    index_delete_with_info(InfoInput(
        index_name='test11',
        texts=[f'{i}' for i in range(3)],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(3, 384))),
        info=[{'value': i} for i in range(0, 3)],
    ))

    ret = index_search(SearchInput(
        index_names=['test11'],
        vectors=list(map(lambda l: list(map(float, l)), np.eye(5, 384))),
        top_k=4,
    ))

    for v_list in ret['data']:
        print('\n--------------------------------------')
        for v in v_list:
            print(v)
