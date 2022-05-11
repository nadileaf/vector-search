import os
from pydantic import Field
from interfaces.base import app
from interfaces.definitions.common import Response
from lib import logs
from core.db import o_faiss, get_metric
from config.path import INDEX_DIR


class IndicesResponse(Response):
    data: dict = Field(description='索引列表 dict')


@app.get('/v1/index/list',
         name="v1 index list",
         response_model=IndicesResponse,
         description="列举索引")
def index_list():
    log_id = logs.uid()
    logs.add(log_id, f'GET {logs.fn_name()}', f'params: ')

    indices = {}
    for index_name, partitions in o_faiss.indices.items():
        indices[index_name] = {}
        for partition, _index in partitions.items():
            indices[index_name][partition] = {
                'storage_type': 'in_memory',
                'dim_size': _index.d,
                'n_total': _index.ntotal,
                'is_trained': _index.is_trained,
                'metric_type': get_metric(_index.metric_type)
            }

    for index_name in os.listdir(INDEX_DIR):
        index_dir = os.path.join(INDEX_DIR, index_name)
        if not os.path.isdir(index_dir):
            continue

        if index_name not in indices:
            indices[index_name] = {}

        for file_name in os.listdir(index_dir):
            if not file_name.endswith('.index'):
                continue

            partition = file_name[:-len('.index')]
            if partition not in indices[index_name]:
                indices[index_name][partition] = {'storage_type': 'in_disk'}

    return logs.ret(log_id, logs.fn_name(), 'POST', {'code': 1, 'data': indices})


if __name__ == '__main__':
    # 测试代码
    import json

    ret = index_list()
    print(json.dumps(ret, ensure_ascii=False, indent=2))
