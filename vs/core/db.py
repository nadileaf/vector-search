import os
import math
import time
import faiss
from queue import Queue
import threading
import numpy as np
from typing import List, Union, Any, Dict
from six.moves import cPickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from vs.config.path import INDEX_DIR
from vs.lib.utils import md5, uid, get_relative_file
from vs.lib.redis_utils import redis_get, redis_save, redis_batch_save, redis_drop, redis_batch_get, redis_del, \
    redis_batch_exist
from vs.lib import logs


class Faiss:
    DEFAULT = '__default'

    def __init__(self):
        # 记录索引
        self.indices = {}

        # 记录 index 每个分区的滑动平均向量
        self.mv_indices = {}

        # 加载已有索引
        self.load('*')

    def index(self, tenant: str, index_name: str, partition: str = '') -> Union[None, faiss.Index]:
        if tenant not in self.indices or index_name not in self.indices[tenant] or \
                (partition and partition not in self.indices[tenant][index_name]):
            return
        partition = partition if partition else self.DEFAULT
        return self.indices[tenant][index_name][partition]

    def train(self, tenant: str, index_name: str, vectors: np.ndarray, partition: str = '',
              log_id: Union[str, int] = 'Faiss'):
        index = self.index(tenant, index_name, partition)
        if index is not None:
            logs.add(log_id, 'train', f'start training index "{index_name}({partition})" '
                                      f'(len: {len(vectors)}, tenant: {tenant})')
            s_time = time.time()
            index.train(vectors)
            logs.add(log_id, 'train', f'finish training index "{index_name}({partition})" '
                                      f'(use time: {time.time() - s_time:.4f}s, tenant: {tenant})')

    def add(self,
            tenant: str,
            index_name: str,
            vectors: np.ndarray,
            texts: List[Any] = None,
            info: List[dict] = None,
            partition: str = '',
            filter_exist=False,
            add_default=False,
            log_id: Union[str, int] = 'Faiss') -> dict:
        """ 插入数据到 index，返回 插入成功的数量 insert_count """

        origin_len = len(vectors)
        partition = partition if partition else self.DEFAULT

        index = self.index(tenant, index_name, partition)
        index_type = get_index_type(index)

        log_name = f'{index_name}({partition}) (tenant: {tenant})'
        logs.add(log_id, 'add', f'start inserting data (len: {origin_len}) to "{log_name}" ...')

        # 预处理 info
        info = process_info(info, len(vectors), partition)

        origin_s_time = s_time = time.time()

        if not filter_exist and index_type.startswith('Flat'):
            ids = list(range(index.ntotal, index.ntotal + origin_len))

            table_name = get_md5_table(tenant, index_name, partition, 'Flat')

            info = [''] * len(texts) if not info else info
            md5_ids = list(map(md5, zip(texts, info)))

            redis_batch_save(md5_ids, ids, table_name)

        else:
            # 根据 index 类型 选用不同的 uid 函数
            if index_type.startswith('Flat'):
                id_queue = Queue()
                for i in range(origin_len):
                    id_queue.put(index.ntotal + i)
            else:
                id_queue = None

            # 获取数据的 id
            ids = get_uids(tenant, index_name, texts if texts else vectors, info, partition, id_queue)

        filter_ids = ids

        logs.add(log_id, 'add', f'finish getting uids '
                                f'(len: {len(ids)}, use_time: {time.time() - s_time:.4f}s) for "{log_name}"')

        # 过滤重复数据
        if filter_exist:
            s_time = time.time()

            filter_indices = filter_duplicate(tenant, index_name, ids, partition)
            if not filter_indices:
                logs.add('Faiss', 'add', f'all data is existed, insert (len: 0/0/{origin_len}) to "{log_name}"')
                return {'count': 0, 'exist_count': origin_len, 'ids': ids}

            # 根据 filter_indices 取 不重复 的 数据
            filter_ids = [filter_ids[i] for i in filter_indices]
            info = [info[i] for i in filter_indices]
            vectors = vectors[filter_indices]

            logs.add(log_id, 'add', f'finish filtering duplicate vectors for {log_name} '
                                    f'(left: {len(filter_ids)}, use_time: {time.time() - s_time:.4f}s) ')

        s_time = time.time()

        # 添加 到 index
        if index_type.startswith('Flat'):
            index.add(vectors)
        else:
            index.add_with_ids(vectors, np.array(filter_ids))

        logs.add(log_id, 'add', f'finish adding data to index (len: {len(filter_ids)}/{origin_len}) "{log_name},'
                                f' use time: {time.time() - s_time:.4f}s)"')

        # 若有 partition，记录该 partition 的滑动平均向量
        if partition and partition != self.DEFAULT:
            s_time = time.time()

            if tenant not in self.mv_indices:
                self.mv_indices[tenant] = {}
            if index_name not in self.mv_indices[tenant]:
                self.mv_indices[tenant][index_name] = {}
            if partition not in self.mv_indices[tenant][index_name]:
                self.mv_indices[tenant][index_name][partition] = None
            mv_index = self.mv_indices[tenant][index_name][partition]

            if not mv_index:
                self.mv_indices[tenant][index_name][partition] = {
                    'vector': np.mean(vectors, axis=0),
                    'count': len(vectors)
                }

            else:
                count = mv_index['count']
                avg_embedding = mv_index['vector']

                for v in vectors:
                    beta = min(0.001, 2 / (1 + count))
                    count += 1
                    avg_embedding = avg_embedding * (1 - beta) + v * beta

                self.mv_indices[tenant][index_name][partition] = {'vector': avg_embedding, 'count': count}

            logs.add(log_id, 'add', f'finish updating moving average index "{log_name}" '
                                    f'(use_time: {time.time() - s_time:.4f}s)')

        if add_default and partition and partition != self.DEFAULT:
            self.add(tenant, index_name, vectors, texts, info, self.DEFAULT, filter_exist, log_id=log_id)

        s_time = time.time()

        # 添加 具体 info 到 redis
        redis_batch_save(filter_ids, info, get_table_name(tenant, index_name, partition))

        logs.add(log_id, 'add', f'finish inserting data (len: {len(filter_ids)}/{origin_len}, '
                                f'save_info time: {time.time() - s_time:.4f}s, '
                                f'total time: {time.time() - origin_s_time:.4f}s) to "{log_name}"')
        return {'count': origin_len, 'exist_count': origin_len - len(filter_ids), 'ids': ids}

    def save_one(self, tenant: str, index_name: str, partition: str = '', log_id: Union[str, int] = 'Faiss') -> int:
        logs.add(log_id, 'save_one', f'Start saving index "{index_name}({partition})" (tenant: {tenant}) ... ')
        s_time = time.time()

        _index = self.index(tenant, index_name, partition)
        if _index is None:
            logs.add(log_id, 'save_one', f'Index "{index_name}({partition})" (tenant: {tenant}) is not existed',
                     _level=logs.LEVEL_WARNING)
            return 0

        if not partition:
            for partition, _index in self.indices[tenant][index_name].items():
                if _index is None:
                    continue

                index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=INDEX_DIR)
                faiss.write_index(_index, index_path)

            if tenant in self.mv_indices and index_name in self.mv_indices[tenant] and \
                    self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=INDEX_DIR), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

        else:
            index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=INDEX_DIR)
            faiss.write_index(_index, index_path)

        logs.add(log_id, 'save_one', f'Successfully saving index "{index_name}({partition})" '
                                     f'(use_time: {time.time() - s_time:.4f}s, tenant: {tenant})')
        return 1

    def save(self, tenant: str, log_id: Union[str, int] = 'Faiss'):
        """ 保存当前的所有索引到文件里 """
        logs.add(log_id, 'save', f'Start saving all indices ... ')
        s_time = time.time()

        if tenant not in self.indices:
            logs.add(log_id, 'save', f'tenant "{tenant}" 没有索引')
            return

        for index_name, index in self.indices[tenant].items():
            logs.add(log_id, 'save', f'Start saving index "{index_name}" (tenant: {tenant}) ... ')
            s_time2 = time.time()

            for partition, _index in index.items():
                if _index is None:
                    continue

                index_path = get_relative_file(tenant, index_name, f'{partition}.index', root=INDEX_DIR)
                faiss.write_index(_index, index_path)

            if tenant in self.mv_indices and index_name in self.mv_indices[tenant] and \
                    self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=INDEX_DIR), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

            logs.add(log_id, 'save', f'Finish saving index "{index_name}" '
                                     f'(use time: {time.time() - s_time2:.4f}s, tenant: {tenant})')

        logs.add(log_id, 'save', f'Finish saving all indices (use time: {time.time() - s_time:.4f}s)')

    def load_one(self, tenant: str, index_name: str, partition: str = '', log_id: Union[str, int] = 'Faiss') -> int:
        logs.add(log_id, 'load_one', f'Start loading index "{index_name}({partition})" ...')
        s_time = time.time()

        if not partition:
            index_dir = os.path.join(INDEX_DIR, tenant, index_name)
            if not os.path.isdir(index_dir) or not os.listdir(index_dir):
                return 0

            logs.add(log_id, 'load_one', f'loading index "{index_name}" (tenant: {tenant}) ... ')

            if tenant not in self.indices:
                self.indices[tenant] = {}
            if tenant not in self.mv_indices:
                self.mv_indices[tenant] = {}
            if index_name not in self.indices[tenant]:
                self.indices[tenant][index_name] = {}

            for file_name in os.listdir(index_dir):
                if not file_name.endswith('.index'):
                    continue

                # 若本身已在内存中，无需重复加载
                partition = file_name[:-len('.index')]
                if self.index(tenant, index_name, partition) is not None:
                    continue

                index_path = os.path.join(index_dir, file_name)
                self.indices[tenant][index_name][partition] = faiss.read_index(index_path)

            # 若文件存在 且 没有被加载到内存
            mv_index_path = os.path.join(index_dir, 'mv_index.pkl')
            if os.path.exists(mv_index_path) and (index_name not in self.mv_indices[tenant] or
                                                  self.mv_indices[tenant][index_name] is None):
                with open(mv_index_path, 'rb') as f:
                    self.mv_indices[tenant][index_name] = pickle.load(f)

        else:
            index_path = os.path.join(INDEX_DIR, tenant, index_name, f'{partition}.index')
            if not os.path.exists(index_path):
                return 0

            if tenant not in self.indices:
                self.indices[tenant] = {}
            if tenant not in self.mv_indices:
                self.mv_indices[tenant] = {}
            if index_name not in self.indices[tenant]:
                self.indices[tenant][index_name] = {}

            if self.index(tenant, index_name, partition) is None:
                self.indices[tenant][index_name][partition] = faiss.read_index(index_path)

        logs.add(log_id, 'load_one', f'Successfully loading index "{index_name}({partition})" '
                                     f'(use time: {time.time() - s_time:.4f}s, tenant: {tenant})')
        return 1

    def load(self, tenant: str, log_id: Union[str, int] = 'Faiss'):
        """ 从文件中加载索引 """
        logs.add(log_id, 'load', f'Start loading all indices (tenant: "{tenant}") ... ')
        s_time = time.time()

        for _tenant in os.listdir(INDEX_DIR):
            if tenant not in ['*', _tenant]:
                continue

            _tenant_dir = os.path.join(INDEX_DIR, _tenant)
            if not os.path.isdir(_tenant_dir):
                continue

            if _tenant not in self.indices:
                self.indices[_tenant] = {}
            if _tenant not in self.mv_indices:
                self.mv_indices[_tenant] = {}

            for index_name in os.listdir(_tenant_dir):
                index_dir = os.path.join(_tenant_dir, index_name)
                if not os.path.isdir(index_dir):
                    continue

                logs.add(log_id, 'load', f'Start loading index "{index_name}" (tenant: {_tenant}) ... ')
                s_time2 = time.time()

                if index_name not in self.indices[_tenant]:
                    self.indices[_tenant][index_name] = {}

                for file_name in os.listdir(index_dir):
                    if not file_name.endswith('.index'):
                        continue

                    # 若已在内存，无需重复加载
                    partition = file_name[:-len('.index')]
                    if self.index(_tenant, index_name, partition) is not None:
                        continue

                    index_path = os.path.join(index_dir, file_name)
                    self.indices[_tenant][index_name][partition] = faiss.read_index(index_path)

                mv_index_path = os.path.join(index_dir, 'mv_index.pkl')
                if os.path.exists(mv_index_path) and (index_name not in self.mv_indices[_tenant] or
                                                      self.mv_indices[_tenant][index_name] is None):
                    with open(mv_index_path, 'rb') as f:
                        self.mv_indices[_tenant][index_name] = pickle.load(f)

                logs.add(log_id, 'load', f'Finish loading index "{index_name}" '
                                         f'(use time: {time.time() - s_time2:.4f}s, tenant: {_tenant})')

        logs.add(log_id, 'load', f'Finish loading all indices '
                                 f'(use time: {time.time() - s_time:.4f}s, tenant: "{tenant}")')

    def release(self, tenant: str, index_name: str, partition: str = '', log_id: Union[str, int] = 'Faiss') -> int:
        log_name = f'"{index_name}({partition})" (tenant: {tenant})'

        # release index 前，先保存索引
        ret = self.save_one(tenant, index_name, partition, log_id)
        if not ret:
            logs.add(log_id, 'release', f'Fail in saving {log_name} before releasing',
                     _level=logs.LEVEL_ERROR)
            return 0

        logs.add(log_id, 'release', f'Start releasing index {log_name} ...')
        s_time = time.time()

        if not partition:
            if tenant in self.indices and index_name in self.indices[tenant]:
                del self.indices[tenant][index_name]

        else:
            if tenant in self.indices and index_name in self.indices[tenant] and \
                    partition in self.indices[tenant][index_name]:
                del self.indices[tenant][index_name][partition]

        logs.add(log_id, 'release', f'Successfully releasing index {log_name} (use time: {time.time() - s_time:.4f}s)')
        return 1

    def search(self,
               vectors: np.ndarray,
               tenant: str,
               index_names: List[str],
               partitions: List[str] = None,
               nprobe=10,
               top_k=20,
               log_id: Union[str, int] = 'Faiss') -> List[List[dict]]:
        log_name = f'"{index_names}({partitions})" (tenant: {tenant})'
        logs.add(log_id, logs.fn_name(), f'start searching in {log_name} for vectors ...')

        if vectors is None or not vectors.any():
            logs.add(log_id, logs.fn_name(), f'Error: vectors cannot be empty', _level=logs.LEVEL_ERROR)
            return []

        total_s_time = time.time()

        results = [[] for i in range(len(vectors))]
        avg_results = [{} for i in range(len(vectors))]
        ids = []
        table_names = []

        partitions = partitions if partitions else [''] * len(index_names)
        for i, index_name in enumerate(index_names):
            partition = partitions[i] if partitions[i] else self.DEFAULT

            # 获取 index
            index = self.index(tenant, index_name, partition)
            if index is None:
                continue

            table_name = get_table_name(tenant, index_name, partition)

            if partition == self.DEFAULT and tenant in self.mv_indices and index_name in self.mv_indices[tenant]:
                s_time = time.time()

                # 获取该 index 每个 partition 的 滑动平均向量
                mv_indices = self.mv_indices[tenant][index_name]
                mv_indices = dict(filter(lambda x: x[1], mv_indices.items()))

                tmp_partitions = list(mv_indices.keys())
                avg_vectors = list(map(lambda x: x['vector'], mv_indices.values()))

                # 根据 滑动平均向量，计算语义相似度
                sims = cosine_similarity(vectors, avg_vectors)

                # 整理、排序 滑动平均向量计算得出的结果
                for _j, sim in enumerate(sims):
                    sim = list(zip(tmp_partitions, sim))
                    sim.sort(key=lambda x: -x[1])
                    avg_results[_j][table_name] = dict(sim)

                logs.add(log_id, logs.fn_name(), f'finish mv sim (use time: {time.time() - s_time:.4f}s)')

            index.nprobe = nprobe

            s_time = time.time()
            D, I = index.search(vectors, top_k)
            logs.add(log_id, logs.fn_name(), f'finish index search (use time: {time.time() - s_time:.4f}s, '
                                             f'index: {index_name}({partition}), tenant: {tenant})')

            tmp_ids = list(set(list(map(int, I.reshape(-1)))))
            tmp_table_names = [table_name] * len(tmp_ids)

            ids += tmp_ids
            table_names += tmp_table_names

            for _i, _result_ids in enumerate(I):
                similarities = D[_i]
                results[_i] += [
                    {'id': _id, 'score': _similarity, 'table_name': table_name}
                    for _id, _similarity in set(list(zip(_result_ids, similarities))) if _id != -1
                ]

        logs.add(log_id, logs.fn_name(), f'finish index search '
                                         f'(use time: {time.time() - total_s_time:.4f}s, tenant: {tenant})')

        s_time = time.time()
        d_table_id_2_info = redis_batch_get(ids, table_names, return_dict=True)
        logs.add(log_id, logs.fn_name(), f'finish get info (use time: {time.time() - s_time:.4f}s, tenant: {tenant})')

        for _i, one_results in enumerate(results):
            new_result = []

            tmp_avg_result = avg_results[_i]
            for val in one_results:
                table_id = f"{val['table_name']}____{val['id']}"
                data = d_table_id_2_info[table_id] if table_id in d_table_id_2_info else None
                if not data:
                    continue

                if val['table_name'] not in tmp_avg_result:
                    avg_similarity = 1.
                else:
                    _partition = data['partition'] if 'partition' in data else ''
                    tmp_avg_ret = tmp_avg_result[val['table_name']]
                    avg_similarity = tmp_avg_ret[_partition] if _partition in tmp_avg_ret else 0.

                new_result.append({'data': data, 'score': combine_avg_score(avg_similarity, val['score'])})

            new_result.sort(key=lambda x: -x['score'])

            d_new_result = {}
            for v in new_result:
                k = f'{v}'
                if k not in d_new_result:
                    d_new_result[k] = v
                if len(d_new_result) >= top_k:
                    break

            results[_i] = list(d_new_result.values())

        logs.add(log_id, logs.fn_name(), f'Finish all searching in {log_name} '
                                         f'(use_time: {time.time() - total_s_time:.4f}s): {results}')
        return results

    def delete_with_id(self, ids: List[int], tenant: str, index_name: str, partition: str = '',
                       log_id: Union[str, int] = 'Faiss'):

        logs.add(log_id, logs.fn_name(), f'Start deleting ids from "{index_name}({partition})" '
                                         f'(len: {len(ids)}, tenant: {tenant}) ... ')

        partition = partition if partition else self.DEFAULT
        ids = list(filter(lambda x: x or x == 0, ids))

        redis_del(ids, table_name=get_table_name(tenant, index_name, partition))

        logs.add(log_id, logs.fn_name(), f'Finish deleting ids from "{index_name}({partition})" (tenant: {tenant})')

    def delete_with_info(self,
                         tenant: str,
                         index_name: str,
                         vectors: np.ndarray,
                         texts: List[Any] = None,
                         info: List[dict] = None,
                         partition: str = '',
                         log_id: Union[str, int] = 'Faiss'):
        partition = partition if partition else self.DEFAULT

        index = self.index(tenant, index_name, partition)
        index_type = get_index_type(index)

        # 预处理 info
        info = process_info(info, len(vectors), partition)

        table_name = get_md5_table(tenant, index_name, partition, 'Flat' if index_type.startswith('Flat') else 'IVF')
        md5_ids = list(map(md5, zip(texts, info)))
        ids = redis_batch_get(md5_ids, table_name)

        self.delete_with_id(ids, tenant, index_name, partition, log_id)


def process_info(info: List[Any], length: int, partition: str = ''):
    info = info if info else [''] * length
    for _i, _v in enumerate(info):
        if not isinstance(_v, dict):
            info[_i] = {'value': _v, 'partition': partition}
        elif 'partition' not in _v:
            _v['partition'] = partition
    return info


def process_score(score) -> float:
    """ 格式化 similarity score """
    return max(min(round(float(score), 4), 1.), 0)


def combine_avg_score(avg_score, score):
    """ 结合 moving avg similarity score 与 当前 similarity score """
    score = process_score(score)
    if avg_score < 0.7:
        return score * 0.85 + 0.15 * avg_score
    else:
        return score


def get_nlist(count: int):
    if count <= 80:
        return 1
    elif count <= 300:
        return int(math.sqrt(count) / 2)
    elif count <= 1000:
        return int(math.sqrt(count) * 0.75)
    elif count <= 5000:
        return int(math.sqrt(count) * 1)
    elif count <= 15000:
        return int(math.sqrt(count) * 1.5)
    elif count <= 50000:
        return int(math.sqrt(count) * 2)
    else:
        return int(math.sqrt(count) * 2.5)


def get_index(count: int, dim: int):
    nlist = get_nlist(count)
    if count <= 1024:
        return faiss.IndexFlatIP(dim)
    elif count <= 20000:
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatIP(dim)
        return faiss.IndexIVFPQ(quantizer, dim, nlist, int(dim / 4), 8, faiss.METRIC_INNER_PRODUCT)


def get_table_name(tenant: str, index_name: str, partition: str = ''):
    return f'{tenant}____{index_name}____{partition}'


def get_md5_table(tenant: str, index_name: str, partition: str = '', id_type='IVF'):
    return get_table_name(tenant, index_name, partition) + f'____md5_{id_type}'


def _get_uid_thread(_queue: Queue, table_name: str, d_mid_2_uid: Dict[int, int]):
    while not _queue.empty():
        mid = _queue.get()

        _uid = redis_get(mid, table_name)
        if not _uid:
            _uid = uid()
            redis_save(mid, _uid, table_name)

        d_mid_2_uid[mid] = int(_uid)


def get_uids(
        tenant: str,
        index_name: str,
        texts: Union[np.ndarray, List[Any]],
        info: List[Any] = None,
        partition: str = '',
        id_queue: Queue = None,
        num_thread=15) -> List[int]:
    """ 并发获取 uid """
    table_name = get_md5_table(tenant, index_name, partition, 'IVF' if id_queue is None else 'Flat')

    info = [''] * len(texts) if not info else info
    md5_ids = list(map(md5, zip(texts, info)))

    if id_queue is not None:
        ids = []
        for mid in md5_ids:
            _uid = redis_get(mid, table_name)
            if not _uid:
                _uid = id_queue.get()
                redis_save(mid, _uid, table_name)
            ids.append(int(_uid))
        return ids

    _queue = Queue()
    for mid in md5_ids:
        _queue.put(mid)

    d_mid_2_uid: Dict[int, int] = {}

    pool = []
    for thread_id in range(num_thread):
        thread = threading.Thread(target=_get_uid_thread, args=(_queue, table_name, d_mid_2_uid))
        thread.start()
        pool.append(thread)

    for thread in pool:
        thread.join()

    return [d_mid_2_uid[mid] for mid in md5_ids]


def filter_duplicate(tenant: str, index_name: str, ids: List[int], partition: str = '') -> List[int]:
    """ 返回 没有重复(已存在db) 的数据的 位置index """
    if not ids:
        return []

    table_name = get_table_name(tenant, index_name, partition)
    rets = redis_batch_exist(ids, table_name)
    return [i for i, ret in enumerate(rets) if not ret]


def get_metric(metric_type: int):
    if metric_type == 0:
        return 'inner_product'
    elif metric_type == 1:
        return 'L1'
    elif metric_type == 2:
        return 'L2'
    elif metric_type == 3:
        return 'L_inf'
    elif metric_type == 4:
        return 'Lp'
    elif metric_type == 22:
        return 'JensenShannon'
    else:
        return ''


def get_index_type(index: faiss.Index) -> str:
    for index_type, index_class in {
        'FlatIP': faiss.IndexFlatIP,
        'IVFFlat': faiss.IndexIVFFlat,
        'IVFPQ': faiss.IndexIVFPQ,
        'Flat': faiss.IndexFlat,
        'IVF': faiss.IndexIVF,
        'index': faiss.Index,
    }.items():
        if isinstance(index, index_class):
            return index_type
    return ''


def _get_from_queue(_queue: Queue):
    return _queue.get()


o_faiss = Faiss()
