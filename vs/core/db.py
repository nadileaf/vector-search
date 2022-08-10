import copy
import os
import math
import faiss
import gevent
import threading
from queue import Queue
import numpy as np
from sqlitedict import SqliteDict
from typing import List, Union, Any
from six.moves import cPickle as pickle
from sklearn.metrics.pairwise import cosine_similarity
from vs.config.path import INDEX_DIR, SQLITE_DIR
from vs.lib.utils import md5, uid, get_relative_file
from vs.lib import logs


class Faiss:
    DEFAULT = '__default'

    def __init__(self):
        # 记录索引
        self.indices = {}

        # 记录 index 每个分区的滑动平均向量
        self.mv_indices = {}

    def index(self, tenant: str, index_name: str, partition: str = '') -> Union[None, faiss.Index]:
        if tenant not in self.indices or index_name not in self.indices[tenant] or \
                (partition and partition not in self.indices[tenant][index_name]):
            return
        partition = partition if partition else self.DEFAULT
        return self.indices[tenant][index_name][partition]

    @logs.log
    def train(self, tenant: str, index_name: str, vectors: np.ndarray, partition: str = '', log_id=None):
        index = self.index(tenant, index_name, partition)
        if index is not None:
            index.train(vectors)

    @logs.log
    def add(self,
            tenant: str,
            index_name: str,
            vectors: np.ndarray,
            texts: List[Any] = None,
            info: List[dict] = None,
            partition: str = '',
            filter_exist=False,
            add_default=False,
            mv_partition='',
            log_id=None) -> dict:
        """ 插入数据到 index，返回 插入成功的数量 insert_count """

        origin_len = len(vectors)
        partition = partition if partition else self.DEFAULT

        index = self.index(tenant, index_name, partition)
        index_type = get_index_type(index)

        # 预处理 info
        info = process_info(info, len(vectors), partition)

        if not filter_exist and index_type.startswith('Flat'):
            ids = list(range(index.ntotal, index.ntotal + origin_len))

            table_name = get_md5_table(tenant, index_name, partition, 'Flat')

            info = [''] * len(vectors) if not info else info
            md5_ids = list(map(md5, zip(texts if texts else vectors, info)))

            with _db(table_name) as d:
                for _i, mid in enumerate(md5_ids):
                    d[mid] = ids[_i]

        else:
            # 根据 index 类型 选用不同的 uid 函数
            if index_type.startswith('Flat'):
                id_queue = Queue()
                for i in range(origin_len):
                    id_queue.put(index.ntotal + i)
            else:
                id_queue = None

            # 获取数据的 id
            ids = get_uids(tenant, index_name, texts if texts else vectors, info, partition, id_queue, log_id=log_id)

        filter_ids = ids

        # 过滤重复数据
        if filter_exist:
            filter_indices = filter_duplicate(tenant, index_name, ids, partition, log_id=log_id)
            if not filter_indices:
                return {'count': 0, 'exist_count': origin_len, 'ids': ids}

            # 根据 filter_indices 取 不重复 的 数据
            filter_ids = [filter_ids[i] for i in filter_indices]
            info = [info[i] for i in filter_indices]
            vectors = vectors[filter_indices]

        # 添加 到 index
        if index_type.startswith('Flat'):
            index.add(vectors)
        else:
            index.add_with_ids(vectors, np.array(filter_ids))

        # 若有 partition，记录该 partition 的滑动平均向量
        if mv_partition or partition and partition != self.DEFAULT:
            partition = mv_partition if mv_partition else partition
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

        if add_default and partition and partition != self.DEFAULT:
            self.add(tenant, index_name, vectors, texts, info, self.DEFAULT, filter_exist, log_id=log_id)

        # 添加 具体 info 到 db
        with _db(get_table_name(tenant, index_name, partition)) as d:
            for _i, _id in enumerate(filter_ids):
                d[_id] = info[_i]

        return {'count': origin_len, 'exist_count': origin_len - len(filter_ids), 'ids': ids}

    @logs.log
    def list_info(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> list:
        partition = partition if partition else self.DEFAULT
        table_name = get_table_name(tenant, index_name, partition)
        if not os.path.exists(os.path.join(SQLITE_DIR, f'{table_name}.sqlite')):
            return []

        with _db(get_table_name(tenant, index_name, partition)) as d:
            # 限制返回内容，避免超内存
            if len(d) > 10000:
                _data = []
                for i, k in enumerate(d.keys()):
                    if i > 10000:
                        break
                    _data.append((k, d[k]))
            else:
                _data = list(d.items())
        return _data

    @logs.log
    def save_one(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> int:
        _index = self.index(tenant, index_name, partition)
        if _index is None:
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

            if partition == self.DEFAULT and tenant in self.mv_indices and index_name in self.mv_indices[tenant] and \
                    self.mv_indices[tenant][index_name] is not None:
                with open(get_relative_file(tenant, index_name, 'mv_index.pkl', root=INDEX_DIR), 'wb') as f:
                    pickle.dump(self.mv_indices[tenant][index_name], f)

        return 1

    @logs.log
    def save(self, tenant: str, log_id=None):
        """ 保存当前的所有索引到文件里 """
        if tenant not in self.indices:
            logs.add(log_id, 'save', f'tenant "{tenant}" 没有索引')
            return

        for index_name, index in self.indices[tenant].items():
            self.save_one(tenant, index_name, log_id=log_id)

    @logs.log
    def load_one(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> int:
        if not partition:
            index_dir = os.path.join(INDEX_DIR, tenant, index_name)
            if not os.path.isdir(index_dir) or not os.listdir(index_dir):
                return 0

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

        return 1

    @logs.log
    def load(self, tenant: str, log_id=None):
        """ 从文件中加载索引 """
        _tenant_dir = os.path.join(INDEX_DIR, tenant)
        if not os.path.isdir(_tenant_dir):
            return

        for index_name in os.listdir(_tenant_dir):
            self.load_one(tenant, index_name, log_id=log_id)

    @logs.log
    def release(self, tenant: str, index_name: str, partition: str = '', log_id=None) -> int:
        # release index 前，先保存索引
        ret = self.save_one(tenant, index_name, partition, log_id=log_id)
        if not ret:
            return 0

        if not partition:
            if tenant in self.indices and index_name in self.indices[tenant]:
                del self.indices[tenant][index_name]

        else:
            if tenant in self.indices and index_name in self.indices[tenant] and \
                    partition in self.indices[tenant][index_name]:
                del self.indices[tenant][index_name][partition]

        return 1

    @logs.log
    def search(self,
               vectors: np.ndarray,
               tenant: str,
               index_names: List[str],
               partitions: List[str] = None,
               nprobe=10,
               top_k=20,
               use_mv=True,
               log_id=None) -> List[List[dict]]:
        if vectors is None or not vectors.any():
            return []

        results = [[] for _ in range(len(vectors))]
        avg_results = [{} for _ in range(len(vectors))]
        d_table_name_2_ids = {}

        partitions = partitions if partitions else [''] * len(index_names)
        for i, index_name in enumerate(index_names):
            partition = partitions[i] if partitions[i] else self.DEFAULT
            self._search_a_index(tenant, index_name, partition, vectors, nprobe, top_k,
                                 avg_results, use_mv, d_table_name_2_ids, results, log_id=log_id)

        # 获取具体的结构化信息
        d_table_id_2_info = _get_info(d_table_name_2_ids, log_id=log_id)

        return _combine_results(results, avg_results, d_table_id_2_info, top_k, log_id=log_id)

    @logs.log
    def delete_with_id(self, ids: List[int], tenant: str, index_name: str, partition: str = '', log_id=None):
        partition = partition if partition else self.DEFAULT
        ids = list(filter(lambda x: x or x == 0, ids))

        with _db(get_table_name(tenant, index_name, partition)) as d:
            for _id in ids:
                if _id in d:
                    del d[_id]

    @logs.log
    def delete_with_info(self,
                         tenant: str,
                         index_name: str,
                         vectors: np.ndarray,
                         texts: List[Any] = None,
                         info: List[dict] = None,
                         partition: str = '',
                         log_id=None):
        partition = partition if partition else self.DEFAULT

        index = self.index(tenant, index_name, partition)
        index_type = get_index_type(index)

        # 预处理 info
        info = process_info(info, len(vectors), partition)

        table_name = get_md5_table(tenant, index_name, partition, 'Flat' if index_type.startswith('Flat') else 'IVF')
        md5_ids = list(map(md5, zip(texts if texts else vectors, info)))

        with _db(table_name) as d:
            ids = [d[mid] for mid in md5_ids if mid in d]

        self.delete_with_id(ids, tenant, index_name, partition, log_id=log_id)

    @logs.log
    def update_with_info(self,
                         tenant: str,
                         index_name: str,
                         vectors: np.ndarray,
                         texts: List[Any] = None,
                         old_info: List[dict] = None,
                         new_info: List[dict] = None,
                         partition: str = '',
                         log_id=None):
        partition = partition if partition else self.DEFAULT

        index = self.index(tenant, index_name, partition)
        index_type = get_index_type(index)

        texts = texts if texts else vectors

        # 预处理 info
        old_info = process_info(old_info, len(texts), partition)
        new_info = process_info(new_info, len(texts), partition)

        table_name = get_md5_table(tenant, index_name, partition, 'Flat' if index_type.startswith('Flat') else 'IVF')

        old_md5_ids = list(map(md5, zip(texts, old_info)))

        updated_ids = []
        updated_info = []

        with _db(table_name) as d:
            for _i, old_mid in enumerate(old_md5_ids):
                if old_mid not in d:
                    continue

                _uid = d[old_mid]
                new_mid = md5((texts[_i], new_info[_i]))
                d[new_mid] = _uid
                del d[old_mid]

                updated_ids.append(_uid)
                updated_info.append(new_info[_i])

        with _db(get_table_name(tenant, index_name, partition)) as d:
            for _i, _uid in enumerate(updated_ids):
                d[_uid] = updated_info[_i]

    @logs.log
    def _search_a_index(self,
                        tenant: str,
                        index_name: str,
                        partition: str,
                        vectors: np.ndarray,
                        nprobe: int,
                        top_k: int,
                        avg_results: List[dict],
                        use_mv: bool,
                        d_table_name_2_ids: dict,
                        results: List[list],
                        log_id=None):
        # 获取 index
        index = self.index(tenant, index_name, partition)
        if index is None:
            return

        table_name = get_table_name(tenant, index_name, partition)

        if use_mv and partition == self.DEFAULT and tenant in self.mv_indices and \
                index_name in self.mv_indices[tenant]:
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

        index.nprobe = nprobe

        D, I = index.search(vectors, top_k)

        d_table_name_2_ids[table_name] = list(set(list(map(int, I.reshape(-1)))))

        for _i, _result_ids in enumerate(I):
            similarities = D[_i]
            results[_i] += [
                {'id': _id, 'score': _similarity, 'table_name': table_name}
                for _id, _similarity in set(list(zip(_result_ids, similarities))) if _id != -1
            ]


def _db(table_name: str = None):
    """ 使用 sqlite 作为缓存 """
    return SqliteDict(os.path.join(SQLITE_DIR, f'{table_name}.sqlite'), tablename=table_name, autocommit=True)


def process_info(info: List[Any], length: int, partition: str = '') -> List[dict]:
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
        return int(math.sqrt(count) * 0.7)
    elif count <= 5000:
        return int(math.sqrt(count) * 0.9)
    elif count <= 15000:
        return int(math.sqrt(count) * 1.2)
    elif count <= 50000:
        return int(math.sqrt(count) * 1.5)
    else:
        return min(int(math.sqrt(count) * 2), 2048)


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


def get_uids_event(table_name: str, md5_ids: list, d_md5_2_id: dict, new_mids: list):
    with _db(table_name) as d:
        while md5_ids:
            mid = md5_ids.pop()
            if mid in d:
                d_md5_2_id[mid] = d[mid]
            else:
                new_mids.append(mid)


@logs.log
def get_uids(tenant: str,
             index_name: str,
             texts: Union[np.ndarray, List[Any]],
             info: List[Any] = None,
             partition: str = '',
             id_queue: Queue = None,
             log_id=None) -> List[int]:
    table_name = get_md5_table(tenant, index_name, partition, 'IVF' if id_queue is None else 'Flat')

    info = [''] * len(texts) if not info else info
    md5_ids = list(map(md5, zip(texts, info)))

    tmp_md5_ids = copy.deepcopy(md5_ids)
    new_mids = []
    d_md5_2_id = {}

    pool = []
    for i in range(min(20, len(tmp_md5_ids))):
        g = gevent.spawn(get_uids_event, table_name, tmp_md5_ids, d_md5_2_id, new_mids)
        pool.append(g)
    gevent.joinall(pool)

    with _db(table_name) as d:
        for mid in new_mids:
            _uid = uid() if id_queue is None else id_queue.get()
            d[mid] = _uid
            d_md5_2_id[mid] = _uid

    return [d_md5_2_id[mid] for mid in md5_ids]

    # ids = []
    # with _db(table_name) as d:
    #     for mid in md5_ids:
    #         if mid not in d:
    #             _uid = uid() if id_queue is None else id_queue.get()
    #             ids.append(_uid)
    #             d[mid] = _uid
    #         else:
    #             ids.append(d[mid])
    # return ids


@logs.log
def filter_duplicate(tenant: str, index_name: str, ids: List[int], partition: str = '', log_id=None) -> List[int]:
    """ 返回 没有重复(已存在db) 的数据的 位置index """
    if not ids:
        return []

    with _db(get_table_name(tenant, index_name, partition)) as d:
        return list(filter(lambda i: ids[i] not in d, range(len(ids))))


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


def _get_info_thread(ids: list, table_name: str, d_table_id_2_info: dict):
    if not ids:
        return

    with _db(table_name) as d:
        while ids:
            _id = ids.pop(0)
            if _id in d:
                table_id = f"{table_name}____{_id}"
                d_table_id_2_info[table_id] = d[_id]


@logs.log
def _get_info(d_table_name_2_ids: dict, log_id=None) -> dict:
    """ 获取具体的结构化信息 """
    d_table_id_2_info = {}

    pool = []
    for table_name, ids in d_table_name_2_ids.items():
        thread = threading.Thread(target=_get_info_thread, args=(ids, table_name, d_table_id_2_info))
        thread.start()
        pool.append(thread)

    for thread in pool:
        thread.join()

    return d_table_id_2_info


@logs.log
def _combine_results(results: List[list], avg_results: List[dict], d_table_id_2_info: dict, top_k: int, log_id=None):
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

            new_result.append({
                'data': data,
                'score': combine_avg_score(avg_similarity, val['score']),
                'mv_score': float(avg_similarity),
            })

        new_result.sort(key=lambda x: (-x['score'], -x['mv_score']))

        d_new_result = {}
        for v in new_result:
            k = f'{v}'
            if k not in d_new_result:
                d_new_result[k] = v
            if len(d_new_result) >= top_k:
                break

        results[_i] = list(d_new_result.values())

    return results


o_faiss = Faiss()
