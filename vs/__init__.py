import os
import sys
from typing import List, Any

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_cur_dir)

from vs.core.db import o_faiss
from vs.interfaces.index.index_add_vectors import index_add_vectors, VectorInput
from vs.interfaces.index.index_create import index_create
from vs.interfaces.index.index_delete_with_ids import index_delete_with_ids, IdsInput
from vs.interfaces.index.index_delete_with_info import index_delete_with_info, InfoInput
from vs.interfaces.index.index_update_with_info import index_update_with_info, UpdateInfoInput
from vs.interfaces.index.index_exist import index_exist
from vs.interfaces.index.index_list import index_list
from vs.interfaces.index.index_load import index_load
from vs.interfaces.index.index_release import index_release
from vs.interfaces.index.index_save import index_save
from vs.interfaces.index.index_search import index_search, SearchInput
from vs.interfaces.index.index_train import index_train, TrainVectorInput
from vs.interfaces.index.index_is_train import index_is_train
from vs.interfaces.index.index_info import index_info
from vs.server import server_run

sys.path.pop()


def train(index_name: str, vectors: List[List[float]], partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_train(TrainVectorInput(
        tenant=tenant,
        index_name=index_name,
        vectors=vectors,
        partition=partition
    ), log_id=log_id)


def add(
        index_name: str,
        vectors: List[List[float]],
        info: List[Any],
        partition: str = '',
        texts: List[Any] = None,
        filter_exist: bool = False,
        add_default_partition: bool = False,
        ret_id: bool = True,
        mv_partition: str = '',
        tenant: str = '_test',
        log_id: int = None,
):
    return index_add_vectors(VectorInput(
        tenant=tenant,
        index_name=index_name,
        vectors=vectors,
        info=info,
        partition=partition,
        mv_partition=mv_partition,
        texts=texts,
        filter_exist=filter_exist,
        add_default_partition=add_default_partition,
        ret_id=ret_id,
    ), log_id=log_id)


def create(
        index_name: str,
        dim_size: int,
        partition: str = '',
        count: int = 1000,
        tenant: str = '_test',
        n_list: int = None,
        log_id: int = None,
):
    return index_create(index_name, dim_size, partition, count, tenant, n_list, log_id=log_id)


def delete_with_ids(index_name: str,
                    ids: List[int],
                    partition: str = '',
                    tenant: str = '_test',
                    log_id: int = None):
    return index_delete_with_ids(IdsInput(
        tenant=tenant,
        index_name=index_name,
        ids=ids,
        partition=partition
    ), log_id=log_id)


def delete_with_info(index_name: str,
                     vectors: List[List[float]],
                     texts: List[str],
                     info: List[Any],
                     partition: str = '',
                     tenant: str = '_test',
                     log_id: int = None):
    return index_delete_with_info(InfoInput(
        tenant=tenant,
        index_name=index_name,
        vectors=vectors,
        texts=texts,
        info=info,
        partition=partition
    ), log_id=log_id)


def update_with_info(index_name: str,
                     vectors: List[List[float]],
                     texts: List[str],
                     old_info: List[Any] = None,
                     new_info: List[Any] = None,
                     partition: str = '',
                     tenant: str = '_test',
                     log_id: int = None):
    return index_update_with_info(UpdateInfoInput(
        tenant=tenant,
        index_name=index_name,
        vectors=vectors,
        texts=texts,
        old_info=old_info,
        new_info=new_info,
        partition=partition
    ), log_id=log_id)


def exist(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_exist(index_name, partition, tenant, log_id=log_id)


def is_train(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_is_train(index_name, partition, tenant, log_id=log_id)


def list_index(tenant: str = '_test', log_id: int = None):
    return index_list(tenant, log_id=log_id)


def load(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_load(index_name, partition, tenant, log_id=log_id)


def release(index_name: str, partition: str = '', tenant: str = '_test', _save: bool = True, log_id: int = None):
    return index_release(index_name, partition, tenant, save=_save, log_id=log_id)


def save(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_save(index_name, partition, tenant, log_id=log_id)


def search(
        index_names: List[str],
        vectors: List[List[float]],
        partitions: List[str] = None,
        nprobe: int = 10,
        top_k: int = 20,
        each_top_k: int = 20,
        use_mv: bool = True,
        tenants: List[str] = [],
        log_id: int = None):
    return index_search(SearchInput(
        tenants=tenants,
        index_names=index_names,
        vectors=vectors,
        partitions=partitions,
        nprobe=nprobe,
        top_k=top_k,
        each_top_k=each_top_k,
        use_mv=use_mv,
    ), log_id=log_id)


def info(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_info(index_name, partition, tenant, log_id=log_id)
