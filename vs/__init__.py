import os
import sys
from typing import List, Any

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_cur_dir)

from core.db import o_faiss
from interfaces.index.index_add_vectors import index_add_vectors, VectorInput, Header
from interfaces.index.index_create import index_create
from interfaces.index.index_delete_with_ids import index_delete_with_ids, IdsInput
from interfaces.index.index_delete_with_info import index_delete_with_info, InfoInput
from interfaces.index.index_update_with_info import index_update_with_info, UpdateInfoInput
from interfaces.index.index_exist import index_exist
from interfaces.index.index_list import index_list
from interfaces.index.index_load import index_load
from interfaces.index.index_release import index_release
from interfaces.index.index_save import index_save
from interfaces.index.index_search import index_search, SearchInput
from interfaces.index.index_train import index_train, TrainVectorInput
from server import run

sys.path.pop()


def train(index_name: str, vectors: List[List[float]], partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_train(TrainVectorInput(
        index_name=index_name,
        vectors=vectors,
        partition=partition
    ), tenant=tenant, log_id=log_id)


def add(
        index_name: str,
        vectors: List[List[float]],
        info: List[Any],
        partition: str = '',
        texts: List[Any] = None,
        filter_exist: bool = False,
        add_default_partition: bool = False,
        ret_id: bool = True,
        tenant: str = '_test',
        log_id: int = None,
):
    return index_add_vectors(VectorInput(
        index_name=index_name,
        vectors=vectors,
        info=info,
        partition=partition,
        texts=texts,
        filter_exist=filter_exist,
        add_default_partition=add_default_partition,
        ret_id=ret_id,
    ), tenant=tenant, log_id=log_id)


def create(
        index_name: str,
        dim_size: int,
        partition: str = '',
        count: int = 1000,
        tenant: str = '_test',
        log_id: int = None,
):
    return index_create(index_name, dim_size, partition, count, tenant, log_id=log_id)


def delete_with_ids(index_name: str,
                    ids: List[int],
                    partition: str = '',
                    tenant: str = '_test',
                    log_id: int = None):
    return index_delete_with_ids(IdsInput(
        index_name=index_name,
        ids=ids,
        partition=partition
    ), tenant=tenant, log_id=log_id)


def delete_with_info(index_name: str,
                     vectors: List[List[float]],
                     texts: List[str],
                     info: List[Any],
                     partition: str = '',
                     tenant: str = '_test',
                     log_id: int = None):
    return index_delete_with_info(InfoInput(
        index_name=index_name,
        vectors=vectors,
        texts=texts,
        info=info,
        partition=partition
    ), tenant=tenant, log_id=log_id)


def update_with_info(index_name: str,
                     vectors: List[List[float]],
                     texts: List[str],
                     old_info: List[Any] = None,
                     new_info: List[Any] = None,
                     partition: str = '',
                     tenant: str = '_test',
                     log_id: int = None):
    return index_update_with_info(UpdateInfoInput(
        index_name=index_name,
        vectors=vectors,
        texts=texts,
        old_info=old_info,
        new_info=new_info,
        partition=partition
    ), tenant=tenant, log_id=log_id)


def exist(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_exist(index_name, partition, tenant, log_id=log_id)


def list_index(tenant: str = '_test', log_id: int = None):
    return index_list(tenant, log_id=log_id)


def load(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_load(index_name, partition, tenant, log_id=log_id)


def release(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_release(index_name, partition, tenant, log_id=log_id)


def save(index_name: str, partition: str = '', tenant: str = '_test', log_id: int = None):
    return index_save(index_name, partition, tenant, log_id=log_id)


def search(
        index_names: List[str],
        vectors: List[List[float]],
        partitions: List[str] = None,
        nprobe: int = 10,
        top_k: int = 20,
        use_mv: bool = True,
        tenant: str = '_test',
        log_id: int = None):
    return index_search(SearchInput(
        index_names=index_names,
        vectors=vectors,
        partitions=partitions,
        nprobe=nprobe,
        top_k=top_k,
        use_mv=use_mv,
    ), tenant=tenant, log_id=log_id)
