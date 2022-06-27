import os
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
_root_dir = os.path.split(_cur_dir)[0]
sys.path.append(_root_dir)

from vs.lib import logs

logs.MODULE = 'vector-search'
logs.PROCESS = 'server'

from vs.interfaces.base import app
from vs.interfaces.index import index_add_vectors
from vs.interfaces.index import index_create
from vs.interfaces.index import index_load
from vs.interfaces.index import index_release
from vs.interfaces.index import index_save
from vs.interfaces.index import index_search
from vs.interfaces.index import index_train
from vs.interfaces.index import index_exist
from vs.interfaces.index import index_list
from vs.interfaces.index import index_train_batch
from vs.interfaces.index import index_delete_with_info
from vs.interfaces.index import index_delete_with_ids
from vs.interfaces.index import index_update_with_info
from vs.interfaces.index import index_is_train
from vs.interfaces.data import data_upload_sqlite
from vs.interfaces.data import data_upload_index

sys.path.pop()


def server_run(port: int = None):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    port = port if port else 80
    uvicorn.run(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    server_run()
