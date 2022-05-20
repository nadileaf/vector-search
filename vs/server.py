import os
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_cur_dir)

from lib import logs

logs.MODULE = 'vector-search'
logs.PROCESS = 'server'

from config.env import ENV, DEV
from interfaces.base import app
from interfaces.index import index_add_vectors
from interfaces.index import index_create
from interfaces.index import index_load
from interfaces.index import index_release
from interfaces.index import index_save
from interfaces.index import index_search
from interfaces.index import index_train
from interfaces.index import index_exist
from interfaces.index import index_list
from interfaces.index import index_train_batch
from interfaces.index import index_delete_with_info
from interfaces.index import index_delete_with_ids
from interfaces.index import index_update_with_info

sys.path.pop()


def run():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host='0.0.0.0', port=80 if ENV != DEV else 333)


if __name__ == '__main__':
    run()
