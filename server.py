import os
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
_root_dir = os.path.split(_cur_dir)[0]
sys.path.append(_root_dir)

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
from interfaces.index import index_train_batch

if __name__ == '__main__':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host='0.0.0.0', port=80 if ENV != DEV else 333)
