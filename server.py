import os
import sys
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

_cur_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(_cur_dir)

from vs.lib import logs

logs.MODULE = 'vector-search'
logs.PROCESS = 'server'

from vs.config.env import ENV, DEV
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

if __name__ == '__main__':
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    uvicorn.run(app, host='0.0.0.0', port=80 if ENV != DEV else 333)
