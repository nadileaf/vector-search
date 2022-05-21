import zipfile
from typing import Optional, Union
from fastapi import File, UploadFile, Header
from vs.interfaces.base import app, log
from vs.interfaces.definitions.common import Response


@app.post('/v1/data/upload',
          name="v1 data upload",
          response_model=Response,
          description="上传数据")
@log
def data_upload(
        sqlite_file: UploadFile = File(..., description="上传的源数据的文件"),
        index_file: UploadFile = File(..., description="上传的源数据的文件"),
        tenant: Optional[str] = Header('_test'),
        log_id: Union[int, str] = None
):
    # 判断文件类型
    sqlite_file_name = sqlite_file.filename
    index_file_name = index_file.filename

    # 读取数据
    sqlite_file_original_content = sqlite_file.file.read()
    index_file_original_content = index_file.file.read()
    #
    # # 解析数据
    # try:
    #     target_schema = json.loads(original_content)
    # except:
    #     ret = {'ret': 0, 'msg': f'"{file_name}" 不是合法的 json 文件'}
    #     logs.add(f'{log_id}', f'POST {logs.fn_name()}', f'return {ret}')
    #     return ret
    #
    # # 检查数据
    # if not isinstance(target_schema, dict):
    #     ret = {'ret': 0, 'msg': f'上传的目标schema需要是 dict 类型'}
    #     logs.add(f'{log_id}', f'POST {logs.fn_name()}', f'return {ret}')
    #     return ret
    #
    # if not target_schema:
    #     ret = {'ret': 0, 'msg': f'"{file_name}" 是空数据'}
    #     logs.add(f'{log_id}', f'POST {logs.fn_name()}', f'return {ret}')
    #     return ret
    #
    # # 展开数据
    # target_schema = expand_schema(target_schema)
    #
    # # task 唯一id
    # _id = get_full_task_id(project_id, task_id)
    #
    # # 保存原文件
    # file_path = utils.get_relative_file(_id.replace('.', '_'), file_name, root=ORIGIN_DIR)
    # file_url = f'/download/original_files/{_id.replace(".", "_")}/{file_name}'
    # with open(file_path, 'wb') as f:
    #     f.write(utils.encode_2_utf8(original_content))
    #
    # # 保存数据
    # with db.target() as d:
    #     d_data = d[_id] if _id in d else {}
    #     d_data['schema'] = target_schema
    #     d[_id] = d_data
    #
    # # 记录状态
    # with db.target_status() as d:
    #     d_data = d[_id] if _id in d else {}
    #     d_data['schema'] = {
    #         'status': 1,
    #         'search_is_updated': 0,
    #         'is_parsed': 0,
    #         'to_std': '',
    #         'file_name': file_name,
    #         'file_url': file_url,
    #         'modified_time': str(time.strftime('%Y-%m-%d %H:%M:%S')),
    #     }
    #     d[_id] = d_data
    #
    # ret = {'ret': 1}
    # logs.add(f'{log_id}', f'POST {logs.fn_name()}', f'return {ret}')
    return {'code': 1}
