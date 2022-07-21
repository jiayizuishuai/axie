import os
model_history_path = './models/12315'
is_isexist = os.path.exists(model_history_path)

if not is_isexist :
    os.makedirs(model_history_path)
    print('创建成功')
else :
    print('已存在路径')