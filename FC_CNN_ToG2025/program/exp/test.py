# import json
#
#
#
#
# def load_parameters(file_path):
#     with open(file_path, 'r') as file:
#         parameters = json.load(file)
#     return parameters
#
# # 使用该函数加载参数
# file_path = '2ply_double.json'  # JSON文件的路径
# training_mod = load_parameters(file_path)
#
# # 现在 training_mod 包含了 JSON 文件中的参数
# print(training_mod)
a = {
    "123":10,
    "256": False,
}
if not (a.get("256")):
    print(a.get("256"))