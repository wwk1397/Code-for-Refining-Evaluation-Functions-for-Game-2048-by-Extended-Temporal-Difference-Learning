import numpy as np
import os

CLUSTER_NAMES = [
    "ei01",
    # "ei02",
    # "ei03",
    # "ei04",
    # "ei07",
    # "ei08",
    # "ei09",
    # "ai01",
    # "ai02",
    # "ai03",
    # "ai04",
    # "ai07",
    # "ai08",
    # "ai09",
]

GAME_NUMBER = 75
EXPECTIMAX_DEPTH = 1
METHOD_NAME = [
    # "TD_afterstate_full_finetuning_l",
    "50percent_2ply_TD_afterstate_finetuning_l",
    "50percent_TD_afterstate_full_finetuning_l",
    "2ply_TD_afterstate_finetuning_l"
]

THREAD_NUM = [
    "1",
    "2",
    "3",
    "4",
]

WEIGHT_NUM = [f"{2000+50*i}" for i in range(1,7)]

GPU_AVAILABLE = [
    "0",
    "1",
]

files = []
files_length = len(CLUSTER_NAMES)
wait_num = np.zeros((files_length))
for cluster in CLUSTER_NAMES:
    file = open(cluster+".sh","w",encoding="utf-8")
    files.append(file)

command_lis = []

class command_item():
    def __init__(self,
                 seed="100",
                 depth = f"D{EXPECTIMAX_DEPTH}",
                 gpu = "0",
                 modnum = "0",
                 thread = "1",
                 weight = "1000",
                 METHOD_NAME = "0",
                 game_number = GAME_NUMBER):
        self.seed = seed
        self.depth = depth
        self.gpu = gpu
        self.modnum = modnum
        self.thread = thread
        self.weight = weight
        self.game_number = game_number
        self.METHOD_NAME = METHOD_NAME

    def get_command(self):
        return f"nohup python expectimax_play.py {self.seed} {self.depth} {self.gpu} {self.modnum} {self.thread} {self.weight} {self.game_number} {self.METHOD_NAME} &\n"

    def __str__(self):
        return self.get_command()

# 生成所有的命令节点
for method_name in METHOD_NAME:
    for weight in WEIGHT_NUM:
        for thread in THREAD_NUM:
            command_node = command_item(
                depth=f"D{EXPECTIMAX_DEPTH}",
                thread=thread,
                weight=weight,
                METHOD_NAME=str(method_name),
            )
            command_lis.append(command_node)

cluster_item_sum = np.zeros((len(CLUSTER_NAMES),), dtype=int)
cluster_len = len(CLUSTER_NAMES)

# 定义单GPU最大并发进程数
MAX_JOBS_PER_GPU = 7

for itr, command_node in enumerate(command_lis):
    current_cluster_number = itr % cluster_len
    current_gpu = GPU_AVAILABLE[ cluster_item_sum[current_cluster_number] % (len(GPU_AVAILABLE)) ]
    command_node.gpu = current_gpu

    # 使用awk精确匹配GPU ID所在字段（假设为$15）
    # 请根据您的进程参数实际情况调整该字段编号
    check_str = f"""
MAX_JOBS_PER_GPU={MAX_JOBS_PER_GPU}
GPU_ID={current_gpu}
while [ $(ps aux | grep "[p]ython expectimax_play.py" | awk -v gpu="$GPU_ID" '{{if ($15 == gpu) print $0}}' | wc -l) -ge $MAX_JOBS_PER_GPU ]; do
    # 输出当前进程数，便于调试
    CURRENT_JOBS=$(ps aux | grep "[p]ython expectimax_play.py" | awk -v gpu="$GPU_ID" '{{if ($15 == gpu) print $0}}' | wc -l)
    echo "GPU $GPU_ID is full, currently running $CURRENT_JOBS jobs, waiting..."
    sleep 10
done
"""

    files[current_cluster_number].write(check_str)
    files[current_cluster_number].write(command_node.get_command())
    cluster_item_sum[current_cluster_number] += 1

# 在脚本末尾等待所有后台任务结束（可选）
for f in files:
    f.write("wait\n")
    f.close()
