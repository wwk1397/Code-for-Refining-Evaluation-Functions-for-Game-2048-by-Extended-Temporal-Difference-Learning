
import numpy as np

#generating shell for expectimax

CLUSTER_NAMES = [
    # "ei01",
    # "ei02",
    # "ei03",
    # "ei04",
    # "ei05",
    # "ei06",
    # "ei07",
    # "ei08",
    # "ei09",
    # "ei10",
    # "ei16",
    # "ei03",
    # "ei04",
    # "ei05",
    # "ei06",
    "ei07",
    # "ei08",
]
# add gpu_name
# GPU_USAGE = 0
GAME_NUMBER = 100
EXPECTIMAX_DEPTH = 2
METHOD_NAME = [
    # "RC_renew_best_simple_move",
    # "2ply_double",
    # "RC_double",
    # "2ply_double_keepB",
    # "RC_double_keepB",
    "2ply_double_s",
    # "2ply_double_ss",
]

THREAD_NUM = [
    # "1",
    # "2",
    # "3",
    # "4",
    # "5",
    # "6",
    "7",
    # "8",
]
WEIGHT_NUM = [f"{2000+10*i}" for i in range(1,13)]

GPU_AVAILABLE = [
    "-1",
    "0",
    "1",
]

# wait_num = [0,0,0,0]
# for num in range(8):
files = []
files_lenth = len(CLUSTER_NAMES)
wait_num = np.zeros((files_lenth))
for cluster in CLUSTER_NAMES:
    file =  open(cluster+".sh","w",encoding = "utf-8")
    files.append(file)

command_lis = []

class command_item():
    def __init__(self,
        seed="100",
        depth = f"D{EXPECTIMAX_DEPTH}",
        gpu = "0",
        modnum = "0",
        # modnnum is useless now
        thread = "1",
        weight = "1000",
        METHOD_NAME = "0",
        game_number = GAME_NUMBER,

    ):


        self.seed = seed
        self.depth = depth
        self.gpu = gpu
        self.modnum = modnum
        # modnnum is useless now
        self.thread = thread
        self.weight = weight
        self.game_number = game_number
        self.METHOD_NAME = METHOD_NAME

    def get_command(self):
        return f"nohup python expectimax_play_B.py {self.seed} {self.depth} {self.gpu} {self.modnum} {self.thread} {self.weight} {self.game_number} {self.METHOD_NAME} &\n"

    def __str__(self):
        return self.get_command()


# for weight in WEIGHT_NUM:
#     for thread in THREAD_NUM:
#         command_node = command_item(
#             depth=f"D{EXPECTIMAX_DEPTH}",
#             thread=thread,
#             weight=weight,
#             METHOD_NAME= str(METHOD_NAME)
#         )
#         command_lis.append(command_node)
for method_name in METHOD_NAME:
    for weight in WEIGHT_NUM:
        for thread in THREAD_NUM:

            command_node = command_item(
                depth=f"D{EXPECTIMAX_DEPTH}",
                thread=thread,
                weight=weight,
                METHOD_NAME= str(method_name),
            )
            command_lis.append(command_node)

cluster_item_sum = np.zeros((len(CLUSTER_NAMES),), dtype=int)
cluster_len = len(CLUSTER_NAMES)
for itr,command_node in enumerate(command_lis):
    current_cluster_number = itr%cluster_len
    current_gpu =  GPU_AVAILABLE[ cluster_item_sum[current_cluster_number]%(len(GPU_AVAILABLE)) ]
    #
    command_node.gpu = current_gpu
    # print(current_gpu,command_node.gpu)
    files[current_cluster_number].write(command_node.get_command())
    cluster_item_sum[current_cluster_number] = cluster_item_sum[current_cluster_number]+1
    if(EXPECTIMAX_DEPTH < 3):
        if(cluster_item_sum[current_cluster_number]% (6* len(GPU_AVAILABLE)) ==0):
            files[current_cluster_number].write("wait\n")
    else:
        if(cluster_item_sum[current_cluster_number]% (2* len(GPU_AVAILABLE) ) ==0):
            files[current_cluster_number].write("wait\n")
    





