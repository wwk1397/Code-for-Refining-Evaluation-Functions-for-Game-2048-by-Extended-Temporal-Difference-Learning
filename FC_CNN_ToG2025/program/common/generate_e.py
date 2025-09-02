
import numpy as np

#generating shell for expectimax

CLUSTER_NAMES = [
    "ei01",
    # "ei02",
    # "ei03",
    # "ei04",
    # "ei07",
    # "ei08",
    # "ei09",
    # "ei10",
    # "ei16",
    # "ei03",
    # "ei04",
    # "ei05",
    # "ei06",
    # "ei07",
    # "ei08",
    # "ei15",
    # "ei16",
    # "ei16-2",
]
# add gpu_name
# GPU_USAGE = 0
GAME_NUMBER = 50
EXPECTIMAX_DEPTH = 3
METHOD_NAME = [
    # "1ply_l",
    # "1ply_l_s",
    # "RC_double",
    # "RC_double_s",
    # "RC_double_l",
    # "RC_double_l_s",

    # "RC_double_separate_s",
    # "RC_double_separate_l",
    # "RC_double_separate_l_s",
    # "2ply",
    # "2ply_l",
    # "2ply_s",
    # "2ply_l_s",
    # "RC",
    # "RC_s",
    # "RC_l",
    # "RC_l_s",


    # "1ply_s",
    # "RC_s"
    # "RC_renew_best_simple_move",
    # "2ply_double",
    # "2ply_double_s",
    # "2ply_double_l",
    # "2ply_double_l_s",
    # "2ply_double_separate",
    # "2ply_double_separate_l",
    # "2ply_double_separate_s",
    # "2ply_double_separate_l_s",
    # "RC_double_separate_l_s",

    # "RC_double",
    # "2ply_double_keepB",
    # "RC_double_keepB",

    # "2ply_double_ss",
    # "D1_weight",
    # "RC_best_l",
    # "RC_best_s",
    # "RC_best_l_s",
    # "D1_totest",
    # "RC_best_totest",
    # "2ply_double_sep_totest",
    # "2ply_double_join_totest",
    # "2ply_totest",

    # "2ply_from0",
    # "RC_best_from0",
    # "D1_totest",


    # "sm_dv",

    # "1ply_from0",
    # "2ply_double_separate_from0",
    # "2ply_double_from0",
    # "1ply_from0",
    # "RC_best_from0",
    # "D1_weight",
    # "sm_dv",
    # "dm_sv",

    # "2ply_TD_afterstate_finetuning",
    # "TD_afterstate_full_finetuning",
    # "50percent_2ply_TD_afterstate_finetuning",
    # "50percent_TD_afterstate_full_finetuning",
    # "2ply_TD_afterstate",
    # "TD_afterstate_full",
    # "50percent_2ply_TD_afterstate",
    # "50percent_TD_afterstate_full",
    # "2ply_TD_afterstate_deep",
    "50percent_2ply_TD_afterstate_l",
    "50percent_TD_afterstate_full_l",
    "2ply_TD_afterstate_l"
]

THREAD_NUM = [
    "1",
    "2",
    "3",
    "4",
    # "5",
    # "6",
    # "7",
    # "8",
]
# WEIGHT_NUM = ["200"]
# WEIGHT_NUM = ["50","150","250"]
WEIGHT_NUM = [f"{2000+50*i}" for i in range(1,4)]
# WEIGHT_NUM = [f"{0+50*i}" for i in range(1,11)]
# WEIGHT_NUM = [f"{0+100*i}" for i in range(1,21)]
# WEIGHT_NUM = [f"{0+20*i}" for i in range(11,16)]
# WEIGHT_NUM = [f"{0+20*i}" for i in range(1,5)]
# WEIGHT_NUM = [
#     "2020",
#     # "2080",
#     # "2100",
#     "2120",
#     # "2140",
#     # "2160",
#     "2180",
#     "2200",
#     # "2220",
#     # "2240",
#     "2260",
#     "2280",
#     "2300",
#               ]

GPU_AVAILABLE = [
    "0",
    "1",
    # "-1",
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
        return f"nohup python expectimax_play.py {self.seed} {self.depth} {self.gpu} {self.modnum} {self.thread} {self.weight} {self.game_number} {self.METHOD_NAME} &\n"

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
        if(cluster_item_sum[current_cluster_number]% (8* len(GPU_AVAILABLE)) ==0):
            files[current_cluster_number].write("wait\n")
    else:
        # if(cluster_item_sum[current_cluster_number]% (6* len(GPU_AVAILABLE) ) ==0):
        if (cluster_item_sum[current_cluster_number] % (7 * len(GPU_AVAILABLE)) == 0):
            files[current_cluster_number].write("wait\n")
    





