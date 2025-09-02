#generating shell for greedy
# usage: python g1.py 100 cnn22B 0 1 1 10
import numpy as np
# mod_name = [
#     "AE-CNN",
#     "BP-CNN",
#     "CNN",
#     "CNN-CNN",
# ]


cluster_name = [
    # "i01",
    # "i02",
    # "i02_2",
    # "i03",
    # "i04_2",
    # "i05_2",
    # "i06_2",
    # "i04",
    # "i05",
    # "i06",
    "i07",
    "i08",
]

gpu_name = 1
# training_name = "RC_shuffle_half_double_static_value_nostart"
# training_name = "D3_shuffle_original_double_another"
# training_name = "D3_shuffle_half_double_move"
# training_name ="D3_shuffle_half_double_move_simpleM"
# training_name ="D3_shuffle_half_double_value_simpleM"
# training_name ="D3_shuffle_half_double_value_simpleV"


# training_name ="D3_shuffle_half_double_move_simpleV"

# training_name = "D3_shuffle_PD1_100"
# training_name = "D3_shuffle_PD1_80"
# training_name = "D3_shuffle_PD1_60"
# training_name = "D3_shuffle_PD1_40"
# training_name = "D3_shuffle_PD1_20"
# training_name = "D3_shuffle_PD1_0"
# training_name = "D3_shuffle_PD1_10"
# training_name = "D3_shuffle_PD1_5"
# training_name = "D3_shuffle_PD1_2"

mod_name = [
    "2",
    # "0",
    # "3",
]

# pat_name = [
#     "../kmatsu_expectimaxC/",
#     "../kmatsu_expectimaxAE/",
#     "../kmatsu_expectimaxCC/",
# ]


# thread_name = [
#     "1",
#     "2",
#     "3",
#     "4",
# ]

weight_num = [f"{1000+20*i}" for i in range(19)]

# weight_num = [
    # "1000",
    # "1005",
    # "1010",
    # "1015",
    # "1020",
    # "1025",
    # "1050",
    # "1060",
    # "1070",
    # "1080",
    # "1090",
    # "1100",
    # "1110",
    # "1120",
    # "1130",
    # "1140",
    # "1150",

# ]
# training_name = "deep_move_greedy_value_from1000"
# training_name = "deep_move_1_from1000"
# training_name = "simple_play_from1000"
# training_name = "renew_children_shuffle_separate"
# training_name = "deep_move_3_from1000"
# training_name = "D1_shuffle_half_double_another"

# weight_num = [
#     "50",
#     "100",
#     "150",
#     "200",
#     "250",
#     "300",
#     "350",
#     "400",
#     # "450",
#     # "500",
#     # "550",
#     # "600",
#     # "650",
#     # "700",
#     # "750",
#     # "800",
#     # "850",
#     # "900",
#     # "950",
#     # "1000",
# ]
# wait_num = [0,0,0,0]
# for num in range(8):
files = []
files_lenth = len(cluster_name)
wait_num = np.zeros((files_lenth))
for cluster in cluster_name:
    file =  open(cluster+".sh","w",encoding = "utf-8")
    files.append(file)


#python g1.py 100 cnn22B 0 1 1 10
# python g1.py 100        cnn22B modname thread weights 300

now_num = 0
for mod_num in mod_name:
    for thread_num in range(1,2):
        for t in range(len(weight_num)):
            if (gpu_name == 0):
                str1 = "nohup python "+ "greedyplay.py" +  f" 100 cnn22B ../exp/{training_name}/training_cnn22B_{thread_num}/weights-1-"+ weight_num[t]+  " 100 &\n"
            else:
                str1 = "nohup python " + "greedyplay_G1.py" + f" 100 cnn22B ../exp/{training_name}/training_cnn22B_{thread_num}/weights-1-" + weight_num[t] + " 100 &\n"
            # str2 = "greedyplay.py 100 cnn22B ../exp/training_cnn22B_1/weights-10 20"
            k = now_num % files_lenth
            files[k].write(str1)
            now_num +=1
            wait_num[k]+=1
            if(wait_num[k] == 4):
                str3 = "wait\n"
                files[k].write(str3)
                wait_num[k] = 0


# for k in range(4):
#     file = open(cluster_name[k]+".sh","w",encoding = "utf-8")
#     # mod1 = "AE-CNN-3"
#     # mod2 = "AE-CNN-3"
#     # mod3 = "AE-CNN-4"
#     # mod4 = "AE-CNN-4"
#     thread_num = k+1
#     thread_num = str(thread_num)
#
#     for i in range (3):
#
#         for t in range(10):
#
#             str1 = "nohup python "+ pat_name[i] + "expectimax_play.py" +  " 100 D3 "+str(t%2)+" "  + mod_name[i]+ " "+ thread_num +" " + weight_num[t]+  " &\n"
#             # str2 = "python e1.py 100 cnn22B 0 1 " + str(50*i-25) + " 3" + " 20 &\n"
#             file.write(str1)
#             wait_num[k]+=1
#             if(wait_num[k] == 8):
#                 str3 = "wait\n"
#                 file.write(str3)
#                 wait_num[k] = 0



