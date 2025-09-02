import numpy as np
import random
import torch
import Game2048
import os
from cnn22B import Model
import deep_play
import cnn22B as modeler
import re
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = modeler.Model().to(device0)
def get_init_value(state_number = 1000, device_number = 0, net = Model()):
    if(device_number == -1):
        device = torch.device(f'cpu')
    else:
        device = torch.device(f'cuda:{device_number}')

    net = net.to(device_number)
    value_lis = []

    for game_itr in range(state_number):
        state = Game2048.State()
        state.initGame()
        move, ev = deep_play.expand_and_get(state, net, 1, greedy_value=False, device_number=device_number, quick=True)
        value_lis.append(ev)

    return sum(value_lis)/len(value_lis)

def get_weight_files(folder, pattern):
    files = os.listdir(folder)
    return [f for f in files if re.match(pattern, f)]

def map_location(storage, _):
    return storage.cuda(device0.index if torch.cuda.is_available() else 'cpu')

def write_init_value(model_name = "", thread_name = ""):
    folder = f"../exp/{model_name}/training_cnn22B_{thread_name}/"
    pattern_A = r"weights-A-\d+"
    pattern = r"weights-\d+"

    if "double" in folder:
        pattern_used = pattern_A
    else:
        pattern_used = pattern

    weight_files = get_weight_files(folder, pattern_used)
    with open(f"{folder}init_evalue.log", "w") as file:
        pass

    for idx, weight_file in enumerate(weight_files):
        if "log" in weight_file:
            continue
        if (idx+1) % 100 == 0:
            # state = Game2048.State()
            # state.initGame()
            # # print("*****",os.path.join(folder, weight_file))
            model.load_state_dict(torch.load(os.path.join(folder, weight_file), map_location=map_location))
            avg_ev = get_init_value(
                state_number=1000,
                device_number=0,
                net=model,
            )
            # print(f"Average EV for {weight_file}: {avg_ev}")
            with open(f"{folder}_init_evalue.log", "a") as file:
                print(f"Average EV for {model_name} {weight_file} : {avg_ev}")
                file.write(f"{weight_file} {avg_ev}\n")
