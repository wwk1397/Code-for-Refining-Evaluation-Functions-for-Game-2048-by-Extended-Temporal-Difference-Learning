import os,sys

#Input of different weight has different "snake order", we should test how to rotate and flip.

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import cnn22B as modeler
import Game2048
import numpy as np
import deep_play

device0 = torch.device(f"cuda:0")
model = modeler.Model().to(device0)
checkpointprefix = "./weights-1000"
# checkpointprefix = "./RC/weights-1-1001"
# checkpointprefix = "./RC/weights-0-1001"
# checkpointprefix = "./2ply/weights-1-1001"
# checkpointprefix = "./2ply/weights-0-1001"
# checkpointprefix = "./2ply/weights-1-1065"
# checkpointprefix = "./2ply/weights-0-1065"
model.load_state_dict(torch.load(checkpointprefix, map_location=device0))

# print(model.DIM_I)

act_number = [0, 0, 0, 0]
actsum_number = []


# for game_number in range(2):
#
#     state = Game2048.State()
#     state.initGame()
#     turn = 0
#
#     while(True):
#         turn += 1
#         dir, ev, root_node = deep_play.expand_and_get(state, model, 1, return_node=True, device_number=0)
#         state.play(dir)
#         state.putNewTile()
#         print(f"Turn:{turn}")
#         actsum_number.append(ev)
#         if state.isGameOver():
#             print(f"GameOver {game_number}")
#             break

for game_number in range(5000):

    state = Game2048.State()
    state.initGame()

    dir, ev, root_node = deep_play.expand_and_get(state, model, 1, return_node=True, device_number=0)
    actsum_number.append(ev)


print(sum(actsum_number)/len(actsum_number))

# state = Game2048.State()
# state.initGame()
# dir, ev, root_node = deep_play.expand_and_get(state, model, 1, return_node=True, device_number=0)
# print(ev)



