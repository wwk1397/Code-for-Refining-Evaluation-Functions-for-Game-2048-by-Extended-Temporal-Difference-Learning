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


for game_number in range(1):

    state = Game2048.State()
    state.initGame()
    turn = 0

    while(True):
        turn += 1
        dir, ev, root_node = deep_play.expand_and_get(state, model, 5, return_node=True, device_number=0, quick=True)
        state.play(dir)
        state.putNewTile()
        print(f"Turn:{turn}")
        act_number[dir] = act_number[dir] + 1
        if state.isGameOver():
            print(f"GameOver {game_number}")
            break

print(act_number)
