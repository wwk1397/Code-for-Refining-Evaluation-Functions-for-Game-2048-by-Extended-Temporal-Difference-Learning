import os,sys
import re
import torch
import Game2048
import cnn22B as modeler
import playalg
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = modeler.Model().to(device0)

def get_weight_files(folder, pattern):
    files = os.listdir(folder)
    return [f for f in files if re.match(pattern, f)]

def calculate_average_ev(model, state, playalg, num_trials=100):
    total_ev = 0
    for _ in range(num_trials):
        state.initGame()
        _, ev = playalg.simplePlay(state, model)
        total_ev += ev
    return total_ev / num_trials

def map_location(storage, _):
    return storage.cuda(device0.index if torch.cuda.is_available() else 'cpu')

if len(sys.argv) > 2:
    folder = f"../exp/{sys.argv[1]}/training_cnn22B_{sys.argv[2]}/"
else:
    print("Usage: script.py <folder_path>")
    sys.exit(1)

pattern_A = r"weights-A-\d+"
pattern = r"weights-\d+"

if "double" in folder:
    pattern_used = pattern_A
else:
    pattern_used = pattern

weight_files = get_weight_files(folder, pattern_used)


for idx, weight_file in enumerate(weight_files):
    # if(weight_file.__contains__("100") == False) and (weight_file.__contains__("200") == False) and (weight_file.__contains__("300") == False):
    #     continue
    if(weight_file.__contains__("300") == False):
        continue
    if "log" in weight_file:
        continue
    if idx % 1 == 0:
        state = Game2048.State()
        state.initGame()
        # print("*****",os.path.join(folder, weight_file))
        model.load_state_dict(torch.load(os.path.join(folder, weight_file), map_location=map_location))
        avg_ev = calculate_average_ev(model, state, playalg)
        print(f"Average EV for {weight_file}: {avg_ev}")

