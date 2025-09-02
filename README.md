Refining Evaluation Functions for Game 2048 by Extended Temporal Difference Learning

This repository provides the source code for the paper:

Refining Evaluation Functions for Game 2048 by Extended Temporal Difference Learning
Weikai Wang and Kiminori Matsuzaki
Accepted in IEEE Transactions on Games, 2025.

🔧 Environment

Python 3.11.5

PyTorch 2.1.1

🚀 Training

Run training with:

python double_gpu_s.py {json_file_name} {thread_and_seed_number} {GPU_number}


Available JSON configs:

1ply_from0 → baseline

TD_afterstate_full → TDA-full

TD_afterstate_full_finetuning → TDA-full-refining

2ply_TD_afterstate → TDA-2ply

2ply_TD_afterstate_finetuning → TDA-2ply-refining

Example:

cd program/exp
python double_gpu_s.py 1ply_from0 4 0

🎮 Evaluation (Greedy / Expectimax)

Run evaluation with:

python expectimax_play.py {seed_number} D{depth_number} {GPU_number} {model_name} {thread_number} {weight_number} {game_number} {method_name}


Example:

cd program/common
python expectimax_play.py 100 D3 0 0 1 2050 50 2ply_TD_afterstate


generate_e.py can be used to generate evaluation codes.

📜 License


This project is licensed under the MIT License
