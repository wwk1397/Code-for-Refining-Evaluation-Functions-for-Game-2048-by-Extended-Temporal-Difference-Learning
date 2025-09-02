import os
import random
import torch
import numpy as np
import time
import multiprocessing
import Game2048
from cnn22B import Model, policy_value
from expectimax import Expectimax_setting, expectimax
import deep_play

# 配置参数
episodes_num = 10
seed = 100
maxdepth = 2  # D1
weight_path = "../exp/TD_afterstate_full_finetuning_l/training_cnn22B_1/weights-2000"  # 请根据实际情况修改
batch_size = 96

# 设置随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def run_on_gpu(gpu_id, result_queue):
    if gpu_id < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu_id}')

    # 加载模型
    policy_value_net = Model().to(device)
    policy_value_net.load_state_dict(torch.load(weight_path, map_location=device))

    # 设置Expectimax参数
    Expectimax_setting.nn_calls = 0
    Expectimax_setting.maxdepth = maxdepth
    Expectimax_setting.policy_value = policy_value

    action_count = 0
    episodes_data = []

    for i in range(episodes_num):
        start_time = time.time()
        start_action_count = action_count

        player = expectimax()
        state = Game2048.State()
        state.initGame()

        while True:
            # 调用expand_and_get一次视为一次动作
            action_count += 1
            move, ev ,node = deep_play.expand_and_get(
                state, policy_value_net, int(maxdepth) * 2 - 1,
                return_node=True,
                greedy_value=False, device_number=gpu_id,
                quick=False,
                # batch_size=batch_size,
                greedy_move=True,
            )

            state.play(move)
            state.putNewTile()
            if state.isGameOver():
                break

        end_time = time.time()
        end_action_count = action_count
        ep_actions = end_action_count - start_action_count
        ep_time = end_time - start_time
        ep_speed = ep_actions / ep_time if ep_time > 0 else 0
        episodes_data.append((i + 1, ep_actions, ep_time, ep_speed))

    # 统计第2-6局的平均速度
    selected_episodes = [ep for ep in episodes_data if 2 <= ep[0] <= 6]
    if selected_episodes:
        avg_speed = sum(x[3] for x in selected_episodes) / len(selected_episodes)
    else:
        avg_speed = 0.0

    # 将结果放入队列中
    result_queue.put((gpu_id, avg_speed))


if __name__ == '__main__':
    result_queue = multiprocessing.Queue()

    # 创建两个进程，分别在GPU0和GPU1上运行
    p0 = multiprocessing.Process(target=run_on_gpu, args=(0, result_queue))
    p1 = multiprocessing.Process(target=run_on_gpu, args=(1, result_queue))

    p0.start()
    p1.start()

    p0.join()
    p1.join()

    # 从队列中获取结果
    results = []
    while not result_queue.empty():
        gpu_id, avg_speed = result_queue.get()
        results.append((gpu_id, avg_speed))

    # 按照gpu_id排序并打印结果
    results.sort(key=lambda x: x[0])
    for gpu_id, avg_speed in results:
        print(f"GPU {gpu_id}: Average speed (episodes 2-6): {avg_speed:.2f} actions/s")
