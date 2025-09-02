# get value+e from expectimax in training.
import numpy as np
import torch

import cnn22B as modeler
from Game2048 import State
from typing import List, Dict
import copy
import random
import rotation_flip_match as rfm

INPUT_SIZE_A = modeler.INPUT_SIZE
INPUT_SIZE_B = modeler.INPUT_SIZE


class double_exp_node(object):
    def __init__(self, state: State, height: int, type="state", possibility=1.0):
        self.children = []
        self.type = type
        self.exp_A = -float("Inf")
        self.exp_B = -float("Inf")
        self.exp_A_B = -float("Inf")
        # exp_A_B means exp_A + exp_B. Only been calculated from root node.
        self.exp_Ab = -float("Inf")
        # a and b means a* and b*, which are same as the paper "Double Q learning"
        self.exp_Ba = -float("Inf")
        self.value_A = -float("Inf")
        self.value_B = -float("Inf")
        self.height = height
        self.state = state
        self.expanded = False
        self.possibility = possibility
        self.game_over = False
        self.available_actions = []
        self.next_move_A = -1
        self.next_move_B = -1
        self.next_move_A_B = -1
        self.next_simple_move_A = -1
        self.next_simple_move_B = -1
        self.next_simple_move_A_B = -1
        # best move from exp_A_B. Only been calculated from root node
        self.next_node_A = None
        self.next_node_B = None
        self.next_node_A_B = None
        self.next_simple_node_A = None
        self.next_simple_node_B = None
        self.next_simple_node_A_B = None
        self.real_target = -float("Inf")


def double_expand(current_node: double_exp_node, bottom_nodes: List[double_exp_node], shallow_value = False, random_state=False) -> List[
    double_exp_node]:
    current_board_state = current_node.state
    # print("**", current_node.state is None)
    if (current_node.state.isGameOver()):
        current_node.game_over = True
        current_node.value_A = 0
        current_node.value_B = 0
    elif (current_node.height == 0):
        bottom_nodes.append(current_node)
        current_node.game_over = True
    elif (shallow_value == True) and (current_node.type == "after_state"):
        bottom_nodes.append(current_node)

    elif current_node.type == "state":
        # "state", but not the end_nodes

        for direction in range(4):
            if current_board_state.canMoveTo(direction):
                child_board_state = current_board_state.clone()
                child_board_state.play(direction)
                current_node.available_actions.append(direction)
                child_node = double_exp_node(state=child_board_state, height=current_node.height - 1,
                                             type="after_state")
                current_node.children.append(child_node)

    else:
        # "after_state", but not the end_nodes
        if not random_state:
            for position in range(16):
                if current_node.state.board[position] != 0:
                    continue
                add_double_child_node(1, position, 0.9, current_node)
                add_double_child_node(2, position, 0.1, current_node)
        else:
            # "random after->state"
            children_position_lis = []
            for position in range(16):
                if current_node.state.board[position] != 0:
                    continue
                children_position_lis.append(position)
            children_position_len = len(children_position_lis)

            position_random = random.randint(0, children_position_len - 1)
            child_position = children_position_lis[position_random]
            tile_random = random.random()
            if (tile_random >= 0.9):
                add_double_child_node(2, child_position, 1, current_node)
            else:
                add_double_child_node(1, child_position, 1, current_node)

    return current_node.children


def add_double_child_node(tile_value: int, position: int, possibility: float, node: double_exp_node):
    child_board_state = node.state.clone()
    child_board_state.board[position] = tile_value
    child_node = double_exp_node(state=child_board_state, height=node.height - 1,
                                 type="state", possibility=possibility)
    node.children.append(child_node)


def calculate_double_value(board: np.ndarray, model_A, model_B,device_0_number = 0, device_1_number = 1,  flip_number = None) -> (np.ndarray, np.ndarray):
    # get value for board
    # board : np.array (seq_len,16)
    board_lenth = board.shape[0]
    input_boards_A = np.zeros((board_lenth, INPUT_SIZE_A), dtype="float32")
    input_boards_B = np.zeros((board_lenth, INPUT_SIZE_B), dtype="float32")

    # if(flip_number != None):
    #     input_boards_B = rfm.transform_2d(input_boards_B,flip_number[0],flip_number[1])

    for itr in range(board_lenth):
        model_A.make_input(input_boards_A[itr, :], board[itr])
        if(flip_number is None):
            model_B.make_input(input_boards_B[itr, :], board[itr])
        else:
            model_B.make_input(input_boards_B[itr, :], rfm.transform_2d(board[itr],flip_number[0],flip_number[1])  )
    answer_A = model_A.predict(input_boards_A, device_0_number).cpu().detach()
    answer_B = model_B.predict(input_boards_B, device_1_number).cpu().detach()

    return answer_A, answer_B


def quick_calculate_double(board: np.ndarray, model_A, model_B,device_0_number = 0, device_1_number = 1,  flip_number = None) -> (np.ndarray, np.ndarray):

    board_length = board.shape[0]
    input_boards_A = np.zeros((board_length, INPUT_SIZE_A), dtype="float32")
    input_boards_B = np.zeros((board_length, INPUT_SIZE_B), dtype="float32")

    for itr in range(board_length):
        model_A.make_input(input_boards_A[itr, :], board[itr])
        if(flip_number is None):
            model_B.make_input(input_boards_B[itr, :], board[itr])
        else:
            model_B.make_input(input_boards_B[itr, :], rfm.transform_2d(board[itr],flip_number[0],flip_number[1]) )

    if torch.is_tensor(input_boards_A) == False:
        input_boards_A = torch.from_numpy(input_boards_A).float()
    if torch.is_tensor(input_boards_B) == False:
        input_boards_B = torch.from_numpy(input_boards_B).float()

    with torch.no_grad():
        if device_0_number is not None:
            device_0 = torch.device(f"cuda:{device_0_number}")
        else:
            device_0 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if device_1_number is not None:
            device_1 = torch.device(f"cuda:{device_1_number}")
        else:
            device_1 = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        input_boards_A = input_boards_A.to(device_0)
        input_boards_B = input_boards_B.to(device_1)
        answer_A = model_A.predict(input_boards_A, device_0_number).cpu().detach()
        answer_B = model_B.predict(input_boards_B, device_1_number).cpu().detach()
    # return answer.numpy()
    return answer_A, answer_B


def get_double_node_exp_first(
        current_node: double_exp_node,
        discount_factor=1,
        random_state=False,
        greedy_move=False,
        normalized = False,
        average_score = 1

) -> (float, float):
    # get exp_A and exp_B
    # we will get exp_Ab, exp_Ba from get_double_node_exp_second
    if (current_node.game_over == True):
        if (current_node.value_B == -float("inf") or current_node.value_A == -float("inf")):
            raise Exception(f"One end node value is not calculated at height {current_node.height}")
        current_node.exp_A = discount_factor * current_node.value_A
        current_node.exp_B = discount_factor * current_node.value_B
        return current_node.exp_A, current_node.exp_B

    if (current_node.type == "after_state"):
        children_exp_reward_lis_A = []
        children_exp_reward_lis_B = []
        for child_node in current_node.children:
            exp_A, exp_B = get_double_node_exp_first(child_node, discount_factor, random_state, greedy_move, normalized, average_score)
            children_exp_reward_lis_A.append(child_node.possibility * exp_A)
            children_exp_reward_lis_B.append(child_node.possibility * exp_B)
        if not random_state:
            current_node.exp_A = 2 * sum(children_exp_reward_lis_A) / len(children_exp_reward_lis_A)
            current_node.exp_B = 2 * sum(children_exp_reward_lis_B) / len(children_exp_reward_lis_B)
        else:
            current_node.exp_A = sum(children_exp_reward_lis_A) / len(children_exp_reward_lis_A)
            current_node.exp_B = sum(children_exp_reward_lis_B) / len(children_exp_reward_lis_B)
        return current_node.exp_A, current_node.exp_B

    if (current_node.type == "state"):

        best_act_A = -1
        best_act_B = -1
        best_act_AB = -1
        best_node_A = None
        best_node_B = None
        best_node_AB = None
        best_value_A = -float("inf")
        best_value_B = -float("inf")
        best_value_AB = -float("inf")
        best_simple_act_A = -1
        best_simple_act_B = -1
        best_simple_act_AB = -1
        best_simple_node_A = None
        best_simple_node_B = None
        best_simple_node_AB = None
        best_simple_value_A = -float("inf")
        best_simple_value_B = -float("inf")
        best_simple_value_AB = -float("inf")

        for itr, child_node in enumerate(current_node.children):
            child_exp_A, child_exp_B = get_double_node_exp_first(child_node, discount_factor, random_state, greedy_move, normalized, average_score)

            local_reward = child_node.state.score - current_node.state.score
            if (normalized):
                local_reward = local_reward / average_score
            children_exp_reward_A = child_exp_A + local_reward
            children_exp_reward_B = child_exp_B + local_reward
            children_exp_reward_AB = (children_exp_reward_A + children_exp_reward_B) / 2
            if (child_node.value_A == -float("inf")):
                raise Exception("We haven't got value for first layer of children")


            children_value_reward_A = child_node.value_A + local_reward
            children_value_reward_B = child_node.value_B + local_reward
            children_value_reward_AB = (children_value_reward_A + children_value_reward_B) / 2

            if (greedy_move):
                if (child_node.value == -float("Inf")):
                    raise Exception("simple move but deeper than 3")
                local_reward = child_node.state.score - current_node.state.score
                if (normalized):
                    local_reward = local_reward/average_score

                children_exp_reward_A = local_reward + child_node.value_A
                children_exp_reward_B = local_reward + child_node.value_B
                children_exp_reward_AB = (children_exp_reward_A + children_exp_reward_B) / 2
                children_value_reward_A = child_node.value_A + local_reward
                children_value_reward_B = child_node.value_B + local_reward
                children_value_reward_AB = (children_value_reward_A + children_value_reward_B) / 2

            if (children_exp_reward_A > best_value_A):
                best_act_A = current_node.available_actions[itr]
                best_value_A = children_exp_reward_A
                best_node_A = child_node
            if (children_exp_reward_B > best_value_B):
                best_act_B = current_node.available_actions[itr]
                best_value_B = children_exp_reward_B
                best_node_B = child_node
            if (children_exp_reward_AB > best_value_AB):
                best_act_AB = current_node.available_actions[itr]
                best_value_AB = children_exp_reward_AB
                best_node_AB = child_node
            if (children_value_reward_A > best_simple_value_A):
                best_simple_act_A = current_node.available_actions[itr]
                best_simple_value_A = children_value_reward_A
                best_simple_node_A = child_node
            if (children_value_reward_B > best_simple_value_B):
                best_simple_act_B = current_node.available_actions[itr]
                best_simple_value_B = children_value_reward_B
                best_simple_node_B = child_node
            if (children_value_reward_AB > best_simple_value_AB):
                best_simple_act_AB = current_node.available_actions[itr]
                best_simple_value_AB = children_value_reward_AB
                best_simple_node_AB = child_node
            # print(children_exp_reward, ",", best_act,",",best_value)

        current_node.next_move_A = best_act_A
        current_node.next_move_B = best_act_B
        current_node.next_move_A_B = best_act_AB
        current_node.next_node_A = best_node_A
        current_node.next_node_B = best_node_B
        current_node.next_node_A_B = best_node_AB
        current_node.next_simple_move_A = best_simple_act_A
        current_node.next_simple_move_B = best_simple_act_B
        current_node.next_simple_move_A_B = best_simple_act_AB
        current_node.next_simple_node_A = best_simple_node_A
        current_node.next_simple_node_B = best_simple_node_B
        current_node.next_simple_node_A_B = best_simple_node_AB
        # print("best_act:",best_act)
        current_node.exp_A = best_value_A
        current_node.exp_B = best_value_B
        current_node.exp_A_B = best_value_AB
        return current_node.exp_A, current_node.exp_B


def get_double_node_exp_second(
        current_node: double_exp_node,
        discount_factor=1,
        act_1="deep",
        act_2="simple",
        random_state=False,
        greedy_move=False,
        renew_children=False,
        normalized = False,
        average_score = 1
) -> (float, float):
    # get exp_Ab and exp_Ba we have got next_move_A,next_move_B,next_node_A,next_node_B,exp_A,exp_B for all nodes
    # with get_double_node_exp_first

    if (current_node.game_over == True):
        if (current_node.value_B == -float("inf") or current_node.value_A == -float("inf")):
            raise Exception(f"One end node value is not calculated at height {current_node.height}")
        current_node.exp_Ab = discount_factor * current_node.value_A
        current_node.exp_Ba = discount_factor * current_node.value_B
        return current_node.exp_Ab, current_node.exp_Ba

    if (current_node.type == "after_state"):
        children_exp_reward_lis_Ab = []
        children_exp_reward_lis_Ba = []
        for child_node in current_node.children:
            exp_Ab, exp_Ba = get_double_node_exp_second(
                current_node=child_node,
                discount_factor=discount_factor,
                act_1=act_1,
                act_2=act_2,
                random_state=random_state,
                greedy_move=greedy_move,
                renew_children=renew_children,
                normalized= normalized,
                average_score= average_score,
            )
            children_exp_reward_lis_Ab.append(child_node.possibility * exp_Ab)
            children_exp_reward_lis_Ba.append(child_node.possibility * exp_Ba)
        if not random_state:
            current_node.exp_Ab = 2 * sum(children_exp_reward_lis_Ab) / len(children_exp_reward_lis_Ab)
            current_node.exp_Ba = 2 * sum(children_exp_reward_lis_Ba) / len(children_exp_reward_lis_Ba)
        else:
            current_node.exp_Ab = sum(children_exp_reward_lis_Ab) / len(children_exp_reward_lis_Ab)
            current_node.exp_Ba = sum(children_exp_reward_lis_Ba) / len(children_exp_reward_lis_Ba)
        return current_node.exp_Ab, current_node.exp_Ba

    if (current_node.type == "state"):

        if (act_1 != "deep") and (act_1 != "simple"):
            raise Exception(f"Act_1 is wrong with act_1: {act_1}")
        if (act_2 != "simple") and (act_2 != "no"):
            raise Exception(f"Act_2 is wrong with act_2: {act_2}")

        if (current_node.height == 3):
            # Do with root_state
            if (act_1 == "deep"):
                best_next_node_A = current_node.next_node_A
                best_next_node_B = current_node.next_node_B
                best_next_node_AB = current_node.next_node_A_B
            else:
                best_next_node_A = current_node.next_simple_node_A
                best_next_node_B = current_node.next_simple_node_B
                best_next_node_AB = current_node.next_simple_node_A_B

            exp_Ab, _ = get_double_node_exp_second(
                current_node=best_next_node_B,
                discount_factor=discount_factor,
                act_1=act_1,
                act_2=act_2,
                random_state=random_state,
                greedy_move=greedy_move,
                renew_children=renew_children,
                normalized = normalized,
                average_score = average_score,
            )
            _, exp_Ba = get_double_node_exp_second(
                current_node=best_next_node_A,
                discount_factor=discount_factor,
                act_1=act_1,
                act_2=act_2,
                random_state=random_state,
                greedy_move=greedy_move,
                renew_children=renew_children,
                normalized=normalized,
                average_score=average_score,
            )

            local_reward_B = best_next_node_B.state.score - current_node.state.score
            local_reward_A = best_next_node_A.state.score - current_node.state.score
            if (normalized):
                local_reward_B = local_reward_B / average_score
                local_reward_A = local_reward_A /average_score

            current_node.exp_Ab = exp_Ab + local_reward_B
            current_node.exp_Ba = exp_Ba + local_reward_A

            if (renew_children == True):
                for child_node in current_node.children:
                    if (child_node is best_next_node_A) or (child_node is best_next_node_B):
                        continue
                    else:
                        _, _ = get_double_node_exp_second(
                            current_node=child_node,
                            discount_factor=discount_factor,
                            act_1=act_1,
                            act_2=act_2,
                            random_state=random_state,
                            greedy_move=greedy_move,
                            renew_children=renew_children,
                            normalized=normalized,
                            average_score=average_score,
                        )

                local_reward_B = best_next_node_B.state.score - current_node.state.score
                local_reward_A = best_next_node_A.state.score - current_node.state.score
                if (normalized):
                    local_reward_B = local_reward_B / average_score
                    local_reward_A = local_reward_A / average_score

                current_node.exp_Ab = exp_Ab + local_reward_B
                current_node.exp_Ba = exp_Ba + local_reward_A

            return current_node.exp_Ab, current_node.exp_Ba
        else:
            # Do with node.height = 1
            if (act_2 == "simple"):
                best_next_node_A = current_node.next_node_A
                best_next_node_B = current_node.next_node_B
                best_next_node_AB = current_node.next_node_A_B

                if (best_next_node_A is None) or (best_next_node_B is None):
                    # print(f"best_next_node_A is None?{best_next_node_A is None}")
                    # print(f"best_next_node_B is None?{best_next_node_B is None}")
                    raise Exception(
                        f"best_next_node_A is None?{best_next_node_A is None} best_next_node_B is None?{best_next_node_B is None}")

                exp_Ab, _ = get_double_node_exp_second(
                    current_node=best_next_node_B,
                    discount_factor=discount_factor,
                    act_1=act_1,
                    act_2=act_2,
                    random_state=random_state,
                    greedy_move=greedy_move,
                    renew_children=renew_children,
                    normalized= normalized,
                    average_score= average_score,
                )
                _, exp_Ba = get_double_node_exp_second(
                    current_node=best_next_node_A,
                    discount_factor=discount_factor,
                    act_1=act_1,
                    act_2=act_2,
                    random_state=random_state,
                    greedy_move=greedy_move,
                    renew_children=renew_children,
                    normalized= normalized,
                    average_score= average_score,
                )

                local_reward_B = best_next_node_B.state.score - current_node.state.score
                local_reward_A = best_next_node_A.state.score - current_node.state.score
                if (normalized):
                    local_reward_B = local_reward_B / average_score
                    local_reward_A = local_reward_A / average_score
                current_node.exp_Ab = exp_Ab + local_reward_B
                current_node.exp_Ba = exp_Ba + local_reward_A
            else:
                # act_2 == "no"
                best_next_node_A = current_node.next_simple_node_A
                best_next_node_B = current_node.next_simple_node_B
                best_next_node_AB = current_node.next_simple_node_A_B
                exp_Ab, _ = get_double_node_exp_second(
                    current_node=best_next_node_A,
                    discount_factor=discount_factor,
                    act_1=act_1,
                    act_2=act_2,
                    random_state=random_state,
                    greedy_move=greedy_move,
                    renew_children=renew_children,
                    normalized= normalized,
                    average_score= average_score,
                )
                _, exp_Ba = get_double_node_exp_second(
                    current_node=best_next_node_B,
                    discount_factor=discount_factor,
                    act_1=act_1,
                    act_2=act_2,
                    random_state=random_state,
                    greedy_move=greedy_move,
                    renew_children=renew_children,
                    normalized= normalized,
                    average_score= average_score,
                )
                local_reward_B = best_next_node_B.state.score - current_node.state.score
                local_reward_A = best_next_node_A.state.score - current_node.state.score
                if (normalized):
                    local_reward_B = local_reward_B / average_score
                    local_reward_A = local_reward_A / average_score
                current_node.exp_Ab = exp_Ab + local_reward_A
                current_node.exp_Ba = exp_Ba + local_reward_B

            return current_node.exp_Ab, current_node.exp_Ba


def double_expand_and_get(root_state: State, model_A, model_B, height=3, greedy_value=False, return_node=False,
                          greedy_move=False, random_state=False, discount_factor=1, double_learning=False,
                          device_0_number=0, device_1_number=1,renew_children=False, act_1="deep", act_2="simple",
                          flip_number = None, normalized = False, average_score = 1,

                          ):
    """
    :param device_1_number:
    :param flip_number: "Rotate counterclockwise + symmetrical about the y-axis."
    :param act_1:
    :param act_2:
    :param renew_children:
    :param device_0_number:
    :param root_state:
    :param model:
    :param height:
    :param greedy_value:
    :param return_node:
    :param greedy_move:
    :param random_state:
    :param discount_factor:
    :param double_learning: Use double learning or not. If so, we use "model" to select move a*, use value model to get
        value from move a*.
    :param value_model:  Only used in double_learning.
    :return:
    """
    if (model_B == None or model_A == None):
        raise Exception(f"It is not the original double learning")

    root_node = double_exp_node(state=root_state.clone(), height=height)
    standby_node_lis = [root_node]
    bottom_node_lis = []

    # expand
    while (standby_node_lis != []):
        children_node_lis = []
        for node in standby_node_lis:
            children_node_lis.extend(double_expand(node, bottom_nodes=bottom_node_lis, random_state=random_state))
        standby_node_lis = children_node_lis

    bottom_board_lis = []
    # calculate bottom_value
    for bottom_node in bottom_node_lis:
        bottom_board_lis.append(bottom_node.state.board)

    if greedy_value or return_node or greedy_move:
        # calculate Value(children)
        for cildren_node in root_node.children:
            bottom_node_lis.append(cildren_node)
            bottom_board_lis.append(cildren_node.state.board)

    bottom_board_array = np.array(bottom_board_lis)
    bottom_answer_array = calculate_double_value(bottom_board_array, model_A, model_B,flip_number=flip_number,device_0_number=device_0_number,device_1_number=device_1_number)
    for itr, bottom_node in enumerate(bottom_node_lis):
        bottom_node.value_A = bottom_answer_array[0][itr].item()
        bottom_node.value_B = bottom_answer_array[1][itr].item()

    # print("***:",len(root_node.children) )
    ev_A, ev_B = get_double_node_exp_first(current_node=root_node, discount_factor=discount_factor,
                                           random_state=random_state, greedy_move=greedy_move, normalized= normalized, average_score= average_score)
    if (root_node.next_node_B == None):
        for itr, bottom_node in enumerate(bottom_node_lis):
            print("-----")
            print(bottom_board_array[itr])
            print(bottom_answer_array[0][itr].item())
            print(bottom_answer_array[1][itr].item())
        raise Exception(f"root.next_node_B is none ,current board ev_A is {root_node.exp_A}, ev_B is {root_node.exp_B}")

    ev_Ab, ev_Ba = get_double_node_exp_second(
        current_node=root_node,
        discount_factor=discount_factor,
        act_1=act_1,
        act_2=act_2,
        random_state=random_state,
        greedy_move=greedy_move,
        renew_children=renew_children,
        normalized=normalized,
        average_score=average_score,
    )

    # We should add more in greedy move
    return root_node.next_move_A_B, root_node.next_move_A, root_node.next_move_B, ev_Ab, ev_Ba, root_node


def test_part(root_state: State, height=3):
    arpha = 1
    gamma = 1
    double_node = double_exp_node(state=root_state.clone(), height=height)
    last_node = double_exp_node(state=root_state.clone(), height=height)

    # deep_simple

    Node_AS_b: double_exp_node = double_node.next_node_B
    r_Ab = (Node_AS_b.state.score - double_node.state.score)
    target_value_A = last_node.value_A + arpha * (r_Ab + gamma * Node_AS_b.exp_Ab - last_node.value_A)

    Node_AS_a: double_exp_node = double_node.next_node_A
    r_Ba = (Node_AS_a.state.score - double_node.state.score)
    target_value_B = last_node.value_B + arpha * (r_Ba + gamma * Node_AS_a.exp_Ba - last_node.value_B)


if __name__ == '__main__':
    board = np.ones((2, 16), dtype="int")
    input_boards = np.zeros((board.shape[0], INPUT_SIZE_A), dtype="float32")
    # for itr in range(2):
    # make_input(input_boards[itr,:],board[itr,:])
    print(input_boards)
