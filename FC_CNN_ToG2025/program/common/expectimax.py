# -*- coding: utf-8 -*-
# tools of expectimax_play.py


import random 
import time
import numpy as np

CATEGORY_afterstate = 342
CATEGORY_state1 = 343
CATEGORY_state2 = 344
input_shape = 416

class Expectimax_setting():
    policy_value = None
    nn_calls = 0
    maxdepth = 0
    model_call_count = 0
    model_call_states = 0
    expand_finished = False
    depth = 0
    @staticmethod
    def debugout(x):
        pass

class TreeNode(object):
    """A node in the Expectimax tree. """

    def __init__(self, state, category):
        self._children = {}  # a map from action to (TreeNode, reward)
        self.category=category
        self.v = 0
        self.state = state
        self.exp = -10000

    def expand_afterstate(self, table):
        if self._children != {}: return False

        legal_actions = []
        x=np.zeros([128,input_shape],dtype="float32")
        # x = np.zeros([128, 384])
        # generate all the states first
        chcount = 0
        for j in range(16):
            if self.state.board[j] != 0: continue

            self.state.board[j]=1
            hv = hash_state(self.state.board, CATEGORY_state1)
            if hv not in table: table[hv] = TreeNode(self.state.clone(), CATEGORY_state1)
            node1 = table[hv]
            self._children[(j,1)] = [node1, 0]
            for d in range(4):
                if self.state.canMoveTo(d):
                    chcount += 1
                    s = self.state.clone(); s.play(d); r = s.score - self.state.score
                    hv = hash_state(s.board, CATEGORY_afterstate)
                    if hv not in table:
                        table[hv] = TreeNode(s, CATEGORY_afterstate)
                        legal_actions.append(hv)
                        idx = len(legal_actions)-1
                        # for k in range(16): x[idx][16 * k + s.board[k]] = 1
                        # for j in range(16):
                        #     b = board[j]
                        #     x[16 * b + j] = 1
                        #     x[16 * (16 + int(j // 4)) + j] = 1
                        #     x[16 * (20 + j % 4) + j] = 1
                        for k in range(16):

                            # x[idx][16 * s.board[k] + k] = 1
                            # x[idx][24 * k + s.board[k] ] = 1
                            # x[idx][24 * k + 16 + int( k//4 ) ] = 1
                            # x[idx][24 * k + 20 + k % 4 ] = 1
                            b = s.board[k]
                            x[idx][16 * b + k] = 1
                            x[idx][16 * (18 + int(k // 4)) + k] = 1
                            x[idx][16 * (22 + k % 4) + k] = 1

                    node1._children[d] = [table[hv], r]
            self.state.board[j]=2
            hv = hash_state(self.state.board, CATEGORY_state2)
            if hv not in table: table[hv] = TreeNode(self.state.clone(), CATEGORY_state2)
            node2 = table[hv]
            self._children[(j,2)] = [node2, 0]
            for d in range(4):
                if self.state.canMoveTo(d):
                    chcount += 1
                    s = self.state.clone(); s.play(d); r = s.score - self.state.score
                    hv = hash_state(s.board, CATEGORY_afterstate)
                    if hv not in table:
                        table[hv] = TreeNode(s, CATEGORY_afterstate)
                        legal_actions.append(hv)
                        idx = len(legal_actions)-1
                        for k in range(16):
                            b = s.board[k]
                            x[idx][16 * b + k] = 1
                            x[idx][16 * (18 + int(k // 4)) + k] = 1
                            x[idx][16 * (22 + k % 4) + k] = 1
                            # x[idx][24 * k + s.board[k] ] = 1
                            # x[idx][24 * k + 16 + int( k//4 ) ] = 1
                            # x[idx][24 * k + 20 + k % 4 ] = 1

                    node2._children[d] = [table[hv], r]
            self.state.board[j]=0

        if (len(legal_actions) == 0): return
        input = x[0:len(legal_actions),:]
        size = input.shape[0]
        # size = chcount
        if Expectimax_setting.nn_calls > 0 and Expectimax_setting.model_call_states + size > Expectimax_setting.nn_calls:
            Expectimax_setting.expand_finished = True
            self._children = {}
            return False
        if Expectimax_setting.nn_calls > 0 and Expectimax_setting.model_call_states + size == Expectimax_setting.nn_calls:
            Expectimax_setting.expand_finished = True
        Expectimax_setting.model_call_states += size
        Expectimax_setting.model_call_count += 1
        # values = Expectimax_setting.policy_value(input)[0]
        values = Expectimax_setting.policy_value(input)
        for i in range(len(legal_actions)):
            table[legal_actions[i]].v = values[i][0]
        return True

    def expand_state(self, table):
        legal_actions = []
        # x = np.zeros([4, 256])
        x=np.zeros([4,input_shape],dtype="float32")
        for d in range(4):
            if self.state.canMoveTo(d):
                s = self.state.clone(); s.play(d); r = s.score - self.state.score
                hv = hash_state(s.board, CATEGORY_afterstate)
                if hv not in table:
                    table[hv] = TreeNode(s, CATEGORY_afterstate) # dummy value
                    legal_actions.append(hv)
                    idx = len(legal_actions)-1
                    for k in range(16):
                        b = s.board[k]
                        x[idx][16 * b + k] = 1
                        x[idx][16 * (18 + int(k // 4)) + k] = 1
                        x[idx][16 * (22 + k % 4) + k] = 1
                        # x[idx][24 * k + s.board[k]] = 1
                        # x[idx][24 * k + 16 + int(k // 4)] = 1
                        # x[idx][24 * k + 20 + k % 4] = 1

                self._children[d] = [table[hv], r]
        input = x[0:len(legal_actions),:]
        Expectimax_setting.model_call_states += input.shape[0]
        Expectimax_setting.model_call_count += 1
        values = Expectimax_setting.policy_value(input)
        # values = Expectimax_setting.policy_value(input)[0]
        # print("legal_actions: ",len(legal_actions))
        # for i in range(len(legal_actions)):
        #    print( legal_actions[i], table[legal_actions[i]].v,values[0].shape)
        for i in range(len(legal_actions)):
            table[legal_actions[i]].v = values[i][0]
        return True
        
    def expand(self, table):
        if self.category==CATEGORY_afterstate:
            return self.expand_afterstate(table)
        else:
            return self.expand_state(table)

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None

hash_items = [[random.randrange(0, 1<<63) for i in range(16)] for j in range(16)]
def hash_state(board, category):
    hash = category
    for i in range(16):
        hash ^= hash_items[i][board[i]]
    return hash

class expectimax():
    def __init__(self):
        self.table = None
        self._root = None

    def get_move(self, state):
        self.table = {}
        self._root = TreeNode(state.clone(), CATEGORY_state1)
        Expectimax_setting.model_call_count = 0
        Expectimax_setting.model_call_states = 0
        Expectimax_setting.expand_finished = False

        hv = hash_state(state.board, CATEGORY_state1)
        self.table[hv] = self._root

        # Generate a tree up to specified condition
        self._root.expand(self.table)
        next_children = [(ch, r) for (act, (ch, r)) in self._root._children.items()]
        for depth in range(1, Expectimax_setting.maxdepth if Expectimax_setting.maxdepth > 0 else 100):
            # cur_list = next_children
            # cur_list = sorted(next_children, key=lambda cr:cr[0].v + cr[1])  # sorting from worst->best
            cur_list = sorted(next_children, key=lambda cr:cr[0].v + cr[1], reverse=True)  # sorting from best->worst
            next_children = []

            for (ch, r) in cur_list:
                if ch.expand(self.table):
                    for (actS, (chS, rS)) in ch._children.items():
                        for (actA, (chA, rA)) in chS._children.items():
                            next_children.append((chA, rS + rA))
                    Expectimax_setting.debugout(f'{depth} {ch.state.board} {r} {Expectimax_setting.model_call_states}')
                if Expectimax_setting.expand_finished: break
            if Expectimax_setting.expand_finished: break
            Expectimax_setting.depth = depth + 1

        # Calculate the expectimax for the generated tree
        values = [(self.expectimax_core(ch, 1, f'{act}({r})') + r, act) for (act, (ch, r)) in self._root._children.items()]
        # values = [(self.expectimax_core(ch, 1, None) + r, act) for (act, (ch, r)) in self._root._children.items()]
        Expectimax_setting.debugout(values)
        return sorted(values)[-1][1]

    def expectimax_core(self, node, depth, path):
        if node.exp >= 0:
            Expectimax_setting.debugout(f'{"  " * depth}{path} reused {node.exp}')
            return node.exp
        if node.is_leaf():
            Expectimax_setting.debugout(f'{"  " * depth}{path} leaf {node.v}')
            # node.exp = node.v
            node.exp = max(node.v, 0)
            return node.exp
        wsum = 0
        for ((pos, num), (state, _)) in node._children.items():
            if (len(state._children)) > 0:
                maxv = max([self.expectimax_core(chC, depth+1, path + f"/{pos},{num}/{actC}({rC})") + rC for (actC, (chC, rC)) in state._children.items()])
                # maxv = max([self.expectimax_core(chC, depth+1, None) + rC for (actC, (chC, rC)) in state._children.items()])
                wsum += maxv * (0.9 if num == 1 else 0.1)

        wave = wsum / (len(node._children) / 2)
        Expectimax_setting.debugout(f'{"  " * depth}{path} node {wave} {node.v}')
        node.exp = wave
        return node.exp

class PrintDot():
    id = 0
    already_output = {}
    @staticmethod
    def rec(node, state, depth, parent, move, fp):
        PrintDot.id += 1
        # myid = PrintDot.id
        myid = hash_state(state.board, node.category)
        if parent != -1:
            print(f'{parent} -> {myid} [label="{move}"];', file=fp)
        if myid in PrintDot.already_output: return
        PrintDot.already_output[myid] = 1
        if node.category == CATEGORY_afterstate:
            color = '' if node._children == {} else ', style="filled", color="#7f7f7f"'
            print(f'{myid} [shape=box, label="{int(node.v)}"{color}];', file=fp)
        else:
            print(f'{myid} [shape=oval, label="{int(node.v)}"];', file=fp)
        if not node.is_leaf():
            for (act,(ch,r)) in node._children.items():
                m = f"{act}" if node.category == CATEGORY_afterstate else f'{"NESW"[act]}/{r}'
                s = state.clone();
                if node.category == CATEGORY_afterstate:
                    pos, num = act
                    s.board[pos] = num
                else:
                    s.play(act)
                PrintDot.rec(ch, s, depth + 1, myid, m, fp)

    @staticmethod
    def run(root, state, filename):
        PrintDot.id = 0
        with open(filename, 'w') as fp:
            print('digraph "mcts_tree" {', file=fp)
            # print('123 [shape = box, label="a"];', file=fp)
            # print('456 [shape = oval];', file=fp)
            # print('789 [shape = oval];', file=fp)
            # print('123 -> 456;', file=fp)
            # print('123 -> 789;', file=fp)
            PrintDot.rec(root, state, 0, -1, '', fp)
            print('}', file=fp);

class PrintDotTree():
    id = 0
    already_output = {}
    @staticmethod
    def rec(node, state, depth, parent, move, fp):
        PrintDotTree.id += 1
        myid = PrintDotTree.id
        if parent != -1:
            print(f'{parent} -> {myid} [label="{move}"];', file=fp)
        if myid in PrintDotTree.already_output: return
        PrintDotTree.already_output[myid] = 1
        if node.category == CATEGORY_afterstate:
            color = '' if node._children == {} else ', style="filled", color="#7f7f7f"'
            print(f'{myid} [shape=box, label="{int(node.v)}"{color}];', file=fp)
        else:
            print(f'{myid} [shape=oval, label="{int(node.v)}"];', file=fp)
        if not node.is_leaf():
            for (act,(ch,r)) in node._children.items():
                m = f"{act}" if node.category == CATEGORY_afterstate else f'{"NESW"[act]}/{r}'
                s = state.clone();
                if node.category == CATEGORY_afterstate:
                    pos, num = act
                    s.board[pos] = num
                else:
                    s.play(act)
                PrintDotTree.rec(ch, s, depth + 1, myid, m, fp)

    @staticmethod
    def run(root, state, filename):
        PrintDotTree.id = 0
        with open(filename, 'w') as fp:
            print('digraph "mcts_tree" {', file=fp)
            # print('123 [shape = box, label="a"];', file=fp)
            # print('456 [shape = oval];', file=fp)
            # print('789 [shape = oval];', file=fp)
            # print('123 -> 456;', file=fp)
            # print('123 -> 789;', file=fp)
            PrintDotTree.rec(root, state, 0, -1, '', fp)
            print('}', file=fp);

def print_rec(node, depth):
    print('  ' * depth, end='')
    print('node_value',node._Q,node._n_visits,node.category )#-member-variables-here
    if not node.is_leaf():
        for ch in node._children.items():
            print_rec(ch[1], depth +1)
