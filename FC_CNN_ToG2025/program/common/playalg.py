import numpy as np
import Game2048

#simple 1-ply play

def simplePlay(state, model, normalized = False, average_score = 0):
    # x = np.zeros([4, model.DIM_I], dtype="int64")
    x = np.zeros([4, model.DIM_I], dtype="float32")
    rs = [0 for i in range(4)]
    if state.testUp():
        s = state.clone(); s.doUp()   ; model.make_input(x[0,:], s.board); rs[0] = s.score - state.score
    if state.testRight():
        s = state.clone(); s.doRight(); model.make_input(x[1,:], s.board); rs[1] = s.score - state.score
    if state.testDown():
        s = state.clone(); s.doDown() ; model.make_input(x[2,:], s.board); rs[2] = s.score - state.score
    if state.testLeft():
        s = state.clone(); s.doLeft() ; model.make_input(x[3,:], s.board); rs[3] = s.score - state.score
    # p = sess.run(model.output, feed_dict={model.input:x})
    # print("12345")
    p = model.predict(x).cpu().detach()
    # print("123")
    if(normalized == True):
        ev_d = sorted([(p[i, 0] + rs[i]/average_score, i) for i in range(4)], reverse=True)
    else:
        ev_d = sorted([(p[i,0] + rs[i], i) for i in range(4)], reverse=True)
    # print("321")
    for (ev, dir) in ev_d:
        if state.canMoveTo(dir):
            return dir, ev

def symboards(board):
    if board is None: return [None] * 8
    board2 = board.reshape(4,4)
    board3 = board2.swapaxes(0,1)
    boards = [bd.reshape(16) for bd in [board2, board2[:,::-1], board2[::-1,:], board2[::-1,::-1], \
                                        board3, board3[:,::-1], board3[::-1,:], board3[::-1,::-1]]]
    return boards

def symPlay(state, alg, model):
    # x = np.zeros([32, model.DIM_I], dtype="int64")
    x = np.zeros([32, model.DIM_I], dtype="float32")
    rs = [0 for i in range(4)]
    if state.testUp():
        s = state.clone(); s.doUp()   ; boards = symboards(s.board); rs[0] = s.score - state.score;
        for i in range(8): model.make_input(x[i+ 0,:], boards[i])
    if state.testRight():
        s = state.clone(); s.doRight(); boards = symboards(s.board); rs[1] = s.score - state.score;
        for i in range(8): model.make_input(x[i+ 8,:], boards[i])
    if state.testDown():
        s = state.clone(); s.doDown() ; boards = symboards(s.board); rs[2] = s.score - state.score;
        for i in range(8): model.make_input(x[i+16,:], boards[i])
    if state.testLeft():
        s = state.clone(); s.doLeft() ; boards = symboards(s.board); rs[3] = s.score - state.score;
        for i in range(8): model.make_input(x[i+24,:], boards[i])
    # p = sess.run(model.output, feed_dict={model.input:x})
    p = model.predict(x)
    ps = [alg(p[i*8:(i+1)*8,0]) for i in range(4)]
    ev_d = sorted([(ps[i] + rs[i], i) for i in range(4)], reverse=True)
    for (ev, dir) in ev_d:
        if state.canMoveTo(dir):
            # logger.info(f'board = \n{state.board.reshape(4,4)}')
            # for i in range(4):
            #     logger.info(f'p[] = {p[8*i:8*i+8,0]}')
            # logger.info(f'ps[] = {ps}, rs[] = {rs}')
            # logger.info(f'selected = {dir}, ev = {ev}')
            return dir, ev

def maxPlay(state, model):
    return symPlay(state, max, model)

def minPlay(state, model):
    return symPlay(state, min, model)

def avePlay(state, model):
    return symPlay(state, np.mean, model)

def noAugmentation(playrecords):
    return playrecords

def symAugmentation(playrecords):
    ret = []
    for playrecord in playrecords:
        lastboards = symboards(playrecord['lastboard'])
        for i in range(8):
            ret.append({'lastboard':lastboards[i], 'target':playrecord['target']})
    return ret
