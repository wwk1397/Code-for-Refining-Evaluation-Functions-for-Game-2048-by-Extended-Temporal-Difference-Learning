import numpy
import random
import sys

# How game plays

class State:
    def __init__(self, bd = None, sc = None):
        self.board = bd if bd is not None else numpy.zeros([16], dtype="int64")
        self.score = sc if sc is not None else 0
    def initGame(self):
        self.board = numpy.zeros([16], dtype="int64")
        self.score = 0
        self.putNewTile()
        self.putNewTile()
    def clone(self):
        ret = State()
        ret.board = self.board.copy()
        ret.score = self.score
        return ret
    def print(self, fp = sys.stdout):
        for j in range(4):
            for i in range(4):
                print(f'{self.board[j*4+i]:3d}', end='', file=fp)
            print('', file=fp)
        print(f'score = {self.score}', file=fp)
    def doUp(self):
        _, s1 = moveTiles(self.board[0::4])
        _, s2 = moveTiles(self.board[1::4])
        _, s3 = moveTiles(self.board[2::4])
        _, s4 = moveTiles(self.board[3::4])
        self.score += s1 + s2 + s3 + s4
    def doRight(self):
        _, s1 = moveTiles(self.board[3::-1])
        _, s2 = moveTiles(self.board[7:3:-1])
        _, s3 = moveTiles(self.board[11:7:-1])
        _, s4 = moveTiles(self.board[15:11:-1])
        self.score += s1 + s2 + s3 + s4
    def doDown(self):
        _, s1 = moveTiles(self.board[12::-4])
        _, s2 = moveTiles(self.board[13::-4])
        _, s3 = moveTiles(self.board[14::-4])
        _, s4 = moveTiles(self.board[15::-4])
        self.score += s1 + s2 + s3 + s4
    def doLeft(self):
        _, s1 = moveTiles(self.board[0:4])
        _, s2 = moveTiles(self.board[4:8])
        _, s3 = moveTiles(self.board[8:12])
        _, s4 = moveTiles(self.board[12:])
        self.score += s1 + s2 + s3 + s4
    def play(self, dir):
        [self.doUp, self.doRight, self.doDown, self.doLeft][dir]()
    def testUp(self):
        m, _ = moveTiles(self.board[0::4].copy())
        if m: return True
        m, _ = moveTiles(self.board[1::4].copy())
        if m: return True
        m, _ = moveTiles(self.board[2::4].copy())
        if m: return True
        m, _ = moveTiles(self.board[3::4].copy())
        return m
    def testRight(self):
        m, _ = moveTiles(self.board[3::-1].copy())
        if m: return True
        m, _ = moveTiles(self.board[7:3:-1].copy())
        if m: return True
        m, _ = moveTiles(self.board[11:7:-1].copy())
        if m: return True
        m, _ = moveTiles(self.board[15:11:-1].copy())
        return m
    def testDown(self):
        m, _ = moveTiles(self.board[12::-4].copy())
        if m: return True
        m, _ = moveTiles(self.board[13::-4].copy())
        if m: return True
        m, _ = moveTiles(self.board[14::-4].copy())
        if m: return True
        m, _ = moveTiles(self.board[15::-4].copy())
        return m
    def testLeft(self):
        m, _ = moveTiles(self.board[0:4].copy())
        if m: return True
        m, _ = moveTiles(self.board[4:8].copy())
        if m: return True
        m, _ = moveTiles(self.board[8:12].copy())
        if m: return True
        m, _ = moveTiles(self.board[12:].copy())
        return m
    def putNewTile(self):
        emptycells = list()
        for i in range(16):
            if self.board[i] == 0: emptycells.append(i)
        pos = random.choice(emptycells)
        self.board[pos] = 1 if random.random() < 0.9 else 2
    def canMoveTo(self,dir):
        return [self.testUp, self.testRight, self.testDown, self.testLeft][dir]()
    def isGameOver(self):
        if self.testUp(): return False
        if self.testRight(): return False
        if self.testDown(): return False
        if self.testLeft(): return False
        return True

def moveTiles(data):
    '''
    @param data: numpy slice with 4 elements
    Move the numbers to the left (smaller index)
    '''
    if data[0] == 0:
        if data[1] == 0:
            if data[2] == 0:
                if data[3] == 0:
                    return False, 0
                else:
                    data[0] = data[3]
                    data[3] = 0
                    return True, 0
            else:
                if data[2] == data[3]:
                    data[0] = data[2] + 1
                    data[2] = data[3] = 0
                    return True, 2 ** data[0]
                else:
                    data[0] = data[2]
                    data[1] = data[3]
                    data[2] = data[3] = 0
                    return True, 0
        elif data[1] == data[2]:
            data[0] = data[1] + 1
            data[1] = data[3]
            data[2] = data[3] = 0
            return True, 2 ** data[0]
        else:
            data[0] = data[1]
            if data[2] == 0:
                if data[0] == data[3]:
                    data[0] = data[0] + 1
                    data[1] = data[2] = data[3] = 0
                    return True, 2 ** data[0]
                else:
                    data[1] = data[3]
                    data[2] = data[3] = 0
                    return True, 0
            else:
                if data[2] == data[3]:
                    data[1] = data[2] + 1
                    data[2] = data[3] = 0
                    return True, 2 ** data[1]
                else:
                    data[1] = data[2]
                    data[2] = data[3]
                    data[3] = 0
                    return True, 0
    elif data[0] == data[1]:
        data[0] = data[0] + 1
        if data[2] == 0:
            data[1] = data[3]
            data[2] = data[3] = 0
            return True, 2 ** data[0]
        elif data[2] == data[3]:
            data[1] = data[2] + 1
            data[2] = data[3] = 0
            return True, 2 ** data[0] + 2 ** data[1]
        else:
            data[1] = data[2]
            data[2] = data[3]
            data[3] = 0
            return True, 2 ** data[0]
    else:
        if data[1] == 0:
            if data[2] == 0:
                if data[3] == 0:
                    return False, 0
                elif data[0] == data[3]:
                    data[0] = data[0] + 1
                    data[1] = data[2] = data[3] = 0
                    return True, 2 ** data[0]
                else:
                    data[1] = data[3]
                    data[2] = data[3] = 0
                    return True, 0
            elif data[0] == data[2]:
                data[0] = data[0] + 1
                data[1] = data[3]
                data[2] = data[3] = 0
                return True, 2 ** data[0]
            else:
                if data[2] == data[3]:
                    data[1] = data[2] + 1
                    data[2] = data[3] = 0
                    return True, 2 ** data[1]
                else:
                    data[1] = data[2]
                    data[2] = data[3]
                    data[3] = 0
                    return True, 0
        elif data[1] == data[2]:
            data[1] = data[1] + 1
            data[2] = data[3]
            data[3] = 0
            return True, 2 ** data[1]
        else:
            if data[2] == 0:
                if data[3] == 0:
                    return False, 0
                elif data[1] == data[3]:
                    data[1] = data[1] + 1
                    data[2] = data[3] = 0
                    return True, 2 ** data[1]
                else:
                    data[2] = data[3]
                    data[3] = 0
                    return True, 0
            else:
                if data[2] == data[3]:
                    data[2] = data[2] + 1
                    data[3] = 0
                    return True, 2 ** data[2]
                else:
                    return False, 0

def test1():
    import datetime
    print(datetime.datetime.now())
    for i in range(1):
        for i1 in range(16):
            for i2 in range(16):
                for i3 in range(16):
                    for i4 in range(16):
                        data = numpy.array([i1, i2, i3, i4], dtype="int64")
                        a, b = moveTiles(data)
                        print(f"{a}, {b}, {data}")
    print(datetime.datetime.now())

def test2():
    bd = State()
    bd.initGame()
    for i in range(10000):
        bd.print()
        if bd.isGameOver():
            print('game over')
            break
        while True:
            d = random.choice([i for i in range(4)])
            if bd.canMoveTo(d):
                bd.play(d)
                break
        # bd.play(int(input()))
        bd.putNewTile()

if __name__ == "__main__":
    # test1()
    test2()
