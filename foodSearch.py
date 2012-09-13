import random
import time
import cProfile

random.seed(234)

class Grid:
    def __init__(self, size=10):
        self.running = True
        self.square_size = size
        self._cell_grid = [0] * size
        for y in range(size):
            self._cell_grid[y] = [0] * size
    def get(self,x,y):
        if x > self.square_size-1 or x<0 or y>self.square_size-1 or y<0:
            return -1000 # Wall is there
        return self._cell_grid[y][x]

    def set(self,x,y,val):
        self._cell_grid[y][x] = val

    def fillWithFood(self, units, maxVal=7):
        for c in range(units):
            x = random.randint(0, self.square_size-1)
            y = random.randint(0, self.square_size-1)
            v = random.randint(0, maxVal)
            self.set(x, y, v)

    def addRobot(self, robot):
        self.robot = robot
        self.set(robot.x, robot.y, 'R')

    def foodExists(self):
        for y in range(self.square_size):
            for x in range(self.square_size):
                if self.get(x,y) != 'R' and self.get(x,y) > 0:
                    return True
        return False

    def update(self):
        if self.running:
            # Obey Rules!
            self.robot.x = max(min(self.robot.x, self.square_size-1), 0)
            self.robot.y = max(min(self.robot.y, self.square_size-1), 0)

            if not ((self.robot.x == self.robot.old_x) and
                (self.robot.y == self.robot.old_y)):
                self.robot.energy += self.get(self.robot.x, self.robot.y)
                self.set(self.robot.old_x, self.robot.old_y, 0)
                self.set(self.robot.x, self.robot.y, 'R')

            reading = [0]*3
            for yi,y in enumerate(range(self.robot.y-1, self.robot.y+2)):
                reading[yi] = [0]*3
                for xi,x in enumerate(range(self.robot.x-1, self.robot.x+2)):
                    reading[yi][xi] = self.get(x,y)

            self.robot.sensor_update(reading)

            if self.robot.energy <= 0:
                self.running = False

class Robot:
    def __init__(self, x_start=5, y_start=5, e_start=10):
        self.energy = e_start
        self.x = x_start
        self.y = y_start
        self.old_x = x_start
        self.old_y = y_start
        self.reading = [([0]*3)]*3
        self.old_reading = self.reading
        self.memory = {}

    def thinkAndMove(self, algorithm):
        nextMove = algorithm(self.reading, self.energy, self.memory)
        self.move(nextMove[0], nextMove[1])

    def move(self, x_delta, y_delta):
        # Obey Rules!
        x_delta = max(min(x_delta, 1), -1)
        y_delta = max(min(y_delta, 1), -1)

        self.old_x = self.x
        self.old_y = self.y

        self.x += x_delta
        self.y += y_delta

        self.energy -= 1

    def sensor_update(self, reading):
        self.old_reading = self.reading
        self.reading = reading

def arrayString(array, size, sep=' ', width=4):
    grid_s = ''
    for y in range(size):
        grid_s += sep
        for x in range(size):
            sqVal = array[y][x]
            if sqVal == 0:
                sqVal = '.'
            grid_s += str(sqVal).center(width,sep)
        grid_s += '\n'
    return grid_s

def gridString(grid, sep=' ', width=4):
    return arrayString(grid._cell_grid, grid.square_size, sep, width)

def visualizeGame(w):
    print(str(w.robot.energy).rjust(5)+
        str((w.robot.x, w.robot.y)).rjust((w.square_size+3)*3))
    print(gridString(w))
    print(arrayString(w.robot.reading, 3, width=6))
    print("Food Exists: "+str(w.foodExists()))

def playGame(food_pieces, algorithm, visualize=True, vis_delay=1):
    lifetime = 0
    r = Robot()
    w = Grid()
    w.fillWithFood(food_pieces) # also need 15, 10, 5
    w.addRobot(r)
    w.update()
    while w.running:
        if visualize:
            visualizeGame(w)
            time.sleep(vis_delay)
        r.thinkAndMove(algorithm)
        lifetime += 1
        w.update()
    if visualize:
        visualizeGame(w)
    if w.foodExists():
        return 0,lifetime
    else:
        return 1,lifetime

def playNGames(n, food_pieces, algorithm):
    score = 0.0
    totalLife = 0.0
    for _ in range(n):
        success, lifetime = playGame(food_pieces, algorithm, visualize=False)
        score += success
        totalLife += lifetime
        print("GAME: "+str(_)+str(success))
    return score/n, totalLife/n

def compareAlgs(food_pieces, algorithms, runs_per=1000):
    for alg_string in algorithms:
        algorithm = globals()[alg_string]
        fraction, life = playNGames(runs_per, food_pieces, algorithm)
        print(alg_string+": "+
            str(fraction*100)+'% '+str(life))

def greedyEater(reading, energy, memory):
    # Memory is a mutable dictionary

    reading[1][1] = 0
    m = max([item for row in reading for item in row])
    if m > 0:
        for y,row in enumerate(reading):
            for x,_ in enumerate(row):
                if reading[y][x] == m:
                    goal = (x-1,y-1)
        return goal

    return (random.randint(-1,1),random.randint(-1,1))

# Better than greedyEater
def greedyEaterNoHalt(reading, energy, memory):
    # Memory is a mutable dictionary

    reading[1][1] = 0
    m = max([item for row in reading for item in row])
    if m > 0:
        for y,row in enumerate(reading):
            for x,_ in enumerate(row):
                if reading[y][x] == m:
                    goal = (x-1,y-1)
        return goal

    goal = (0,0)
    while goal == (0,0):
        goal = (random.randint(-1,1),random.randint(-1,1))

    return goal

def greedyEaterNoBacktrack(reading, energy, memory):
    # Memory is a mutable dictionary
    if not 'last' in memory:
        memory['last'] = (0,0)
    last = memory['last']

    reading[1][1] = 0
    m = max([item for row in reading for item in row])
    if m > 0:
        for y,row in enumerate(reading):
            for x,_ in enumerate(row):
                if reading[y][x] == m:
                    goal = (x-1,y-1)
        memory['last'] = goal
        return goal

    goal = (0,0)
    while goal == (0,0) or (goal[0]+last[0],goal[1]+last[1]) == (0,0):
        goal = (random.randint(-1,1),random.randint(-1,1))

    memory['last'] = goal
    return goal

def greedyEaterChooseDir(reading, energy, memory):
    # Memory is a mutable dictionary
    if not 'last' in memory:
        memory['last'] = (0,0)
    last = memory['last']

    reading[1][1] = 0
    m = max([item for row in reading for item in row])
    if m > 0:
        for y,row in enumerate(reading):
            for x,_ in enumerate(row):
                if reading[y][x] == m:
                    goal = (x-1,y-1)
        memory['last'] = goal
        return goal

    goal = last
    return goal

def greedyEaterChooseDirHugWalls(reading, energy, memory):
    def readingInDir(reading,dir):
        return reading[dir[1]+1][dir[0]+1]

    # Memory is a mutable dictionary
    if not 'last' in memory:
        memory['last'] = (0,0)
    last = memory['last']

    reading[1][1] = 0
    m = max([item for row in reading for item in row])
    if m > 0:
        for y,row in enumerate(reading):
            for x,_ in enumerate(row):
                if reading[y][x] == m:
                    goal = (x-1,y-1)
        memory['last'] = goal
        return goal

    goal = last

    if readingInDir(reading, goal) == -1000:
        goal = (0,0)
    while goal == (0,0) or (goal[0]+last[0],goal[1]+last[1]) == (0,0):
        goal = (random.randint(-1,1),random.randint(-1,1))

    memory['last'] = goal
    return goal

def returnToFood(reading, energy, memory):
    # Memory is a mutable dictionary

    def readingInDir(reading,dir):
        return reading[dir[1]+1][dir[0]+1]

    def movesBetween(posA, posB):
        return max([abs(posA[0]-posB[0]),abs(posA[1]-posB[1])])

    def sign(num):
        if num > 0:
            return 1
        elif num < 0:
            return -1
        return 0

    reading[1][1] = -1000

    if not 'relPos' in memory:
        memory['relPos'] = (0,0)
    relPos = memory['relPos']

    if not 'allFound' in memory:
        memory['allFound'] = dict()

    if not 'last' in memory:
        memory['last'] = (0,0)
    last = memory['last']

    for y in [-1,0,1]:
        for x in [-1,0,1]:
            if not (y==0 and x==0):
                val = readingInDir(reading,(x,y))
                if val > 0:
                    memory['allFound'][(x+relPos[0],y+relPos[1])] = val

    goal = (0,0)

    shortest = 1000 # Absurdly high
    goalPos = (0,0)
    goalEnergy = 0
    for possibility in memory['allFound']:
        temp = movesBetween(possibility, relPos)
        if (temp < shortest or (temp == shortest and goalEnergy < memory['allFound'][possibility])) and memory['allFound'][possibility] > 0:
            shortest = temp
            goalPos = possibility
            memory['goalPos'] = goalPos
            goalEnergy = memory['allFound'][possibility]

    if movesBetween(goalPos, relPos) < energy:
        goal = (sign(goalPos[0] - relPos[0]), sign(goalPos[1] - relPos[1]))
        memory['last'] = goal

    if readingInDir(reading, goal) > 0:
        memory['allFound'][(relPos[0]+goal[0], relPos[1]+goal[1])] = 0
        memory['relPos'] = (relPos[0]+goal[0],relPos[1]+goal[1])
        return goal

    goal = last

    if readingInDir(reading, goal) == -1000:
        goal = (0,0)

    while goal == (0,0) or (goal[0]+last[0],goal[1]+last[1]) == (0,0):
        goal = (random.randint(-1,1),random.randint(-1,1))

    memory['last'] = goal
    memory['allFound'][(relPos[0]+goal[0], relPos[1]+goal[1])] = 0
    memory['relPos'] = (relPos[0]+goal[0],relPos[1]+goal[1])
    return goal

def exploreGood(reading, energy, memory):
    # Memory is a mutable dictionary

    def readingInDir(dir):
        return reading[dir[1]+1][dir[0]+1]

    def movesBetween(posA, posB):
        return max([abs(posA[0]-posB[0]),abs(posA[1]-posB[1])])

    def sign(num):
        if num > 0:
            return 1
        elif num < 0:
            return -1
        return 0

    reading[1][1] = -1000

    if not 'relPos' in memory:
        memory['relPos'] = (0,0)
    relPos = memory['relPos']

    if not 'allFound' in memory:
        memory['allFound'] = dict()

    for y in range(-2,2+1):
        for x in range(-2,2+1):
            if (readingInDir((0,-1)) == -1000 and y == -2) or (readingInDir( (0,1)) == -1000 and y == 2) or (readingInDir( (-1,0)) == -1000 and x == -2) or (readingInDir((1,0)) == -1000 and x == 2):
                memory['allFound'][(x+relPos[0],y+relPos[1])] = -1000

            if abs(x) == 2 or abs(y) == 2:
                if not ((x+relPos[0],y+relPos[1]) in memory['allFound']):
                    memory['allFound'][(x+relPos[0],y+relPos[1])] = 0.1
            else:
                if not (y==0 and x==0):
                    val = readingInDir((x,y))
                    if val>0:
                        memory['allFound'][(x+relPos[0],y+relPos[1])] = val

    goal = (0,0)

    shortest = 1000 # Absurdly high
    goalPos = (0,0)
    goalEnergy = -1000
    for possibility in memory['allFound']:
        moves_to_here = movesBetween(possibility, relPos)
        energy_here = memory['allFound'][possibility]
        if moves_to_here > 0 and (moves_to_here < shortest or ((moves_to_here == shortest) and goalEnergy < energy_here)):
            shortest = moves_to_here
            goalPos = possibility
            goalEnergy = energy_here
            memory['goalPos'] = (goalPos, goalEnergy)

    #if movesBetween(goalPos, relPos) < energy:
    goal = (sign(goalPos[0] - relPos[0]), sign(goalPos[1] - relPos[1]))



    memory['allFound'][(relPos[0]+goal[0], relPos[1]+goal[1])] = -0.1
    memory['relPos'] = (relPos[0]+goal[0],relPos[1]+goal[1])
    return goal

def evalBenefit(reading, energy, memory):
    if not 'visited' in memory: memory['visited'] = set()
    if not 'relPos' in memory: memory['relPos'] = (0,0)
    if not 'last' in memory: memory['last'] = (0,0)

    memory['visited'].add(memory['relPos'])

    def readingInDir(direction):
        return reading[direction[1]+1][direction[0]+1]

    def movesBetween(posA, posB):
        return max([abs(posA[0]-posB[0]),abs(posA[1]-posB[1])])

    def allMoves(position):
        return [(x,y) for x in range(-1,1+1) for y in range(-1,1+1) if (not (x==0 and y==0))]

    def consequence(start,move):
        return (start[0]+move[0],start[1]+move[1])

    def inception(initial, depth):
        revealedPoints = 0
        for new_option in allMoves(initial):
            if depth==0:
                if not consequence(initial, new_option) in memory['visited']:
                    revealedPoints += 1
            else:
                revealedPoints += inception(consequence(initial, new_option), depth-1)

        return revealedPoints

    def benefit(move):
        immediatePoints = readingInDir(move)
        move_consequence = consequence(memory['relPos'],move)
        #punishment = 0
        #if move_consequence == memory['last']:
        #    punishment = -10000
        explorationPoints = 0
        if not move_consequence in memory['visited']:
            explorationPoints += 1
        '''
        revealedPoints = 0
        for new_option in allMoves(move_consequence):
            if not consequence(move_consequence, new_option) in memory['visited']:
                revealedPoints += 1
        '''
        revealedPoints = inception(move_consequence, 4)

        return immediatePoints*1000 + revealedPoints + explorationPoints#+punishment

    def chooseBest(aList):
        return sorted(aList, reverse=True)[0]

    move = chooseBest((benefit(move), move) for move in allMoves(memory['relPos']))[1]
    memory['last'] = memory['relPos']
    memory['visited'].add(consequence(memory['relPos'],move))
    memory['relPos'] = consequence(memory['relPos'],move)
    return move


def realTree(reading, curr_energy, memory):
    if not 'seen' in memory: memory['seen'] = dict()
    if not 'relPos' in memory: memory['relPos'] = (0,0)
    if not 'last' in memory: memory['last'] = (0,0)

    memory['seen'][memory['relPos']] = 0

    def readingInDir(direction):
        return reading[direction[1]+1][direction[0]+1]

    def movesBetween(posA, posB):
        return max([abs(posA[0]-posB[0]),abs(posA[1]-posB[1])])

    def allMoves(position):
        # Add in wall move prohibition
        general = set() #set([(x,y) for x in range(-1,1+1) for y in range(-1,1+1) if (not (x==0 and y==0))])
        for x in range(-1,1+1):
            for y in range(-1,1+1):
                test = (memory['relPos'][0]+x,memory['relPos'][1]+y)
                if test in memory['seen']:
                    if not (memory['seen'][test] == -1000 or (x==0 and y==0)):
                        general.add((x,y))
                elif not (x==0 and y==0):
                    general.add((x,y))

        return general

    def consequence(incoming_state,action):
        pos, energy = incoming_state
        return ((pos[0]+action[0],pos[1]+action[1]), energy - 1)

    def score(state):
        pos, energy = state
        energyLoss = 0
        if energy <= 0:
            energyLoss = -10000
        explorationPoints = 0
        immediatePoints = 0
        if not pos in memory['seen']:
            explorationPoints += 1
        else:
            immediatePoints = memory['seen'][pos]

        for a in allMoves(pos):
            if not consequence((pos, energy-1), a)[0] in memory['seen']:
                explorationPoints += 1

        return (immediatePoints+energy) + explorationPoints*0.1 + energyLoss

    def inception(incoming_state, depth):
        pos, energy = incoming_state

        incomingScore = score(incoming_state)

        subScores = set()
        for action in allMoves(pos):
            if depth == 0:
                outcoming = consequence(incoming_state, action)
                subScores.add(score(outcoming))
            else:
                subScores.add(inception(consequence(incoming_state, action), depth-1))

        return incomingScore+max(subScores)

    def takeInReadings():
        for move in allMoves(memory['relPos']):
            immediatePoints = readingInDir(move)
            pos, energy = consequence((memory['relPos'], curr_energy), move)
            memory['seen'][pos] = immediatePoints

    def benefit(move):
        pos, energy = consequence((memory['relPos'], curr_energy), move)

        punishment = 0
        if pos == memory['last']:
            punishment = -1

        curr = (pos, curr_energy-1)
        return inception(curr, 3)#+punishment


    def chooseBest(aList):
        return sorted(aList, reverse=True)[0]

    memory['seen'][memory['relPos']] = 0
    takeInReadings()
    evaluation = [(benefit(move), move) for move in allMoves(memory['relPos'])]
    #print(sorted(evaluation,reverse=True))
    #print(memory['seen'])
    move = chooseBest(evaluation)[1]
    memory['last'] = memory['relPos']
    memory['relPos'] = consequence((memory['relPos'],curr_energy),move)[0]
    memory['seen'][memory['relPos']] = 0
    return move



def realTree2(reading, curr_energy, memory):
    if not 'seen' in memory: memory['seen'] = dict()
    if not 'relPos' in memory: memory['relPos'] = (0,0)
    if not 'last' in memory: memory['last'] = (0,0)

    memory['seen'][memory['relPos']] = 0

    def readingInDir(direction):
        return reading[direction[1]+1][direction[0]+1]

    def movesBetween(posA, posB):
        return max([abs(posA[0]-posB[0]),abs(posA[1]-posB[1])])

    def allMoves(position):
        # Add in wall move prohibition
        general = set() #set([(x,y) for x in range(-1,1+1) for y in range(-1,1+1) if (not (x==0 and y==0))])
        for x in range(-1,1+1):
            for y in range(-1,1+1):
                test = (memory['relPos'][0]+x,memory['relPos'][1]+y)
                if test in memory['seen']:
                    if not (memory['seen'][test] == -1000 or (x==0 and y==0)):
                        general.add((x,y))
                elif not (x==0 and y==0):
                    general.add((x,y))

        return general

    def consequence(incoming_state,action):
        pos, energy = incoming_state
        return ((pos[0]+action[0],pos[1]+action[1]), energy - 1)

    def score(state):
        pos, energy = state
        energyLoss = 0
        if energy <= 0:
            energyLoss = -10000
        explorationPoints = 0
        immediatePoints = 0
        if not pos in memory['seen']:
            explorationPoints += 1
        else:
            immediatePoints = memory['seen'][pos]

        for a in allMoves(pos):
            if not consequence((pos, energy-1), a)[0] in memory['seen']:
                explorationPoints += 1

        return (immediatePoints+energy) + explorationPoints*0.1 + energyLoss

    def inception(incoming_state, depth):
        pos, energy = incoming_state

        incomingScore = score(incoming_state)

        subScores = set()
        for action in allMoves(pos):
            if depth == 0:
                outcoming = consequence(incoming_state, action)
                subScores.add(score(outcoming))
            else:
                subScores.add(inception(consequence(incoming_state, action), depth-1))

        return incomingScore+max(subScores)

    def takeInReadings():
        for move in allMoves(memory['relPos']):
            immediatePoints = readingInDir(move)
            pos, energy = consequence((memory['relPos'], curr_energy), move)
            memory['seen'][pos] = immediatePoints

    def benefit(move):
        pos, energy = consequence((memory['relPos'], curr_energy), move)

        punishment = 0
        if pos == memory['last']:
            punishment = -1

        curr = (pos, curr_energy-1)
        return inception(curr, 3)#+punishment


    def chooseBest(aList):
        return sorted(aList, reverse=True)[0]

    memory['seen'][memory['relPos']] = 0
    takeInReadings()
    evaluation = [(benefit(move), move) for move in allMoves(memory['relPos'])]
    #print(sorted(evaluation,reverse=True))
    #print(memory['seen'])
    move = chooseBest(evaluation)[1]
    memory['last'] = memory['relPos']
    memory['relPos'] = consequence((memory['relPos'],curr_energy),move)[0]
    memory['seen'][memory['relPos']] = 0
    return move




if __name__ == '__main__':
    playGame(20, realTree2, visualize=True, vis_delay=0.4)
    #playGame(10, evalBenefit, visualize=True, vis_delay=0.1)
    #compareAlgs(20, ['exploreGood', 'realTree2'], runs_per=10)
    #print(str(playNGames(1000, 20, greedyEater)*100.0)+'%')
    #cProfile.run('playNGames(1000, 20, greedyEater)')
