import numpy as np
import random
import sys

def makeSpecificGrid():
    grid = np.zeros((10,10),dtype=np.int16)
    for xx in range(10):
        for yy in range(10):
                grid[xx][yy] = random.randrange(1,3)
    grid[8][9] = 5
    for yy in range(0,9):
        grid[8][yy] = random.randrange(4,5)
    for xx in range(1,8):
        grid[xx][0] = random.randrange(4,5)
    grid[0][0] = 1
    grid[9][9] = 0
    print(grid)
    return grid

def makeSpecificGrid2():
    grid = np.zeros((10,10),dtype=np.int16)
    for xx in range(10):
        for yy in range(10):
                grid[xx][yy] = random.randrange(1,3)
    for xx in range(0,9):
        grid[xx][xx] = random.randrange(4,5)
    grid[0][0] = 1
    grid[9][9] = 0
    print(grid)
    return grid


def aStarSearch(grid):
    heuristicGrid = np.zeros((10,10),dtype=np.int16)
    for xx in range(10):
        for yy in range(10):
            heuristicGrid[xx][yy] = 10*(xx + yy)
    #print(heuristicGrid)
    bestForPosition = np.zeros((10,10),dtype=np.int16)
    #visitedList = {}
    #visitedList[(9,9)] = heuristicGrid[9,9]
    currentlyActiveList = []
    currentPosition = (9,9)
    currentlyActiveList.append( (currentPosition,heuristicGrid[9][9]) ) # coordinates, a* value
    temp = 0
    ## find max values
    while currentlyActiveList:
        temp += 1
        currentlyActiveList = \
            [(coords, score) for (coords, score) in currentlyActiveList if coords != currentPosition]

        if currentPosition[0]>0:
            possPosition1x = currentPosition[0]-1
            possPosition1y = currentPosition[1]
            aStarScore1 = bestForPosition[currentPosition[0]][currentPosition[1]] \
                +grid[possPosition1x][possPosition1y] \
                +heuristicGrid[possPosition1x][possPosition1y]
            currentlyActiveList.append( ((possPosition1x, possPosition1y), aStarScore1))
            bestForPosition[possPosition1x][possPosition1y] =\
                max( bestForPosition[currentPosition[0]][currentPosition[1]]
                     + grid[possPosition1x][possPosition1y],\
                 bestForPosition[possPosition1x][possPosition1y])

        if currentPosition[1]>0:
            possPosition2x = currentPosition[0]
            possPosition2y = currentPosition[1]-1
            aStarScore2 = bestForPosition[currentPosition[0]][currentPosition[1]] \
                +grid[possPosition2x][possPosition2y] \
                +heuristicGrid[possPosition2x][possPosition2y]
            currentlyActiveList.append( ((possPosition2x, possPosition2y), aStarScore2))
            bestForPosition[possPosition2x][possPosition2y] =\
                max( bestForPosition[currentPosition[0]][currentPosition[1]]
                     + grid[possPosition2x][possPosition2y],\
                 bestForPosition[possPosition2x][possPosition2y])

        sorted_by_second = currentlyActiveList.sort(key=lambda tuple: tuple[1], reverse=True)
        #print("currently active: ",currentlyActiveList)
        if not currentlyActiveList:
            break
        currentPosition = currentlyActiveList[0][0]
        #print(currentPosition)
        #print(grid)
        #print(bestForPosition)
        #input()
    ## construct best path
    #print(bestForPosition)
    pathGrid = np.zeros( (10,10), dtype=np.int8) #just for illustration
    path = []
    currentPosition = (9,9)
    path.append( (9,9) )
    pathGrid[9][9] = True
    while currentPosition != (0,0):

        if currentPosition[0]>0:
            possPosition1x = currentPosition[0]-1
            possPosition1y = currentPosition[1]
            pos1 = True
        if currentPosition[1]>0:
            possPosition2x = currentPosition[0]
            possPosition2y = currentPosition[1]-1
            pos2 = True
        pos1Bigger = bestForPosition[possPosition1x][possPosition1y]>\
            bestForPosition[possPosition2x][possPosition2y]
        if (pos1 and not pos2) or (pos1 and pos2 and pos1Bigger):
            path.append( (possPosition1x,possPosition1y) )
            pathGrid[possPosition1x][possPosition1y] = 1
            currentPosition = (possPosition1x,possPosition1y)
        if (pos2 and not pos1) or (pos1 and pos2 and not pos1Bigger):
            path.append( (possPosition2x,possPosition2y) )
            pathGrid[possPosition2x][possPosition2y] = 1
            currentPosition = (possPosition2x,possPosition2y)
    print(grid)
    print(path)
    print(pathGrid)
    return(path)

aStarSearch(makeSpecificGrid())
#aStarSearch(makeSpecificGrid2())

    
