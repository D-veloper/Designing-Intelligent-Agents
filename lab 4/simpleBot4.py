import tkinter as tk
import random
import math
import numpy as np
from aStar import aStarSearch

class Brain():

    def __init__(self,botp):
        self.bot = botp
        self.turningCount = 0
        self.movingCount = random.randrange(50,100)
        self.currentlyTurning = False
        self.map = self.bot.map()
        self.path = aStarSearch(self.map)
        print(self.path)

    # modify this to change the robot's behaviour
    def thinkAndAct(self, x, y, sl, sr, count):
        newX = None
        newY = None

        # wandering behaviour
        if self.currentlyTurning==True:
            speedLeft = -2.0
            speedRight = 2.0
            self.turningCount -= 1
        else:
            speedLeft = 5.0
            speedRight = 5.0
            self.movingCount -= 1
        if self.movingCount==0 and not self.currentlyTurning:
            self.turningCount = random.randrange(20,40)
            self.currentlyTurning = True
        if self.turningCount==0 and self.currentlyTurning:
            self.movingCount = random.randrange(50,100)
            self.currentlyTurning = False

        #toroidal geometry
        if x>1000:
            newX = 0
        if x<0:
            newX = 1000
        if y>1000:
            newY = 0
        if y<0:
            newY = 1000

        return speedLeft, speedRight, newX, newY

class Bot():

    def __init__(self,namep,passiveObjectsp,counterp):
        self.name = namep
        self.x = random.randint(100,900)
        self.y = random.randint(100,900)
        self.theta = random.uniform(0.0,2.0*math.pi)
        #self.theta = 0
        self.ll = 60 #axle width
        self.sl = 0.0
        self.sr = 0.0
        self.passiveObjects = passiveObjectsp
        self.counter = counterp

    def thinkAndAct(self, agents, passiveObjects):
        self.sl, self.sr, xx, yy = self.brain.thinkAndAct\
            (self.x, self.y, self.sl, self.sr, self.counter.dirtCollected)
        if xx != None:
            self.x = xx
        if yy != None:
            self.y = yy
        
    def setBrain(self,brainp):
        self.brain = brainp

    #returns the result from the ceiling-mounted dirt camera
    def map(self):
        map = np.zeros((10,10),dtype=np.int16)
        for p in self.passiveObjects:
            if isinstance(p,Dirt):
                xx = int(math.floor(p.centreX/100.0))
                yy = int(math.floor(p.centreY/100.0))
                map[xx][yy] += 1
        return map

    def distanceTo(self,obj):
        xx,yy = obj.getLocation()
        return math.sqrt( math.pow(self.x-xx,2) + math.pow(self.y-yy,2) )

    def distanceToRightSensor(self,lx,ly):
        return math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                          (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )

    def distanceToLeftSensor(self,lx,ly):
        return math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                            (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )

    # what happens at each timestep
    def update(self,canvas,passiveObjects,dt):
        self.move(canvas,dt)

    # draws the robot at its current position
    def draw(self,canvas):
        points = [ (self.x + 30*math.sin(self.theta)) - 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - 30*math.cos(self.theta)) - 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - 30*math.sin(self.theta)) - 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + 30*math.cos(self.theta)) - 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x - 30*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y + 30*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta), \
                   (self.x + 30*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                   (self.y - 30*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta)  \
                ]
        canvas.create_polygon(points, fill="blue", tags=self.name)

        self.sensorPositions = [ (self.x + 20*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y - 20*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta), \
                                 (self.x - 20*math.sin(self.theta)) + 30*math.sin((math.pi/2.0)-self.theta), \
                                 (self.y + 20*math.cos(self.theta)) + 30*math.cos((math.pi/2.0)-self.theta)  \
                            ]
    
        centre1PosX = self.x 
        centre1PosY = self.y
        canvas.create_oval(centre1PosX-16,centre1PosY-16,\
                           centre1PosX+16,centre1PosY+16,\
                           fill="gold",tags=self.name)

        wheel1PosX = self.x - 30*math.sin(self.theta)
        wheel1PosY = self.y + 30*math.cos(self.theta)
        canvas.create_oval(wheel1PosX-3,wheel1PosY-3,\
                                         wheel1PosX+3,wheel1PosY+3,\
                                         fill="red",tags=self.name)

        wheel2PosX = self.x + 30*math.sin(self.theta)
        wheel2PosY = self.y - 30*math.cos(self.theta)
        canvas.create_oval(wheel2PosX-3,wheel2PosY-3,\
                                         wheel2PosX+3,wheel2PosY+3,\
                                         fill="green",tags=self.name)

        sensor1PosX = self.sensorPositions[0]
        sensor1PosY = self.sensorPositions[1]
        sensor2PosX = self.sensorPositions[2]
        sensor2PosY = self.sensorPositions[3]
        canvas.create_oval(sensor1PosX-3,sensor1PosY-3, \
                           sensor1PosX+3,sensor1PosY+3, \
                           fill="yellow",tags=self.name)
        canvas.create_oval(sensor2PosX-3,sensor2PosY-3, \
                           sensor2PosX+3,sensor2PosY+3, \
                           fill="yellow",tags=self.name)

    # handles the physics of the movement
    # cf. Dudek and Jenkin, Computational Principles of Mobile Robotics
    def move(self,canvas,dt):
        if self.sl==self.sr:
            R = 0
        else:
            R = (self.ll/2.0)*((self.sr+self.sl)/(self.sl-self.sr))
        omega = (self.sl-self.sr)/self.ll
        ICCx = self.x-R*math.sin(self.theta) #instantaneous centre of curvature
        ICCy = self.y+R*math.cos(self.theta)
        m = np.matrix( [ [math.cos(omega*dt), -math.sin(omega*dt), 0], \
                        [math.sin(omega*dt), math.cos(omega*dt), 0],  \
                        [0,0,1] ] )
        v1 = np.matrix([[self.x-ICCx],[self.y-ICCy],[self.theta]])
        v2 = np.matrix([[ICCx],[ICCy],[omega*dt]])
        newv = np.add(np.dot(m,v1),v2)
        newX = newv.item(0)
        newY = newv.item(1)
        newTheta = newv.item(2)
        newTheta = newTheta%(2.0*math.pi) #make sure angle doesn't go outside [0.0,2*pi)
        self.x = newX
        self.y = newY
        self.theta = newTheta        
        if self.sl==self.sr: # straight line movement
            self.x += self.sr*math.cos(self.theta) #sr wlog
            self.y += self.sr*math.sin(self.theta)
        canvas.delete(self.name)
        self.draw(canvas)

    def collectDirt(self, canvas, passiveObjects, count):
        toDelete = []
        for idx,rr in enumerate(passiveObjects):
            if isinstance(rr,Dirt):
                if self.distanceTo(rr)<30:
                    canvas.delete(rr.name)
                    toDelete.append(idx)
                    count.itemCollected(canvas)
        for ii in sorted(toDelete,reverse=True):
            del passiveObjects[ii]
        return passiveObjects
        
class Dirt:
    def __init__(self,namep,xx,yy):
        self.centreX = xx
        self.centreY = yy
        self.name = namep

    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-1,self.centreY-1,\
                                  self.centreX+1,self.centreY+1,\
                                  fill="grey",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY

class Counter:
    def __init__(self):
        self.dirtCollected = 0

    def itemCollected(self,canvas):
        self.dirtCollected += 1
        canvas.delete("dirtCount")
        canvas.create_text(50,50,anchor="w",\
                           text="Dirt collected: "+str(self.dirtCollected),\
                           tags="dirtCount")
    
def initialise(window):
    window.resizable(False,False)
    canvas = tk.Canvas(window,width=1000,height=1000)
    canvas.pack()
    return canvas

def buttonClicked(x,y,agents):
    for rr in agents:
        if isinstance(rr,Bot):
            rr.x = x
            rr.y = y

def createObjects(canvas):
    agents = []
    passiveObjects = []

    # place line of dirt across top
    i = 0
    for xx in range(0,10):
        for _ in range(50+random.randrange(-10,10)):
            x = xx*100+random.randrange(0,100)
            y = 0+random.randrange(0,100)
            dirt = Dirt("Dirt"+str(i),x,y)
            i += 1
            passiveObjects.append(dirt)
            dirt.draw(canvas)
            
    # place line of dirt down side
    for yy in range(1,10):
        for _ in range(100+random.randrange(-10,10)):
            x = 9*100+random.randrange(0,100)
            y = yy*100+random.randrange(0,100)
            dirt = Dirt("Dirt"+str(i),x,y)
            i += 1
            passiveObjects.append(dirt)
            dirt.draw(canvas)

    # place less dirt everywhere else
    for xx in range(0,9):
        for yy in range(1,10):
            for _ in range(10+random.randrange(-3,3)):
                x = xx*100+random.randrange(0,100)
                y = yy*100+random.randrange(0,100)
                dirt = Dirt("Dirt"+str(i),x,y)
                i += 1
                passiveObjects.append(dirt)
                dirt.draw(canvas)

    count = Counter()
    
    # place Bot
    bot = Bot("Bot1",passiveObjects,count)
    brain = Brain(bot)
    bot.setBrain(brain)
    agents.append(bot)
    bot.draw(canvas)

    canvas.bind( "<Button-1>", lambda event: buttonClicked(event.x,event.y,agents) )


    return agents, passiveObjects, count

def moveIt(canvas,agents,passiveObjects,count):
    for rr in agents:
        rr.thinkAndAct(agents,passiveObjects)
        rr.update(canvas,passiveObjects,1.0)
        passiveObjects = rr.collectDirt(canvas,passiveObjects,count)
    canvas.after(50,moveIt,canvas,agents,passiveObjects,count)

def main():
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects, count = createObjects(canvas)
    moveIt(canvas,agents,passiveObjects,count)
    window.mainloop()

main()
