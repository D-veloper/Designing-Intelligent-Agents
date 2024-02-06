import tkinter as tk
import random
import math
import numpy as np

class Brain():

    def __init__(self,botp):
        self.bot = botp
        self.moves = 0  # attribute to estimate distance covered by bot.
        self.batteryLevel = 100

    # modify this to change the robot's behaviour
    def thinkAndAct(self, lightL, lightR, x, y, sl, sr):
        # wheels not moving - no movement - no response to light
        # total_light = lightL + lightR
        self.moves = self.moves+1 if self.moves < 90 else 0  # increment distance on each turn. reset to 0 on 100.
        self.batteryLevel = max(self.batteryLevel - 1, 0) if self.moves % 10 == 0 and self.batteryLevel > 0 else self.batteryLevel
        # Make the bot turn left and right at random with speed depending on proximity to light source.
        random_turn = random.randint(1,2)  # randomly select a direction. 1 - left, 2 -right
        # #  set leftSpeed to 10 if distance is 60 and bot chose left or is already going left and was not going right.
        # speedLeft = 10 if self.moves > 59 and (random_turn == 1 or sl == 10) and sr != 10 else 3 * total_light
        # #  set rightSpeed to 10 so bot turns right if it did not choose left at random.
        # speedRight = 10 if self.moves > 59 and speedLeft != 10 else 3 * total_light

        # Make the bot turn towards the light, but it wobbles.
        # speedLeft = 10 if lightL < lightR and self.moves < 15 else 3
        # speedRight = 10 if lightR < lightL and self.moves < 15 else 3

        # Make the bot run away from the light
        speedLeft = 10 if lightL > 200 or (self.moves > 79 and (random_turn == 1 or sl == 1) and sr != 10) else 3
        speedRight = 10 if lightR > 400 or (self.moves > 79 and speedLeft != 10) else 3
        newX = 1 if x > 999 else 999 if x < 1 else None
        newY = 1 if y > 999 else 999 if y < 1 else None
        return speedLeft, speedRight, newX, newY

class Bot():

    def __init__(self,namep):
        self.name = namep
        self.x = random.randint(100,900)
        self.y = random.randint(100,900)
        self.theta = random.uniform(0.0,2.0*math.pi)
        #self.theta = 0
        self.ll = 60 #axle width
        self.sl = 0.0
        self.sr = 0.0

    def thinkAndAct(self, agents, passiveObjects):
        lightL, lightR = self.senseLight(passiveObjects)
        self.sl, self.sr, xx, yy = self.brain.thinkAndAct(lightL, lightR, self.x, self.y, self.sl, self.sr)
        if xx != None:
            self.x = xx
        if yy != None:
            self.y = yy
        
    def setBrain(self,brainp):
        self.brain = brainp

    #returns the output from polling the light sensors
    def senseLight(self, passiveObjects):
        lightL = 0.0
        lightR = 0.0
        for pp in passiveObjects:
            if isinstance(pp,Lamp):
                lx,ly = pp.getLocation()
                distanceL = math.sqrt( (lx-self.sensorPositions[0])*(lx-self.sensorPositions[0]) + \
                                       (ly-self.sensorPositions[1])*(ly-self.sensorPositions[1]) )
                distanceR = math.sqrt( (lx-self.sensorPositions[2])*(lx-self.sensorPositions[2]) + \
                                       (ly-self.sensorPositions[3])*(ly-self.sensorPositions[3]) )
                lightL += 200000/(distanceL*distanceL)
                lightR += 200000/(distanceR*distanceR)
        return lightL, lightR
    

    # what happens at each timestep
    def update(self,canvas,dt):
        # for now, the only thing that changes is that the robot moves
        #   (using the current settings of self.sl and self.sr)
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
        canvas.create_oval(centre1PosX-8,centre1PosY-8,\
                           centre1PosX+8,centre1PosY+8,\
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
        if self.brain.batteryLevel == 0:
            return
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
        

class Lamp():
    def __init__(self,namep):
        self.centreX = random.randint(100,900)
        self.centreY = random.randint(100,900)
        self.name = namep
        
    def draw(self,canvas):
        body = canvas.create_oval(self.centreX-10,self.centreY-10, \
                                  self.centreX+10,self.centreY+10, \
                                  fill="yellow",tags=self.name)

    def getLocation(self):
        return self.centreX, self.centreY


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

def createObjects(canvas,noOfBots,noOfLights):
    agents = []
    passiveObjects = []
    for i in range(0,noOfBots):
        bot = Bot("Bot"+str(i))
        brain = Brain(bot)
        bot.setBrain(brain)
        agents.append(bot)
        bot.draw(canvas)
    for i in range(0,noOfLights):
        lamp = Lamp("Lamp"+str(i))
        passiveObjects.append(lamp)
        lamp.draw(canvas)
    canvas.bind( "<Button-1>", lambda event: buttonClicked(event.x,event.y,agents) )
    return agents, passiveObjects

def moveIt(canvas,agents,passiveObjects):
    for rr in agents:
        rr.thinkAndAct(agents,passiveObjects)
        rr.update(canvas,1.0)
    canvas.after(50,moveIt,canvas,agents,passiveObjects)

def main():
    window = tk.Tk()
    canvas = initialise(window)
    agents, passiveObjects = createObjects(canvas,noOfBots=1,noOfLights=1)
    moveIt(canvas,agents,passiveObjects)
    window.mainloop()

main()
