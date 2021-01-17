"""A* grid planning"""
import math
from matplotlib import pyplot as plt
from datetime import datetime
from ruta import Aruta

#from IPython.display import clear_output
from time import sleep
plt.rcParams['axes.facecolor'] = 'green'
# set obstable positions
show_animation = True

class Amapeo:
    def __init__(self, width_field, height_field, scaled, grid_size):

        # current date and time
        self.now = datetime.now()
        self.timestamp = str(datetime.timestamp(self.now))

        self.width_field = width_field
        self.height_field = height_field
        self.scaled = scaled
        self.grid_size = grid_size # [cm]
        self.robot_rad = 8.0 # [cm]

        self.go_x = 100 #110//self.scaled #Puntos a donde debe de ir (pelota o meta) go_x
        self.go_y = 125 #100//self.scale
        self.ball_radius = 5

        self.robot_x = round(self.width_field/2) #45 # robot_x, robot_y posicion de robot
        self.robot_y = 5 #1

        self.ox = [] #posicion de lineas 
        self.oy = []

        self.oxx = [] #posicion de obstaculos 
        self.oyy = []

        self.obstacle_goalkeeper_x = [] #posición de portero
        self.obstacle_goalkeeper_y = []

        self.offset_obstacles = 5 
        #COMMENT PLOT 
        self.AfieldCompleted() #LLamar una funcion de la clase misma

    def Aclean_obstacles(self):
        self.oxx = [] #posicion de obstaculos 
        self.oyy = []

    def AfieldCompleted(self):
        for i in range(0, self.width_field):  # For de abajo
            self.ox.append(i)
            self.oy.append(0.0)
        for i in range(0, self.height_field):  # For de derecha
            self.ox.append(self.width_field)
            self.oy.append(i)
        for i in range(0, self.width_field + 1):  # For arriba
            self.ox.append(i)
            self.oy.append(self.height_field)
        for i in range(0, self.height_field + 1):  # For izquierda
            self.ox.append(0.0) 
            self.oy.append(i)
        
    def Aplot_obstacle(self, obs_x, obs_y, obs_radio):
        #ox.append(obs_x/self.scaled)
        #oy.append(obs_y/self.scaled)
        self.oxx.append(obs_x)
        self.oyy.append(obs_y)

        #Dar espacio , rango de (1:5), 5 cm
        for i in list(range(1, round(obs_radio/2)+self.offset_obstacles)):  
            self.oxx.append(obs_x + i)
            self.oyy.append(obs_y)
            self.oxx.append(obs_x - i)
            self.oyy.append(obs_y)

    def Aplot_ball(self, ball_x, ball_y):
        self.go_x = ball_x
        self.go_y = ball_y

    def Aplot_goalkeeper(self, x, y):
        self.obstacle_goalkeeper_x.append(x)
        self.obstacle_goalkeeper_y.append(y)

    def Aplot_ball_robot(self):
        self.ox.extend(self.oxx) #Concatanate lists
        self.oy.extend(self.oyy)
        self.ox.extend(self.obstacle_goalkeeper_x) #Concatanate lists
        self.oy.extend(self.obstacle_goalkeeper_y)

        # start and goal position, Star robot (x), [cm]
        if self.go_x <= self.width_field and self.go_x >= 0 and self.go_y <= self.height_field and self.go_y >= 0:
            #print("Positions are: True, to be continued...")
            if show_animation:  # pragma: no cover
                plt.clf()

                plt.plot(self.ox, self.oy, "sw")  # Lines (obstacles)
                plt.plot(self.oxx, self.oyy, "sb")  # Obstacles
                plt.plot(self.obstacle_goalkeeper_x, self.obstacle_goalkeeper_y, "sk")  # Goalkeeper
                plt.plot(self.robot_x, self.robot_y, "^k")  # Star robot
                plt.plot(self.go_x, self.go_y, "or")  # Ball position
                plt.grid(True)
                obj_ruta = Aruta(self.ox, self.oy, self.grid_size, self.robot_rad) #Planner
                rx, ry = obj_ruta.planning(self.robot_x, self.robot_y, self.go_x, self.go_x) #Planning
                plt.plot(rx, ry, "ok")
                #print("rx")
                #print(rx)
                #print("ry")
                #print(ry)
                #print('rx: ',rx,'\n ry:',ry) #Print ruta


                plt.grid(True)
                plt.axis("equal")
                #plt.show(block=False) #Para que no se congele la ejecución
                plt.savefig('evidence_image/actual_field_status.png')
        else:
            print("- 400 code error, Posiciones no correctas -")
        return None