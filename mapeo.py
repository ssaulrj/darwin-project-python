"""A* grid planning"""
import math
from matplotlib import pyplot as plt
#from IPython.display import clear_output
from time import sleep
plt.rcParams['axes.facecolor'] = 'green'
# set obstable positions
show_animation = True

class Amapeo:
    def __init__(self, width_field, height_field, scaled, grid_size):
        self.width_field = width_field
        self.height_field = height_field
        self.scaled = scaled
        self.grid_size = grid_size # [cm]
        self.robot_rad = 8.0 # [cm]

        self.gx = 100 #110//self.scaled
        self.gy = 125 #100//self.scale
        self.gx_past = round(self.width_field/2) #110//self.scaled
        self.gy_past = 150 #100//self.scaled
        self.ball_radius = 5

        self.sx = round(self.width_field/2) #45 # sx, sy posicion de robot
        self.sy = 5 #1

        self.ox = [] #posicion de lineas 
        self.oy = []

        self.oxx = [] #posicion de obstaculos 
        self.oyy = []

        self.oxx_offset = [] #offset de obstaculos 
        self.oyy_offset = []
        self.offset_obstacles = 5 
        #COMMENT PLOT 
        self.AfieldCompleted() #LLamar una funcion de la clase misma

    def Aclean_obstacles(self):
        self.oxx = [] #posicion de obstaculos 
        self.oyy = []

        self.oxx_offset = [] #offset de obstaculos 
        self.oyy_offset = []

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

            self.oxx_offset.append(obs_x + i)
            self.oyy_offset.append(obs_y+5)
            self.oxx_offset.append(obs_x - i)
            self.oyy_offset.append(obs_y+5)
            self.oxx_offset.append(obs_x + i)
            self.oyy_offset.append(obs_y-5)
            self.oxx_offset.append(obs_x - i)
            self.oyy_offset.append(obs_y-5)

        self.oxx_offset.append(obs_x+round(obs_radio/2)+self.offset_obstacles+5)
        self.oyy_offset.append(obs_y)
        self.oxx_offset.append(obs_x-round(obs_radio/2)-self.offset_obstacles-5)
        self.oyy_offset.append(obs_y)

    def Aplot_ball(self, ball_x, ball_y):
        self.gx_past = self.gx
        self.gy_past = self.gy
        self.gx = ball_x
        self.gy = ball_y

    def Aplot_goalkeeper(self, x, y):
        self.oxx.append(x)
        self.oyy.append(y)

    def Aplot_ball_robot(self):
        self.ox.extend(self.oxx) #Concatanate lists
        self.ox.extend(self.oxx_offset)
        self.oy.extend(self.oyy)
        self.oy.extend(self.oyy_offset)

        # start and goal position, Star robot (x), [cm]
        if self.gx <= self.width_field and self.gx >= 0 and self.gy <= self.height_field and self.gy >= 0:
            #gx, gy 
            print("Positions are: True, to be continued...")
            if show_animation:  # pragma: no cover
                plt.plot(self.ox, self.oy, "sy")  # Lines (obstacles)
                plt.plot(self.oxx, self.oyy, "sb")  # Obstacles
                plt.plot(self.oxx_offset, self.oyy_offset, "sc")  # Obstacles
                plt.plot(self.sx, self.sy, "^k")  # Star robot
                plt.plot(self.gx_past, self.gy_past, "og")
                plt.plot(self.gx, self.gy, "or")  # Ball position
                plt.grid(True)
                a_star = AStarPlanner(self.ox, self.oy, self.grid_size, self.robot_rad) #Planner
                rx, ry = a_star.planning(self.sx, self.sy, self.gx, self.gy) #Planning
                plt.plot(rx, ry, "ok")
                print("rx")
                print(rx)
                print("ry")
                print(ry)
                #print('rx: ',rx,'\n ry:',ry) #Print ruta
                #sleep(1)
                plt.grid(True)
                plt.axis("equal")
                plt.show(block=False) #Para que no se congele la ejecución
                plt.savefig('evidence_image/actual_field.png')
                #sleep(2)  # La imagen se mostrará x segundos.
                #plt.clf()
                #sleep(1) # esperamos 1 segundo para generar la nueva imagen.
        else:
            #gx, gy = 0,0
            print("Positions are: False")
        #self.Aclean_obstacles()
        return None

class AStarPlanner:
    def __init__(self, ox, oy, reso, rr): #Initialize grid map for a star planning
        """ox: x position list of Obstacles [m], oy: y position list of Obstacles [m]
        reso: grid resolution [m], rr: robot radius[m]
        """
        self.reso = reso
        self.rr = rr
        self.calc_obstacle_map(ox, oy)
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x, y, cost, pind):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.pind = pind

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

    def planning(self, sx, sy, gx, gy): #A star path search
        """input: sx: start x position [m], sy: start y position [m], gx: goal x position [m], gy: goal y position [m]
           output: rx: x position list of the final path, ry: y position list of the final path"""
        nstart = self.Node(self.calc_xyindex(sx, self.minx), self.calc_xyindex(sy, self.miny), 0.0, -1)
        ngoal = self.Node(self.calc_xyindex(gx, self.minx), self.calc_xyindex(gy, self.miny), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(nstart)] = nstart

        while 1:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            c_id = min(open_set,key=lambda o: open_set[o].cost + self.calc_heuristic(ngoal, open_set[o]))
            current = open_set[c_id]

            if show_animation:  # show graph, pragma: no cover
                plt.plot(self.calc_grid_position(current.x, self.minx), self.calc_grid_position(current.y, self.miny), "xc")
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event', lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            if current.x == ngoal.x and current.y == ngoal.y:
                print("Find goal")
                ngoal.pind = current.pind
                ngoal.cost = current.cost
                break

            del open_set[c_id] # Remove the item from the open set
            
            closed_set[c_id] = current # Add it to the closed set

            for i, _ in enumerate(self.motion): # expand_grid search grid based on motion model
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node): # If the node is not safe, do nothing
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node # This path is the best until now. record it
        rx, ry = self.calc_final_path(ngoal, closed_set)
        return rx, ry

    def calc_final_path(self, ngoal, closedset): # generate final course
        rx, ry = [self.calc_grid_position(ngoal.x, self.minx)], [
            self.calc_grid_position(ngoal.y, self.miny)]
        pind = ngoal.pind
        while pind != -1:
            n = closedset[pind]
            rx.append(self.calc_grid_position(n.x, self.minx))
            ry.append(self.calc_grid_position(n.y, self.miny))
            pind = n.pind
        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, minp): #calc grid position
        pos = index * self.reso + minp
        return pos

    def calc_xyindex(self, position, min_pos):
        return round((position - min_pos) / self.reso)

    def calc_grid_index(self, node):
        return (node.y - self.miny) * self.xwidth + (node.x - self.minx)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.minx)
        py = self.calc_grid_position(node.y, self.miny)

        if px < self.minx:
            return False
        elif py < self.miny:
            return False
        elif px >= self.maxx:
            return False
        elif py >= self.maxy:
            return False

        if self.obmap[node.x][node.y]: # collision check
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.minx = round(min(ox))
        self.miny = round(min(oy))
        self.maxx = round(max(ox))
        self.maxy = round(max(oy))
        #print("minx:", self.minx)
        #print("miny:", self.miny)
        #print("maxx:", self.maxx)
        #print("maxy:", self.maxy)

        self.xwidth = round((self.maxx - self.minx) / self.reso)
        self.ywidth = round((self.maxy - self.miny) / self.reso)
        #print("xwidth:", self.xwidth)
        #print("ywidth:", self.ywidth)

        # obstacle map generation
        self.obmap = [[False for i in range(self.ywidth)]
                      for i in range(self.xwidth)]
        for ix in range(self.xwidth):
            x = self.calc_grid_position(ix, self.minx)
            for iy in range(self.ywidth):
                y = self.calc_grid_position(iy, self.miny)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obmap[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model(): # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion

class Ainstrucciones:
    def __init__(self):
        self.x= 0

    def set_x(self): 
        pass

    def get_x(self):
        pass

    def main(self):
        pass