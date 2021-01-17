from vision import Avision
from procesamiento import Aprocesamiento #Clase de vision
from mapeo import Amapeo #Import class from a_star
from robot import Arobot #Clase de robot 

class Amain:
    def __init__(self):
        self.event = 0 #1 Penalty kick / 2 Obstacle run
        self.scaled = 1
        self.width_field = 250//self.scaled #90 width x
        self.height_field = 200//self.scaled #45 height y
        self.grid_size = 10.0
        self.clipping_distance_m = 3 #Vista distancia de cámara

    def set_event(self):
        event_number = int(input("#0 Obstacle run || 1 Penalty kick: "))
        if self.event == 0 or self.event == 1: 
            self.event = event_number

    def main(self):
        print(__file__ + " start!!")
        self.set_event()
        #Constructor de vision.py 
        obj_vision = Avision(self.clipping_distance_m)
        #Constructor de robot.py 
        obj_robot = Arobot() 
        #Constructor de mapeo.py 
        obj_mapeo = Amapeo(self.width_field, self.height_field, self.scaled, self.grid_size) #Inicializar variables en A_field de a_star.py       
        #Constructor de procesamiento.py 
        obj_procesamiento = Aprocesamiento(obj_mapeo, obj_robot, obj_vision, self.event) #Inicializar variables en A_vision -> clase Acamera
        obj_procesamiento.main()

if __name__ == '__main__':
    obj_main = Amain()
    #Inicio de programa
    obj_main.main()