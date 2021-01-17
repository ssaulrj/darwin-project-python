class Arobot:
	def __init__(self):
		self.angle_robot_z = 0
		self.angle_robot_camera = 90 - 30
		self.robot_radius = 10 #Pendiente
		#self.set_angle_robot_z(-250)
		self.angle_robot_camera_x = 0 
		self.angle_robot_camera_y = 0

	#Funciones set------------------------------------------------------------------------------
	def set_angle_robot_z(self, giroscopio_z): #Giroscopio eje z, va de -500 a 500
		angle_z = ((giroscopio_z)*(-180))/(-500)
		self.angle_robot_z = angle_z

	def set_angle_robot_camera_x(self, angle): #eje x izquierda derecha: -90 izquierda, 95 derecha
		self.angle_robot_camera_x = 90 - angle

	def set_angle_robot_camera_y(self, angle): #eje y arriba abajo, -55 abajo, + 55 arriba
		self.angle_robot_camera_y = 90 - angle

	#Funciones get------------------------------------------------------------------------------
	def get_angle_robot_camera_x(self):
		return self.angle_robot_camera_x

	def get_angle_robot_camera_y(self):
		return self.angle_robot_camera_y

	def get_angle_robot_z(self): #Giroscopio eje z, va de -500 a 500
		#input_angle = int(input("Robot angle?: ")) #OBTENER EL VALOR DE POSICION DEL ROBOT 
		input_angle = 0
		return input_angle
		#return self.angle_robot_z #Return el actual valor

	def actions(self):
		pass #No hacer nada

	def net_cpp_python(self):
		pass
