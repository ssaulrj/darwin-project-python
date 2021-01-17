#Obtener ancho de ciertas coordenadas
	def get_width_objs(self): 
		ix1, iy1, iz1 = x, y, z#Punto 1 x y z 
		ix2, iy2, iz2 = x2, y2, z2#Punto 2 x y z
		point1 = rs.rs2_deproject_pixel_to_point(self.color_intrin, [ix1, iy1], iz1)
		point2 = rs.rs2_deproject_pixel_to_point(self.color_intrin, [ix2, iy2], iz2)
		return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2) + math.pow(point1[2] - point2[2], 2))