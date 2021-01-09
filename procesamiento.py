import cv2
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import colorsys
import random

# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic, felzenszwalb, watershed
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.util import img_as_float
from skimage import io
from skimage import data, segmentation, measure, color, img_as_float
from skimage.measure import regionprops
from sklearn import metrics #ROC graphics, metrics
import itertools


dic_colors = { "lower_color_blue"   : np.array([95, 100, 40], dtype=np.uint8), #Azul
               "upper_color_blue"   : np.array([135, 255, 255], dtype=np.uint8), #Azul
               "lower_color_green"  : np.array([30, 0, 0], dtype=np.uint8), #35,100,20,255/25,52,72 ideal->25, 52, 20, Colorverde 30,0,0     
               "upper_color_green"  : np.array([90, 255, 255], dtype=np.uint8), #102, 255, 255 Color verde 90,255,255
               "lower_color_white"  : np.array([0, 0, 168], dtype=np.uint8), #blanco 0, 0, 212
               "upper_color_white"  : np.array([172,111,255], dtype=np.uint8), #blanco 131, 255, 255
               "lower_color_yellow" : np.array([20, 70,   70], dtype=np.uint8), #amarillo
               "upper_color_yellow" : np.array([35, 255, 255], dtype=np.uint8), #amarillo
               "lower_color_red"    : np.array([0,20,20], dtype=np.uint8), #rojo (175,50,20)
               "upper_color_red"    : np.array([7,255,255], dtype=np.uint8), #rojo (180,255,255)
}

class Aprocesamiento:
    def __init__(self, obj_mapeo, obj_robot, obj_vision):
        self.obj_mapeo  = obj_mapeo
        self.obj_robot  = obj_robot
        self.obj_vision = obj_vision

        self.color_intrin = 0 #Obtener ancho de objeto
        self.width_obj = 0 #Ancho de un objeto
        self.color_image = 0 #Imagen a color
        self.bg_removed = 0 #Imagen a color limitado a distancia
        self.var_limits_inside_object = 5 #Pixeles dentro de objeto detectado

        self.real_array_roc = [] #Lista de real 
        self.pred_array_roc = [] #Lista pred

        self.numbers_array_obj = list(range(0,192))
        self.distance_array_obj = [0]
        self.width_array_obj = [0]

        self.saved_model_loaded = tf.saved_model.load("custom-416")
        self.infer = self.saved_model_loaded.signatures['serving_default']
        self.file_names = "obj.names"
        self.fps = 30 #Numero de fotogramas deseado  (maximo 30)

    #Funciones------------------------------------------------------------------------------------------------------------------------------------------
    def configBox(self, pred_bbox):
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        return boxes,pred_conf

    def preprocessToTF(self, frame):
        image_data = cv2.resize(frame, (416, 416))
        image_data = image_data /255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        return image_data

    # Funcion que lee el archivo .names
    def read_class_names(self, class_file_name):
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names

    # Funcion que regresa la imagen con los cuadros dibujados
    def draw_bbox(self, image, bboxes, class_dir, show_label=True):
        classes = self.read_class_names(class_dir)
        num_classes = len(classes)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)
        out_boxes, out_scores, out_classes, num_boxes = bboxes
        for i in range(num_boxes[0]):
            if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue
            coor = out_boxes[0][i]
            coor[0] = int(coor[0] * image_h)
            coor[2] = int(coor[2] * image_h)
            coor[1] = int(coor[1] * image_w)
            coor[3] = int(coor[3] * image_w)

            fontScale = 0.5
            score = out_scores[0][i]
            if score > 0.7:
                class_ind = int(out_classes[0][i])
                bbox_color = colors[class_ind]
                bbox_thick = int(0.6 * (image_h + image_w) / 600)
                c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
                #image =Identificar_Compa_Ene(image,c1,c2,colores,Imagen_Profundidad)
                cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
                #cx1, cx2 = (coor[1], coor[0]), (coor[3], coor[2])
                cx1, cy2 = int(round(coor[1]+((coor[3]-coor[1])/2))), int(round(coor[0]+((coor[2]-coor[0])/2)))
                cv2.circle(image, (cx1, cy2), 3, (0, 0, 255), 5)

                #print(classes[class_ind]) #Identifica la clase encontrada

                if classes[class_ind] == "balon": #1 ball
                    self.put_obj_in_map(cx1,cy2,round(self.obj_vision.depth_image[cy2,cx1]/10,2), image, 0, 0, 1) 
                    print("hello balon")

                elif classes[class_ind] == "portero": #2 goalkeeper
                    self.put_obj_in_map(cx1,cy2,round(self.obj_vision.depth_image[cy2,cx1]/10,2), image, 0, 0, 2) 
                    print("hello portero")

                elif classes[class_ind] == "porteria":
                    print("hello porteria")

                """if show_label:
                    bbox_mess = '%s: %.2f' % (classes[class_ind], score)
                    t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                    c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                    
                    cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), 
                                bbox_color, -1) #filled

                    cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)"""
                   
        return image

    def cen_moments(self, countours_found):
        M = cv2.moments(countours_found) # Encontrar el centroide del mejor contorno y marcarlo
        if M["m00"] != 0:
            cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        else:
            cx, cy = 0, 0  # set values as what you need in the situation
        return cx, cy

    def paint_region_with_avg_intensity(self, img, rp, mi, channel): #Marcar regiones de segmentacion
        for i in range(rp.shape[0]):
            img[rp[i][0]][rp[i][1]][channel] = mi
        return img

    def seg_superpix(self, img): #Felzenszwalb es mas estable, realizar pruebas con los 3 (slic, watershed & fel..)
        #segments = slic(img, n_segments=200, compactness=10, multichannel=True, enforce_connectivity=True, convert2lab=True)
        segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=60)
        #gradient = sobel(rgb2gray(img))
        #segments = watershed(gradient, markers=250, compactness=0.001)
        for i in range(3):
            regions = regionprops(segments, intensity_image=img[:,:,i])
            for r in regions:
                img = self.paint_region_with_avg_intensity(img, r.coords, int(r.mean_intensity), i)
        return img 

    def filter_color(self, image, color): #Funcion para filtrar color
        hsv_frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_frame, dic_colors.get("lower_color_"+color), dic_colors.get("upper_color_"+color))
        color_total = cv2.bitwise_and(image, image, mask=color_mask)
        diff_total = cv2.absdiff(image, color_total)
        #cv2.imshow('Diferencias detectadas', diff_total)
        cv2.imwrite("evidence_image/frame_green.png", diff_total) #diff_total imagen sin verde
        return diff_total
         
    def get_real_distance_objs(self, cz_blue): #Obtener distancia z dado un valor dado de z, considerando al robot, analisis en 4.5.1
        print("Centroid in {} cm.".format(cz_blue)) #Distancia de objeto
        A_dis_total = cz_blue * math.sin(math.radians(self.obj_robot.angle_robot_camera))  
        print("Centroid real in {} cm.".format(A_dis_total)) #Distancia de objeto
        return A_dis_total

    def get_coordenates_map(self, distance_robot_obj): #Obtener las coordenadas para mapear objeto con respecto a robot
        xb = math.sin(math.radians(self.obj_robot.get_angle_robot_z())) * distance_robot_obj
        yb = math.cos(math.radians(self.obj_robot.get_angle_robot_z())) * distance_robot_obj
        xb_object = self.obj_mapeo.sx + xb #Sx posicion robot en mapa
        yb_object = self.obj_mapeo.sy + yb #Sy posicion robot en mapa
        return xb_object, yb_object

    #Poner objetos en mapa-----------------------------------------------------------------------------------------------------------------------------
    #obj 0 Put blue obstacles
    #obj 1 Put ball 
    #obj 2 Put portero - goalkeeper
    #obj 3 Put porteria - goal
    #obj 4 Put lines
    def put_obj_in_map(self, cx, cy, cz, image_blue, cnt, var_limits_inside, obj):
        #Plot objects blue 
        #cv2.rectangle(self.image_blue, (x, y), (x+w, y+h), (0, 255, 0), 2) #(image, starrpoint, endpoint,color,thickness(-1 fill))
        #cv2.circle(self.image_blue, (cx_blue, cy_blue), 5, 255, -1)
        cv2.line(image_blue, (cx, cy), (round(self.obj_vision.width/2), round(self.obj_vision.height/2)), 255, 2) #Línea centro del frame al centroide
        #
        #cv2.circle(self.image_blue, (x+var_limits_inside, y+round(h/2)), 10, (0, 255, 0), -1)
        #cv2.circle(self.image_blue, (x+w-var_limits_inside, y+round(h/2)), 10, (0, 255, 0), -1)
        #cv2.line(self.image_blue, (x+var_limits_inside, y+round(h/2)), (x+w-var_limits_inside, y+round(h/2)), (0, 0, 255), 2) #Línea centro del frame al centroide
        
        cz_real = round(self.get_real_distance_objs(cz),2) #Obtener distacia real
                
        #Afield_obj.Aplot_ball_robot(self.pos_robot_x, self.pos_robot_y, self.pos_ball_x, self.pos_ball_y) #Posiciones del robot, pelota y ruta (rx, ry)
        if obj == 0: #print('Obj blue')
            x, y, w, h = cv2.boundingRect(cnt) #Dibujar un rectángulo alrededor del objeto
            width_obj = self.obj_vision.get_width_objs(x, y, w, h, var_limits_inside) #Obtener ancho de objeto
            print('Result width: '+str(round(width_obj,3)))
            self.width_obj = round(width_obj,2)
            self.width_obj = round(width_obj,2)
            x_obs, y_obs = self.get_coordenates_map(cz_real)

            #COMMENT PLOTpass
            distance_center = self.obj_vision.get_distance_points(cx, cy, cz_real, 
                            round(self.obj_vision.width/2), round(self.obj_vision.height/2), 
                            round(self.obj_vision.depth_image[round(self.obj_vision.height/2),round(self.obj_vision.width/2)]/10,2)) 

            x_obs, y_obs = self.get_coordenates_map(cz_real)
            #COMMENT PLOTpass
            if cx <= round(self.obj_vision.width/2):
                self.obj_mapeo.Aplot_obstacle(x_obs - distance_center, y_obs, self.width_obj) #Ubicar obj, obstacle

            elif cx >= round(self.obj_vision.width/2):
                self.obj_mapeo.Aplot_obstacle(x_obs + distance_center, y_obs, self.width_obj) #Ubicar obj, obstacle
        
        elif obj == 1: #print('Obj ball')
            print("Z ball: "+str(cz_real))
            #Obtener distancia entre centro (orientacion de camara y objeto identificado)
            distance_center = self.obj_vision.get_distance_points(cx, cy, cz_real, 
                            round(self.obj_vision.width/2), round(self.obj_vision.height/2), 
                            round(self.obj_vision.depth_image[round(self.obj_vision.height/2),round(self.obj_vision.width/2)]/10,2)) 

            x_obs, y_obs = self.get_coordenates_map(cz_real)
            #COMMENT PLOTpass
            if cx < round(self.obj_vision.width/2):
                self.obj_mapeo.Aplot_ball(x_obs - distance_center, y_obs)

            elif cx > round(self.obj_vision.width/2):
                self.obj_mapeo.Aplot_ball(x_obs + distance_center, y_obs)

        elif obj == 2: #print('Obj goalkeeper')
            pass
            #COMMENT PLOTpass
            #self.obj_mapeo.Aplot_goalkeeper(x_obs, y_obs)

        elif obj == 3: #print('Obj goal')
            pass
            #COMMENT PLOTpass
            #self.obj_mapeo.Aplot_goal(x_obs, y_obs)

        elif obj == 4: #print('Obj lines')
            pass
            #COMMENT PLOTpass
            #self.obj_mapeo.Aplot_lines(x_obs, y_obs)

        return image_blue, cz_real, self.width_obj

    def put_ball(self):
        return None

    def put_portero(self):
        return None

    def put_lines(self):
        return None

    def put_line_goal(self):
        return None

    def put_goal(self):
        return None

    #Buscar objetos-------------------------------------------------------------------------------------------------------------------------------------
    def search_blue(self, image_blue, color):
        self.x_z_object = 0
        self.x_width_object = 0
        cx, cy = 0, 0
        self.image_blue = image_blue
        frame = cv2.blur(self.image_blue, (15, 15))  # Aplicar desenfoque para eliminar ruido
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convertir frame de BRG a HSV
        thresh = cv2.inRange(hsv, dic_colors.get("lower_color_"+color), dic_colors.get("upper_color_"+color)) #Aplicar umbral a img y extraer los pixeles en el rango de colores
        cnts, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # Encontrar los contornos en la imagen extraída
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if area > 300:
                cx, cy = self.cen_moments(cnt)
                self.image_blue, self.x_z_object, self.x_width_object = self.put_obj_in_map(cx,cy,round(self.obj_vision.depth_image[cy,cx]/10,2), self.image_blue, cnt, self.var_limits_inside_object, 0)
        
        cv2.imwrite("evidence_image/frame_blue.png", self.image_blue) 
        cv2.imshow('blue object', self.image_blue)
        return self.x_z_object, self.x_width_object

    def search_lines(self, image_line, color):
        self.image_color = image_line
        filered = cv2.GaussianBlur(self.image_color, (5, 5), 0)  # (7,7),2
        hsv = cv2.cvtColor(filered, cv2.COLOR_BGR2HSV) # Convertir frame de BRG a HSV
        thresh = cv2.inRange(hsv, dic_colors.get("lower_color_"+color), dic_colors.get("upper_color_"+color)) #Aplicar umbral a img y extraer los pixeles en el rango de colores        

        bw = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2) # ADAPTIVE_THRESH_MEAN_C
        # HORIZONTAL --------------------------------------------------------------------------------------
        horizontal = np.copy(bw)

        # Specify size on horizontal axis
        cols = horizontal.shape[1]
        horizontal_size = cols / 20
        horizontal_size=int(horizontal_size)

        # Create structure element for extracting horizontal lines through morphology operations
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

        # Apply morphology operations
        horizontal = cv2.erode(horizontal, horizontalStructure)
        horizontal = cv2.dilate(horizontal, horizontalStructure) 

        horizontal = cv2.bitwise_not(horizontal)
        # Step 1
        edges = cv2.adaptiveThreshold(horizontal, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
        # Step 2
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel)
        # Step 3
        smooth = np.copy(horizontal)
        # Step 4
        smooth = cv2.blur(smooth, (2, 2))
        # Step 5
        (rows, cols) = np.where(edges != 0) #(edges != 0)
        horizontal[rows, cols] = smooth[rows, cols]
        
        # HoughLines ----------------------------------------------------------
        #cv2.Canny(horizontal, rango menor, rango mayor, apertureSize=3)
        dst = cv2.Canny(horizontal, 25, 300, apertureSize=3) #200, 300 
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

        lines = cv2.HoughLines(dst, 3, (90*np.pi/180), 100) # horizontal (90* np.pi)/180, 225 , vertical np.pi/180
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(image_line, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.imshow("cdst", cdst) 

        return image_line

    # Line ends filter
    def lineEnds(P):
        """Central pixel and just one other must be set to be a line end"""
        return 255 * ((P[4]==255) and np.sum(P)==510)
        
    def seach_lines(self):
        return None

    def search_line_goal(self):
        return None

    #Buscar balon, porteria y portero
    def search_objs(self, frame):
        # Iniciar a contar tiempo para contar FPS
        start_time = time.time()
        # Convertir a float32 para TF
        batch_data = tf.constant(self.preprocessToTF(frame))
        # Crear predicciones
        pred_bbox = self.infer(batch_data)
        # obtener configuracion de cuadros
        boxes,pred_conf = self.configBox(pred_bbox)
        #
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes = tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores = tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
        )
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        # Dibujar cuadro de reconocimiento
        image = self.draw_bbox(frame,pred_bbox,self.file_names)

        result = np.asarray(image)
        # Calcular FPS
        fps = 1.0 / (time.time() - start_time)
        # MOstrar FPS
        print("FPS: %.2f" % fps)
        return result

    #Main-----------------------------------------------------------------------------------------------------------------------------------------------
    def main(self):
        x_graphics = []
        y_graphics = []
        num_frames_limit = 60
        num_frames_count = 0

        true_positive_input = 0
        false_positive_input = 0
        true_negative_input = 0
        false_negative_input = 0

        try: #Streaming loop
            while True:

                num_frames_count += 1
                print(num_frames_count)
                if num_frames_count < 10:
                    pass
                else:
                    self.bg_removed = self.obj_vision.get_image_depth() #Eliminar pixeles mayores a 3 metros
                    #self.bg_removed = self.color_image
                    self.bg_removed_green = self.filter_color(self.bg_removed.copy(), "green") #Filtro color verde
                    self.bg_removed_green_blue = self.filter_color(self.bg_removed_green.copy(), "blue") #Filtro color azul 

                    #new_image = self.seg_superpix(bg_removed_green)                    
                    #self.search_lines(self.bg_removed_green_blue.copy(), "white")

                    result_objs = self.search_objs(self.bg_removed.copy()) #solo bg_removed
                    cz_blue_real, width_object = self.search_blue(self.bg_removed_green.copy(), "blue") #Search blue obstacles
                    #Search lines
                    #lines_yellow = self.search_lines(self.bg_removed_green.copy(), "yellow") 
                    #cv2.imshow("lineas yellow", lines_yellow)                                    
                    #lines_white = self.search_lines(self.bg_removed_green.copy(), "white")    
                    #cv2.imshow("lineas white", lines_white)              
                    #Mostrar resultado
                    cv2.namedWindow("result-objs", cv2.WINDOW_AUTOSIZE)
                    cv2.imwrite("evidence_image/search_objs.png", result_objs)
                    cv2.imshow("result-objs", result_objs)

                    #cv2.imshow('xxx blue object', self.bg_removed)
                    #cv2.imwrite("evidence_image/framegreen.png", bg_removed_green_blue) 

                    #cv2.imshow('blue object', img_blue)

                    #self.distance_array_obj.append(cz_blue_real)
                    #self.width_array_obj.append(width_object)

                    #self.graphic_roc()

                    #COMMENT PLOT
                    self.obj_mapeo.Aplot_ball_robot()

                # Salir del bucle si se presiona ESC
                k = cv2.waitKey(5) & 0xFF
                if k == 27:
                    break
                if k == ord("q"):
                    print("That's all folks :)")
                    break

        finally:
            pass

"""if __name__ == '__main__':
    obj_procesamiento = Aprocesamiento()
    #print(str(obja.clipping_distance_in_meters)) #Obtener un valor de la clase
    obj_procesamiento.main()"""