# Importar las librerías OpenCV y Numpy
import cv2
import numpy as np
import imutils

# Ancho y alto del video que se desea capturar
frame_w = 640
frame_h = 480
# Coordenadas del centro del video
fcx = int(frame_w/2)
fcy = int(frame_h/2)

# Definir rango de color a identificar (HSV)
#lower_color = np.array([90, 90, 90], dtype=np.uint8)
#upper_color = np.array([120, 255, 255], dtype=np.uint8)
lower_color = np.array([90, 50, 50 ], dtype=np.uint8)#AZUL
upper_color = np.array([135, 255, 255], dtype=np.uint8)#AZUL

#lower_color = np.array([25, 70, 70 ], dtype=np.uint8) #AMARILLO
#upper_color = np.array([35, 255, 255], dtype=np.uint8) #AMARILLO
#low_blue = np.array([100,50,50]) #35, 100, 20, 255 / 25, 52, 72 ideal->25, 52, 20
#high_blue = np.array([135,255,255]) #102, 255, 255

# Iniciar captura de video con el tamaño deseado
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("carp.mp4")

# Repetir mientras halla señal de video
while True:
    # Leer un frame
    _, img = cap.read()
    img = imutils.resize(img, width=640)
    # Aplicar desenfoque para eliminar ruido
    frame = cv2.blur(img, (15, 15))
    # Convertir frame de BRG a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Aplicar umbral a la imagen y extraer los pixeles en el rango de colores
    thresh = cv2.inRange(hsv, lower_color, upper_color)
    # Encontrar los contornos en la imagen extraída
    cnts, h = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno de mayor área y especificarlo como best_cnt
    max_area = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)

        if area > 2000:
        #    max_area = area
            best_cnt = cnt
            #print(area)
            # Ejecutar este bloque solo si se encontró un área
            #if max_area > 0:
                # Encontrar el centroide del mejor contorno y marcarlo
            M = cv2.moments(best_cnt)
            if M["m00"] != 0:
                cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
            else:
                cx, cy = 0, 0  # set values as what you need in the situation
            cv2.circle(img, (cx, cy), 5, 255, -1)
                # Dibujar un rectángulo alrdedor del objeto
            x, y, w, h = cv2.boundingRect(best_cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Dibujar una línea del centro del frame al centroide
                #cv2.line(img, (cx, cy), (fcx, fcy), 255, 2)

                # print(errx, erry, px.level, py.level)
            font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(img, str(cx), (30, 30), font, 0.8, 255, 2, 8)
            #cv2.putText(img, str(cy), (120, 30), font, 0.8, 244, 2, 8)
                #cv2.putText(img, str(px.level), (210, 30), font, 0.8, 244, 2, 8)
                #cv2.putText(img, str(py.level), (280, 30), font, 0.8, 244, 2, 8)

    # Mostrar la imagen original con todos los overlays
    cv2.imshow('img', img)
    # Mostrar la máscara con los pixeles extraídos
    cv2.imshow('thresh', thresh)

    # Salir del bucle si se presiona ESC
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
