from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

plt.figure(0).clf()

pred = np.random.randint(2, size=1000)
label_real = np.random.randint(2, size=1000)
fpr, tpr, thresh = metrics.roc_curve(label_real, pred) #(true_positive, false_positive)
auc = metrics.roc_auc_score(label_real, pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=0)
plt.savefig('metrics_roc.png')

#True  Negative [TN] : No hay, sistema dice no hay
#True P ositive [TP] : Hay azul y sí hay azul
#False Positive [FP] : No hay azul, sistema dice que sí hay
#False Negative [FN] : Hay azul, sistema dice que no 

#			 |	Predicción 0 | Predicción 1  |
#Realidad 0  |		TN 		 |		FP       |
#Realidad 1  |		FN 		 |		TP       |

class Ametricas:
    def __init__(self, obj_mapeo, obj_robot):
    	pass

    def metricas(self):
    	pass

    #Main-----------------------------------------------------------------------------------------------------------------------------------------------
    def main(self):
    	pass

    def hist_image(self, image): #Know histogram of image
        color = ('b','g','r')
        for i, c in enumerate(color):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color = c)
            plt.savefig('hist_image.png')
                
    #Graficar-------------------------------------------------------------------------------------------------------------------------------------------
    def graficas(self, number):
        #Generamos una grafica lineal para una recta en X
        #plt.plot(self.numbers_array_obj,self.distance_array_obj,label='Distancia '+str(number)+' cm')
        plt.plot(self.numbers_array_obj,self.width_array_obj,label='Dimensión ancho '+str(number)+' cm')
        plt.legend()
        plt.xlabel('Número de muestra')
        plt.ylabel('Ancho medido')
        plt.title('Pruebas de medidas ancho, objeto azul')
        plt.grid()

        plt.show(block=False)
        plt.savefig('graphics_blue_object.png')

    #Graficar ROC Table-------------------------------------------------------------------------------------------------------------------------------------------
    def roc_graphics(self, fpr, tpr, number):
        # Print ROC curve
        plt.plot(tpr, fpr, 'o',label='Muestras: '+str(number))
        plt.legend()
        plt.xlabel('Especificidad, TPR')
        plt.ylabel('Sensibilidad, FPR')
        plt.title('ROC')
        plt.xlim(0, 1)     # set the xlim to left, right
        plt.ylim(0, 1)     # set the xlim to left, right
        plt.grid(True)
        plt.axis("equal")
        plt.show(block=False)
        plt.savefig('metrics_roc_x.png')
        #True  Negative [TN] : No hay, sistema dice no hay
        #True P ositive [TP] : Hay azul y sí hay azul
        #False Positive [FP] : No hay azul, sistema dice que sí hay
        #False Negative [FN] : Hay azul, sistema dice que no 

        #            |  Predicción 0 | Predicción 1  |
        #Realidad 0  |      TN       |      FP       |
        #Realidad 1  |      FN       |      TP       |

    def graphic_roc(self):
        roc_input = int(input("ROC input, true/false Positive: "))
        if roc_input == 0:
            true_positive_input += 1
        elif roc_input == 1:
            false_positive_input +=1
        elif roc_input == 2:
            true_negative_input +=1
        elif roc_input == 3:
            false_negative_input +=1
        else:
            plt.plot([1],[1],label='Línea de no discriminación')
        
        #true_positive_input = int(input("Real input: "))
        #false_positive_input = int(input("Pred input: "))

        #self.real_array_roc.append(real_input)
        #self.pred_array_roc.append(pred_input)

        #hi

        #True  Negative [TN] : No hay, sistema dice no hay
        #True P ositive [TP] : Hay azul y sí hay azul
        #False Positive [FP] : No hay azul, sistema dice que sí hay
        #False Negative [FN] : Hay azul, sistema dice que no 

        #            |  Predicción 0 | Predicción 1  |
        #Realidad 0  |      TN       |      FP       |
        #Realidad 1  |      FN       |      TP       |

    # Press esc or 'q' to close the image window, 
        if num_frames_count >= num_frames_limit:

            #self.graficas(num_dist_actual)
            print("true_positive_input: "+str(true_positive_input))
            print("false_positive_input: "+str(false_positive_input))
            tpr = true_positive_input/(true_positive_input+false_negative_input)
            if false_positive_input != 0:
                fpr = false_positive_input/(false_positive_input+true_negative_input)
            else:
                fpr = 0
            print("Result TPR: "+str(tpr))
            print("Result FPR: "+str(fpr))
        
            #num_frames_limit = 0

            self.roc_graphics(tpr, fpr, num_frames_limit-10)

            #self.distance_array_obj = [0]
            #self.width_array_obj = [0]
            #num_frames_limit = 0
            #num_dist_actual += 60

            #input("Press any key to continue the program")

            #self.graficas(x_graphics, y_graphics) #Graficar para pruebas
            #self.roc_graphics()
            #cv2.destroyAllWindows()
            #break
            num_frames_count = 0
            num_frames_limit += 50
            true_positive_input = 0
            false_positive_input = 0
            true_negative_input = 0
            false_negative_input = 0

"""if __name__ == '__main__':
    obj_metricas = Ametricas()
    #print(str(obja.clipping_distance_in_meters)) #Obtener un valor de la clase
    obj_metricas.main()"""
