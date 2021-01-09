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