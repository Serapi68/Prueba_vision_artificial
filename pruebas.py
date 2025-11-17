import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt # type: ignore

img = cv.imread('yu_eguru_narukami.jpg')

gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #Pasamos imagen a blanco y negro

blurred = cv.GaussianBlur(img, (5, 5), 0) #Aplicamos un desenfoque gaussiano para reducir el ruido

mediana = cv.medianBlur(img,5)#Aplicamos un filtro de mediana para reducir el ruido

def filtrar_blanco(img):
    #Definimos el rango de colores para filtrar el blanco
    blanco_bajo = np.array([200, 200, 200]) 
    blanco_alto = np.array([255, 255, 255])
    #Creamos una mascara para extraer las areas blancas
    mascara = cv.inRange(img, blanco_bajo, blanco_alto)
    mascara_blanco = cv.inRange(gris,0,0,200)
    #Eliminamos el ruido de la mascara usando operaciones morfologicas
    mascara_sin_ruido = mediana
    #Aplicamos la mascara a la imagen original
    resultado = cv.bitwise_and(img, img, mascara)
    
    return resultado, mascara_blanco, mascara_sin_ruido

resultado, mascara_blanco, mascara_sin_ruido = filtrar_blanco(img)
plt.figure(figsize=(10,10))
plt.subplot(1,3,1), plt.imshow(cv.cvtColor(resultado, cv.COLOR_BGR2RGB)), plt.title('Imagen Filtrada'), plt.axis('off')
plt.subplot(1,3,2), plt.imshow(mascara_blanco, cmap='gray'), plt.title('Mascara Blanco'), plt.axis('off')
plt.subplot(1,3,3), plt.imshow(mascara_sin_ruido, cmap='gray'), plt.title('Mascara sin Ruido'), plt.axis('off')
plt.show()                  






