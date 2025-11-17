import cv2
import numpy as np
import matplotlib.pyplot as plt # type: ignore

def enmarcar_objeto_principal(imagen_path, umbral_blanco=250, area_minima=1000):
    """
    Enmarca solo el objeto principal, ignorando fondo blanco y áreas blancas internas
    """
    # Cargar imagen en escala de grises
    imagen_gris = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    
    # Crear máscara binaria (objeto = negro, fondo = blanco)
    _, mascara = cv2.threshold(imagen_gris, umbral_blanco, 255, cv2.THRESH_BINARY_INV)
    
    # Encontrar todos los contornos
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área (eliminar pequeños ruidos)
    contornos_filtrados = [cnt for cnt in contornos if cv2.contourArea(cnt) > area_minima]
    
    # Crear imagen de resultado
    resultado = cv2.cvtColor(imagen_gris, cv2.COLOR_GRAY2BGR)
    
    if contornos_filtrados:
        # Encontrar el contorno más grande (objeto principal)
        contorno_principal = max(contornos_filtrados, key=cv2.contourArea)
        
        # Obtener el bounding box del objeto principal
        x, y, w, h = cv2.boundingRect(contorno_principal)
        
        # Dibujar el bounding box
        cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 0, 255), 3)
        
        # Dibujar el contorno del objeto
        cv2.drawContours(resultado, [contorno_principal], -1, (0, 255, 0), 10)
        
        # Información
        cv2.putText(resultado, f'Objeto: {w}x{h}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return resultado, mascara, contorno_principal, (x, y, w, h)
    
    return resultado, mascara, None, None

# Ejemplo de uso
imagen_path = 'yu_eguru_narukami.jpg'
resultado, mascara, contorno, bbox = enmarcar_objeto_principal(imagen_path)

plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE), cmap='gray'), plt.title('Original')
plt.subplot(132), plt.imshow(mascara, cmap='gray'), plt.title('Máscara Objeto')
plt.subplot(133), plt.imshow(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)), plt.title('Objeto Enmarcado')
plt.show()