import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def visualize_missing_products(image, comparison_results):
    """
    Visualiza espacios donde faltan productos según el planograma
    
    Args:
        image: Imagen original
        comparison_results: Resultados de la comparación con el planograma
        
    Returns:
        Imagen con espacios faltantes resaltados
    """
    # Crear una copia de la imagen para dibujar
    vis_image = image.copy()
    height, width = vis_image.shape[:2]
    
    # Convertir a PIL para usar funciones avanzadas de dibujo
    pil_image = Image.fromarray(vis_image)
    draw = ImageDraw.Draw(pil_image)
    
    # Intentar cargar fuente, si no usar default
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    # Color rojo para espacios faltantes
    red_color = (255, 0, 0)
    
    # Filtrar productos faltantes
    missing_products = [p for p in comparison_results if p["Estado"] == "Falta"]
    
    # Si no hay productos faltantes, devolver la imagen original
    if not missing_products:
        return vis_image
    
    # Definir coordenadas fijas para cada charola (basadas en divisiones de la imagen)
    # Formato: [y_start, y_end, divisiones_horizontales]
    shelf_coords = {
        1: [int(height * 0.05), int(height * 0.25), 6],  # Charola superior (snacks)
        2: [int(height * 0.30), int(height * 0.50), 8],  # Charola media superior
        3: [int(height * 0.51), int(height * 0.70), 8],  # Charola media inferior
        4: [int(height * 0.71), int(height * 0.90), 8]   # Charola inferior
    }
    
    # INVERTIR ORDEN: Crear mapa para invertir charolas físicamente
    # Si es charola 1 → mostrar en posición de charola 4
    # Si es charola 4 → mostrar en posición de charola 1
    charola_invertida = {
        1: 4,  # La charola 1 se muestra en posición de charola 4
        2: 3,  # La charola 2 se muestra en posición de charola 3
        3: 2,  # La charola 3 se muestra en posición de charola 2
        4: 1   # La charola 4 se muestra en posición de charola 1
    }
    
    # Para cada producto faltante
    for product in missing_products:
        # Obtener la charola del producto
        charola = product.get("Charola", 1)  # Default a charola 1 si no está definido
        
        # INVERTIR ORDEN: Usar el mapeo para obtener la charola invertida
        charola_mostrar = charola_invertida.get(charola, charola)
        
        # Verificar si tenemos coordenadas para esta charola
        if charola_mostrar not in shelf_coords:
            continue
            
        # Obtener coordenadas de la charola
        y_start, y_end, divisions = shelf_coords[charola_mostrar]
        division_width = width / divisions
        
        # Determinar posición horizontal basada en recomendación o secuencial
        position = 0
        if "Recomendación" in product:
            recomendacion = product["Recomendación"]
            if "posición" in recomendacion:
                try:
                    # Extraer número de posición
                    position = int(''.join(filter(str.isdigit, recomendacion.split("posición")[1].split()[0])))
                    position = position - 1  # Ajustar a índice 0-based
                except:
                    position = 0
        
        # Asegurar que la posición está en rango
        position = max(0, min(position, divisions - 1))
        
        # Calcular coordenadas del rectángulo
        x1 = int(position * division_width)
        y1 = y_start
        x2 = int(x1 + division_width * 0.9)  # Un poco menos ancho para separación visual
        y2 = y_end
        
        # Dibujar rectángulo rojo semitransparente
        # Usamos PIL directamente en lugar de crear overlay
        red_overlay = Image.new('RGBA', (x2-x1, y2-y1), (255, 0, 0, 128))
        pil_image.paste(red_overlay, (x1, y1), red_overlay)
        
        # Añadir etiqueta con el nombre del producto
        label = f"FALTA: {product['Clase']}"
        draw.text((x1 + 5, y1 + 5), label, fill=(255, 255, 255), font=font)
    
    # Convertir de vuelta a numpy array
    return np.array(pil_image)
