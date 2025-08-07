import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import os
import json
import uuid
import base64
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from roboflow import Roboflow
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualize_missing import visualize_missing_products
from collections import Counter

# Configurar página
st.set_page_config(
    page_title="Shell-E", 
    page_icon="🐚", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #34a853 0%, #1e8e3e 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #34a853;
        margin-top: 2rem;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        background-color: transparent;
        border: 1px solid rgba(52, 168, 83, 0.3);
    }
    .success-metric {
        font-size: 1.5rem;
        font-weight: bold;
        color: #34a853;
    }
    .warning-metric {
        font-size: 1.5rem;
        font-weight: bold;
        color: #fbbc05;
    }
    .error-metric {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ea4335;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #5f6368;
    }
    .btn-primary {
        background-color: #34a853;
        color: white;
        padding: 10px 24px;
        border-radius: 4px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border: none;
    }
    .btn-primary:hover {
        background-color: #1e8e3e;
    }
    .sidebar .sidebar-content {
        background-color: transparent;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #202124;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .dataframe-container {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(52, 168, 83, 0.3);
    }
    .stDataFrame {
        border-radius: 10px !important;
        background-color: transparent !important;
    }
    /* Remove white backgrounds from Streamlit elements */
    div.stBlock, div.stDownloadButton, div.stFileUploader, 
    div.stSelectbox, div.stNumberInput, div.stSlider, div.stTextInput {
        background-color: transparent !important;
    }
    .stAlert {
        background-color: rgba(52, 168, 83, 0.1) !important;
        border-color: #34a853 !important;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<h1 class='main-header'>🐚 Shell-E</h1>", unsafe_allow_html=True)

# Inicializar estado de la sesión
if 'image' not in st.session_state:
    st.session_state.image = None
if 'csv_data' not in st.session_state:
    st.session_state.csv_data = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'comparison_results' not in st.session_state:
    st.session_state.comparison_results = None
if 'detections' not in st.session_state:
    st.session_state.detections = None
if 'highlighted_image' not in st.session_state:
    st.session_state.highlighted_image = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'mode' not in st.session_state:
    st.session_state.mode = "Simple"
if 'empty_spaces' not in st.session_state:
    st.session_state.empty_spaces = []
if 'empty_spaces_image' not in st.session_state:
    st.session_state.empty_spaces_image = None
if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {
        "total_productos": 0,
        "ok_productos": 0,
        "falta_productos": 0,
        "sobra_productos": 0,
        "mover_productos": 0,
        "porcentaje_cumplimiento": 0
    }
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# Funciones para cargar y ejecutar el modelo
@st.cache_resource
def load_model():
    """Cargar el modelo de Roboflow"""
    rf = Roboflow(api_key="KkzFysqU16FP7cfNZARz")
    project = rf.workspace("hackathon-cud1g").project("custom-workflow-object-detection-oidmg")
    model = project.version(8).model
    return model

def process_image(image, confidence=0.5):
    """Procesar imagen con el modelo y retornar detecciones"""
    # Guardar imagen temporalmente para procesarla con Roboflow
    temp_img_path = "/tmp/temp_image.jpg"
    cv2.imwrite(temp_img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Cargar modelo
    model = load_model()
    
    # Realizar predicciones
    predictions = model.predict(temp_img_path, confidence=confidence)
    predictions_json = predictions.json()
    
    # Opcional: Eliminar archivo temporal
    try:
        os.remove(temp_img_path)
    except:
        pass
    
    # Convertir predicciones al formato necesario
    detections = []
    for prediction in predictions_json["predictions"]:
        detection = {
            "class": prediction["class"],
            "confidence": prediction["confidence"],
            "x": int(prediction["x"]),
            "y": int(prediction["y"]),
            "width": int(prediction["width"]),
            "height": int(prediction["height"]),
            "bbox": [
                int(prediction["x"] - prediction["width"] / 2),
                int(prediction["y"] - prediction["height"] / 2),
                int(prediction["x"] + prediction["width"] / 2),
                int(prediction["y"] + prediction["height"] / 2)
            ]
        }
        detections.append(detection)
    
    return detections

def draw_detections(image, detections, box_thickness=3, use_custom_colors=True):
    """Dibujar bounding boxes y etiquetas en la imagen con opciones avanzadas"""
    img = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(img)
    
    # Intentar cargar una fuente, si no está disponible usar la predeterminada
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Definir colores por clase para consistencia visual si se requiere
    class_colors = {}
    
    for detection in detections:
        # Asignar un color para esta clase si aún no existe
        if detection["class"] not in class_colors and use_custom_colors:
            # Usar colores distintivos para clases diferentes
            hue = hash(detection["class"]) % 360 / 360.0  # Valor entre 0 y 1 basado en el hash de la clase
            rgb = tuple(int(i * 255) for i in plt.cm.hsv(hue)[:3])
            class_colors[detection["class"]] = rgb
        else:
            # Si no usamos colores personalizados, usar colores aleatorios
            if detection["class"] not in class_colors:
                class_colors[detection["class"]] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Obtener coordenadas del bounding box
        x1, y1, x2, y2 = detection["bbox"]
        
        # Calcular coordenadas para mostrar texto usando textbbox
        text_bbox = draw.textbbox((0, 0), detection["class"], font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Añadir confianza al texto si está disponible
        display_text = detection["class"]
        if "confidence" in detection:
            display_text = f"{detection['class']} ({detection['confidence']:.2f})"
            
        # Recalcular dimensiones del texto con confianza
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        
        # Obtener color para esta clase
        color = class_colors[detection["class"]]
        
        # Dibujar rectángulo con grosor personalizado
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)
        
        # Dibujar fondo para texto con más padding para mejor legibilidad
        draw.rectangle([x1, y1 - text_height - 8, x1 + text_width + 8, y1], fill=color)
        
        # Dibujar texto con sombra para mejor contraste
        draw.text((x1 + 4 + 1, y1 - text_height - 4 + 1), display_text, fill="black", font=font)  # sombra
        draw.text((x1 + 4, y1 - text_height - 4), display_text, fill="white", font=font)  # texto principal
    
    return np.array(img)

def crop_detection(image, detection, padding=10):
    """Recortar una detección de la imagen original con padding opcional"""
    x1, y1, x2, y2 = detection["bbox"]
    
    # Añadir padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    
    return image[y1:y2, x1:x2]

def get_thumbnail(image, detection, size=(150, 150), padding=10):
    """Generar miniatura para una detección con tamaño personalizable"""
    crop = crop_detection(image, detection, padding)
    if crop.size == 0:  # Si el recorte está vacío
        return None
    
    # Convertir a PIL para redimensionar manteniendo proporción
    pil_crop = Image.fromarray(crop)
    pil_crop.thumbnail(size, Image.LANCZOS)
    
    return np.array(pil_crop)

def highlight_status(val):
    """Función para resaltar estados en DataFrames"""
    if val == "OK":
        return 'background-color: #d4edda; color: #155724'
    elif val == "Falta":
        return 'background-color: #f8d7da; color: #721c24'
    elif val == "Sobra":
        return 'background-color: #fff3cd; color: #856404'
    elif val == "Mover":
        return 'background-color: #d1ecf1; color: #0c5460'
    return ''

# Funciones avanzadas para análisis de planogramas
def generate_heatmap(image, detections, resolution=100):
    """Generar un mapa de calor de la densidad de productos"""
    # Crear una matriz vacía para el mapa de calor
    height, width = image.shape[:2]
    heatmap = np.zeros((int(height/resolution), int(width/resolution)))
    
    # Para cada detección, incrementar el valor en la posición correspondiente
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
        x_bin, y_bin = x_center // resolution, y_center // resolution
        
        # Asegurarse de que estamos dentro de los límites
        if 0 <= x_bin < heatmap.shape[1] and 0 <= y_bin < heatmap.shape[0]:
            heatmap[y_bin, x_bin] += 1
    
    return heatmap

def calculate_metrics(comparison_results):
    """Calcular métricas de desempeño a partir de los resultados"""
    metrics = {
        "total_productos": len(comparison_results),
        "ok_productos": 0,  # Mantenemos pero ya no se usa
        "falta_productos": 0,
        "sobra_productos": 0,
        "mover_productos": 0,
        "charolas": {},
        "score": 0.0
    }
    
    # Contar productos por estado
    for result in comparison_results:
        estado = result["Estado"]
        charola = result["Charola"]
        
        # Inicializar contador para esta charola si no existe
        if charola not in metrics["charolas"]:
            metrics["charolas"][charola] = {
                "total": 0,
                "falta": 0,
                "sobra": 0,
                "mover": 0,
                "score": 0.0
            }
        
        # Incrementar contadores
        metrics["charolas"][charola]["total"] += 1
        if estado == "Falta":
            metrics["falta_productos"] += 1
            metrics["charolas"][charola]["falta"] += 1
        elif estado == "Sobra":
            metrics["sobra_productos"] += 1
            metrics["charolas"][charola]["sobra"] += 1
        elif estado == "Mover":
            metrics["mover_productos"] += 1
            metrics["charolas"][charola]["mover"] += 1
    if "sobra_productos" not in metrics or metrics["sobra_productos"] is None:
        metrics["sobra_productos"] = 0
    if "mover_productos" not in metrics or metrics["mover_productos"] is None:
        metrics["mover_productos"] = 0
        
    # Calcular score por charola y global - basado solo en acciones necesarias
    for charola in metrics["charolas"]:
        total = metrics["charolas"][charola]["total"]
        falta = metrics["charolas"][charola]["falta"]
        sobra = metrics["charolas"][charola]["sobra"]
        mover = metrics["charolas"][charola]["mover"]
        
        # Calcular score basado en acciones necesarias (penalización)
        problemas = falta + sobra + mover
        if total > 0:
            metrics["charolas"][charola]["score"] = max(0, 100 - (problemas / total) * 100)
        
    # Calcular score ponderado (penalizando más los faltantes)
    ponderacion_falta = 1.5  # Penalizar más los faltantes
    ponderacion_sobra = 1.0
    ponderacion_mover = 0.5  # Penalizar menos los que solo necesitan moverse
    
    penalizacion = (
        metrics["falta_productos"] * ponderacion_falta + 
        metrics["sobra_productos"] * ponderacion_sobra + 
        metrics["mover_productos"] * ponderacion_mover
    )
    
    # Score final (100 - penalización)
    if metrics["total_productos"] > 0:
        metrics["score"] = max(0, 100 - (penalizacion / metrics["total_productos"]) * 100)
    else:
        metrics["score"] = 0
    
    return metrics

# Función principal de análisis avanzado de planogramas
def analyze_planogram_advanced(detections, expected_planogram):
    """Función mejorada para comparar detecciones con el planograma esperado"""
    
    # Convertir detecciones al formato necesario
    cajas = []
    for idx, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["bbox"]
        cajas.append({
            "class": detection["class"],
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "confidence": detection.get("confidence", 0.5),
            "original_detection": detection
        })
    
    # Si no hay detecciones, retornar lista vacía
    if not cajas:
        st.warning("No se encontraron detecciones en la imagen.")
        return []
    
    # Obtener altura de la imagen
    image_height = max([c["y"] + c["height"] for c in cajas])
    
    # Generar líneas horizontales cada 60 px
    lineas = list(range(int(image_height), 0, -60))
    linea_a_cajas = {}
    for y_linea in lineas:
        linea_a_cajas[y_linea] = []
        for idx, caja in enumerate(cajas):
            y_caja, h_caja = caja['y'], caja['height']
            if y_caja <= y_linea <= y_caja + h_caja:
                linea_a_cajas[y_linea].append(idx)
    
    # Seleccionar líneas óptimas
    def clases_aportadas(y_linea):
        return set(cajas[i]['class'] for i in linea_a_cajas[y_linea])
    
    # Ordenar líneas por cantidad de clases que aportan
    lineas_con_cajas = [(y, ids) for y, ids in linea_a_cajas.items() if ids]
    lineas_ordenadas = sorted(
        lineas_con_cajas,
        key=lambda x: len(clases_aportadas(x[0])),
        reverse=True
    )
    
    clases_totales = set(c['class'] for c in cajas)
    clases_cubiertas = set()
    lineas_finales = []
    
    # Seleccionar líneas que cubran todas las clases
    for y_linea, indices in lineas_ordenadas:
        nuevas = clases_aportadas(y_linea) - clases_cubiertas
        if nuevas:
            lineas_finales.append({'y': y_linea, 'cajas': indices})
            clases_cubiertas |= nuevas
        if clases_cubiertas == clases_totales:
            break
    
    # Detectar columnas del CSV
    cols = expected_planogram.columns.tolist()
    
    # Buscar columnas de charola, posición y nombre
    shelf_col = None  # Columna de charola/nivel
    pos_col = None    # Columna de posición
    name_col = None   # Columna de nombre/producto
    
    # Búsqueda de columnas por términos comunes
    shelf_terms = ['charola', 'nivel', 'shelf', 'cha', 'niv']
    pos_terms = ['posicion', 'posición', 'posi', 'pos', 'number', 'num', 'orden']
    name_terms = ['producto', 'nombre', 'name', 'class', 'clase', 'item', 'articulo', 'artículo']
    
    for col in cols:
        col_lower = col.lower()
        if any(term in col_lower for term in shelf_terms):
            shelf_col = col
        if any(term in col_lower for term in pos_terms):
            pos_col = col
        if any(term in col_lower for term in name_terms):
            name_col = col
    
    # Si no encontramos alguna columna, usar heurísticas
    if not shelf_col and len(cols) > 0:
        shelf_col = cols[0]
        
    if not pos_col and len(cols) > 1:
        for col in cols:
            try:
                if expected_planogram[col].dtype in ('int64', 'float64') or \
                   expected_planogram[col].astype(str).str.isnumeric().all():
                    pos_col = col
                    break
            except:
                continue
        if not pos_col and len(cols) > 1:
            pos_col = cols[1]
    
    if not name_col and len(cols) > 2:
        name_col = cols[2]
    elif not name_col and len(cols) > 0:
        name_col = cols[0]
        
    # 5) Construcción de la comparación línea a línea
    # Ordenar líneas por posición vertical (Y), de arriba hacia abajo
    lineas_finales.sort(key=lambda x: x['y'])  # Ordenar de arriba a abajo: Y menor = más arriba
    comparaciones = []
    
    # Determinar números de charola en el CSV
    charolas_en_csv = set()
    try:
        # Intentar obtener números de charola como enteros
        charolas_en_csv = set(expected_planogram[shelf_col].astype(float).astype(int).unique())
    except Exception:
        try:
            # Intentar con formato de texto
            charolas_en_csv = set([int(float(x)) for x in expected_planogram[shelf_col].astype(str).str.strip().unique() 
                                  if x.strip().replace('.', '', 1).isdigit()])
        except Exception:
            # Si no se pueden extraer charolas, utilizar números secuenciales
            charolas_en_csv = set(range(1, len(lineas_finales) + 1))
    
    # Obtener lista ordenada de números de charola
    charolas_ordenadas = sorted(list(charolas_en_csv))
    num_charolas = len(charolas_ordenadas)
    
    # SOLUCIÓN EXPLÍCITA: Mapeo directo de líneas detectadas a números de charola
    # Para un anaquel típico de 4 niveles:
    # - Nivel superior (idx=0, Y menor) = Charola 4 
    # - Segundo nivel (idx=1) = Charola 3
    # - Tercer nivel (idx=2) = Charola 2
    # - Nivel inferior (idx=3, Y mayor) = Charola 1
    
    # Limitar el número de líneas al número de charolas disponibles
    lineas_a_procesar = lineas_finales[:min(len(lineas_finales), num_charolas)]
    
    # Generar mapeo directo y explícito entre posiciones y números de charola
    mapeo_linea_a_charola = {}
    
    # Crear mapeo explícito: posición vertical -> número de charola
    for idx, linha in enumerate(lineas_a_procesar):
        # Mapeo inverso: índice 0 (arriba) -> charola más alta numéricamente
        # Índice max (abajo) -> charola más baja numéricamente (usualmente 1)
        charola_idx = num_charolas - idx - 1
        if charola_idx >= 0 and charola_idx < len(charolas_ordenadas):
            mapeo_linea_a_charola[idx] = charolas_ordenadas[charola_idx]
        else:
            mapeo_linea_a_charola[idx] = idx + 1
    
    # Procesar cada línea con su asignación de charola correspondiente
    for idx_linea, linha in enumerate(lineas_a_procesar):
        y = linha['y']
        
        # Usar el mapeo directo para asignar el número de charola
        charola_num = mapeo_linea_a_charola[idx_linea]
        
        # Filtrar el CSV por esa charola
        try:
            esperado_df = expected_planogram[expected_planogram[shelf_col].astype(str).str.strip() == str(charola_num)]
            # Si está vacío, intentar convertir a número
            if esperado_df.empty:
                esperado_df = expected_planogram[expected_planogram[shelf_col].astype(float).astype(int) == charola_num]
        except Exception as e:
            esperado_df = pd.DataFrame()
        
        # Ordenar por posición si es posible
        if not esperado_df.empty and pos_col is not None:
            try:
                esperado_df = esperado_df.sort_values(by=pos_col)
            except Exception as e:
                pass
        
        # Obtener nombres esperados
        try:
            if not esperado_df.empty and name_col in esperado_df.columns:
                expected_names = esperado_df[name_col].astype(str).tolist()
            else:
                expected_names = []
        except Exception as e:
            expected_names = []
            
        # Ordenar detecciones de izquierda a derecha
        detected_idxs = sorted(linha['cajas'], key=lambda i: cajas[i]['x'])
        detected_names = [cajas[i]['class'] for i in detected_idxs]
        
        # Comparar posición a posición
        max_pos = max(len(expected_names), len(detected_names))
        detalle = []
        
        for p in range(max_pos):
            exp = expected_names[p] if p < len(expected_names) else None
            det = detected_names[p] if p < len(detected_names) else None
            # Verificar coincidencia usando lowercase para mayor flexibilidad
            match = bool(exp and det and (det.lower() in exp.lower() or exp.lower() in det.lower()))
            
            detalle_item = {
                'posicion': p+1,
                'esperado': exp,
                'detectado': det,
                'coincide': match
            }
            
            # Añadir información de la detección original si existe
            if det and p < len(detected_idxs):
                idx = detected_idxs[p]
                if idx < len(cajas):
                    detalle_item['detection'] = cajas[idx]['original_detection']
            
            # Añadir a la lista de detalles
            detalle.append(detalle_item)
        
        comparaciones.append({
            'charola': charola_num,
            'y': y,
            'detalle': detalle
        })
    
    # 6) Generar instrucciones de ajuste de forma inteligente
    def generar_instrucciones_ajuste(comparaciones):
        # Construimos un JSON con todas las instrucciones organizadas por charola
        resultado_json = []
        
        for comp in comparaciones:
            charola = comp['charola']
            detalle = comp['detalle']
            
            # Filtrar posiciones inválidas (esperado=None)
            detalle_valido = [d for d in detalle if d['esperado'] is not None]
            
            # Extraer listas de nombres para operaciones
            expected = [d['esperado'] for d in detalle_valido]
            detected = [d['detectado'] for d in detalle if d['detectado'] is not None]
            
            # Estructurar acciones para esta charola
            acciones_charola = {
                "charola": charola,
                "y": comp['y'],
                "instrucciones": {
                    "mover": [],  # Elementos que deben moverse a otra posición
                    "eliminar": [],  # Elementos que deben eliminarse
                    "agregar": []  # Elementos que deben agregarse
                },
                "resumen": ""
            }
            
            # 1. FASE DE MOVIMIENTOS: Identificar elementos en posición incorrecta
            for d in detalle_valido:
                pos = d['posicion']
                det = d['detectado']
            
                # Solo nos importan los que existen y no coinciden
                if det and not d['coincide'] and det in expected:
                    pos_correcta = expected.index(det) + 1
                    
                    # Obtener detalle de la posición destino
                    destino = next((item for item in detalle if item['posicion'] == pos_correcta), None)
                
                    # Solo movemos si allí NO hay ya un elemento correcto
                    if destino and not destino['coincide']:
                        acciones_charola["instrucciones"]["mover"].append({
                            "producto": det,
                            "desde": pos,
                            "hacia": pos_correcta,
                            "detection": d.get('detection')
                        })
                    else:
                        acciones_charola["instrucciones"]["eliminar"].append({
                            "producto": det,
                            "posicion": pos,
                            "detection": d.get('detection')
                        })
            
            # 2. FASE DE ELIMINACIÓN: Identificar elementos que sobran
            elementos_movidos = set(m["producto"] for m in acciones_charola["instrucciones"]["mover"])
            elementos_sobrantes = []
            
            for d in detalle:
                det = d['detectado']
                if det and det not in expected and det not in elementos_sobrantes:
                    elementos_sobrantes.append(det)
                    acciones_charola["instrucciones"]["eliminar"].append({
                        "producto": det,
                        "posicion": d['posicion'],
                        "detection": d.get('detection')
                    })
            
            # 3. FASE DE ADICIÓN: Identificar elementos que faltan
            for idx, exp in enumerate(expected):
                if exp not in detected and exp not in elementos_movidos:
                    acciones_charola["instrucciones"]["agregar"].append({
                        "producto": exp,
                        "posicion": idx + 1
                    })
            
            # 4. Generar resumen de acciones para esta charola
            total_acciones = (
                len(acciones_charola["instrucciones"]["mover"]) +
                len(acciones_charola["instrucciones"]["eliminar"]) +
                len(acciones_charola["instrucciones"]["agregar"])
            )
            
            if total_acciones == 0:
                acciones_charola["resumen"] = f"Charola {charola}: Correcta, no requiere ajustes."
            else:
                resumen = f"Charola {charola}: Requiere {total_acciones} ajustes - "
                partes = []
                
                if acciones_charola["instrucciones"]["mover"]:
                    partes.append(f"mover {len(acciones_charola['instrucciones']['mover'])} productos")
                if acciones_charola["instrucciones"]["eliminar"]:
                    partes.append(f"eliminar {len(acciones_charola['instrucciones']['eliminar'])} productos")
                if acciones_charola["instrucciones"]["agregar"]:
                    partes.append(f"agregar {len(acciones_charola['instrucciones']['agregar'])} productos")
                    
                acciones_charola["resumen"] = resumen + ", ".join(partes) + "."
            
            resultado_json.append(acciones_charola)
        
        return resultado_json
    
    # Generar instrucciones de ajuste
    instrucciones = generar_instrucciones_ajuste(comparaciones)
    
    # 7) Convertir las instrucciones al formato esperado por la aplicación
    comparison_results = []
    
    for charola_data in instrucciones:
        charola = charola_data['charola']
        instr = charola_data['instrucciones']
        
        # Procesar movimientos
        for mov in instr['mover']:
            result_entry = {
                "Charola": charola,
                "Clase": mov['producto'],
                "Estado": "Mover",
                "Recomendación": f"Mover de posición {mov['desde']} a {mov['hacia']}",
                "Detection": mov.get('detection')
            }
            # Añadir aliases para compatibilidad
            result_entry["Nivel"] = charola
            result_entry["Producto"] = mov['producto']
            comparison_results.append(result_entry)
            
        # Procesar eliminaciones
        for elim in instr['eliminar']:
            result_entry = {
                "Charola": charola,
                "Clase": elim['producto'],
                "Estado": "Sobra",
                "Recomendación": f"Retirar de posición {elim['posicion']}",
                "Detection": elim.get('detection')
            }
            # Añadir aliases para compatibilidad
            result_entry["Nivel"] = charola
            result_entry["Producto"] = elim['producto']
            comparison_results.append(result_entry)
            
        # Procesar adiciones
        for add in instr['agregar']:
            result_entry = {
                "Charola": charola,
                "Clase": add['producto'],
                "Estado": "Falta",
                "Recomendación": f"Agregar en posición {add['posicion']}",
            }
            # Añadir aliases para compatibilidad
            result_entry["Nivel"] = charola
            result_entry["Producto"] = add['producto']
            comparison_results.append(result_entry)
    
    # Guardar datos adicionales para visualización avanzada
    st.session_state.lineas_detectadas = lineas_finales
    st.session_state.instrucciones_detalladas = instrucciones
    st.session_state.comparaciones_detalladas = comparaciones
    
    # Calcular métricas para el dashboard
    st.session_state.metrics = calculate_metrics(comparison_results)
    
    # Generar datos para el mapa de calor si no existen
    if st.session_state.heatmap_data is None:
        st.session_state.heatmap_data = generate_heatmap(st.session_state.image, detections)
    
    return comparison_results

# Función para generar visualizaciones avanzadas
def generate_product_distribution_chart(comparison_results):
    """Generar gráfico de distribución de productos por estado y charola"""
    # Crear DataFrame para la visualización
    df = pd.DataFrame(comparison_results)
    
    # Si no hay resultados, devolver un gráfico vacío con mensaje
    if len(df) == 0:
        fig = go.Figure()
        fig.add_annotation(text="No hay datos disponibles", showarrow=False, font=dict(size=20))
        return fig
    
    # Contar productos por charola y estado
    counts = df.groupby(['Charola', 'Estado']).size().reset_index(name='Cantidad')
    
    # Asegurar que todos los estados estén representados en todas las charolas
    all_charolas = df['Charola'].unique()
    all_estados = ['Falta', 'Sobra', 'Mover']
    
    # Crear todas las combinaciones posibles de charola y estado
    complete_index = pd.MultiIndex.from_product([all_charolas, all_estados], names=['Charola', 'Estado'])
    complete_df = pd.DataFrame(index=complete_index).reset_index()
    
    # Fusionar con los conteos reales y rellenar los valores faltantes con 0
    merged_counts = complete_df.merge(counts, on=['Charola', 'Estado'], how='left').fillna(0)
    
    # Crear gráfico con Plotly
    fig = px.bar(merged_counts, x='Charola', y='Cantidad', color='Estado', barmode='group',
                title='Distribución de Productos por Charola y Estado',
                labels={'Charola': 'Número de Charola', 'Cantidad': 'Cantidad de Productos'},
                color_discrete_map={
                    'Falta': '#dc3545',  # Rojo
                    'Sobra': '#ffc107',  # Amarillo
                    'Mover': '#17a2b8'   # Azul
                })
    
    # Mejorar el diseño del gráfico
    fig.update_layout(
        legend_title_text='Estado del Producto',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    # Añadir etiquetas a las barras para mejor legibilidad
    fig.update_traces(texttemplate='%{y}', textposition='outside')
    
    return fig

# Función de radar de cumplimiento eliminada porque dependía de productos OK

# Función para visualizar mapa de calor de productos
def plot_heatmap(heatmap_data, title="Mapa de Densidad de Productos"):
    """Visualizar mapa de calor de densidad de productos"""
    fig = px.imshow(heatmap_data, 
                   labels=dict(x="Posición X", y="Posición Y", color="Densidad"),
                   title=title,
                   color_continuous_scale="Viridis")
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Función para generar imagen con áreas resaltadas por estado
def generate_highlighted_image(image, detections, comparison_results):
    """Generar una imagen con áreas correctas e incorrectas resaltadas visualmente
    
    Args:
        image: Imagen original
        detections: Lista de detecciones
        comparison_results: Resultados de la comparación con el planograma
        
    Returns:
        Imagen con áreas resaltadas por estado
    """
    # Crear una copia de la imagen original
    output_img = image.copy()
    img_height, img_width = output_img.shape[:2]
    
    # Convertir a PIL para dibujar
    output_img = Image.fromarray(output_img)
    draw = ImageDraw.Draw(output_img)
    
    # Crear una capa semitransparente para las etiquetas y áreas
    overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    
    # Intentar cargar una fuente más grande para etiquetas
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small_font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        small_font = font
    
    # Crear un diccionario para mapear detection a su estado
    detection_status = {}
    
    # Procesar los resultados
    for result in comparison_results:
        if "Detection" in result and result["Detection"] is not None:
            # Guardamos la referencia al objeto detection y su estado usando una clave más estable
            # Creamos una clave de cadena basada en las coordenadas del bounding box y la clase
            detection = result["Detection"]
            detection_key = f"{detection['bbox'][0]}_{detection['bbox'][1]}_{detection['bbox'][2]}_{detection['bbox'][3]}_{detection['class']}"
            detection_status[detection_key] = {
                "estado": result["Estado"],
                "recomendacion": result["Recomendación"],
                "charola": result["Charola"],
                "producto": result["Clase"]
            }
            
            # No es necesario marcar como procesado ya que no estamos buscando productos OK
    
    # Colores por estado (RGBA con transparencia)
    status_colors = {
        "Falta": (220, 53, 69, 180),  # Rojo semitransparente
        "Sobra": (255, 193, 7, 180),  # Amarillo semitransparente
        "Mover": (23, 162, 184, 180)  # Azul semitransparente
    }
    
    # Colores de borde (opacos)
    border_colors = {
        "Falta": (220, 53, 69, 255),  # Rojo
        "Sobra": (255, 193, 7, 255),  # Amarillo
        "Mover": (23, 162, 184, 255)  # Azul
    }
    
    # Etiquetas para la leyenda
    status_labels = {
        "Falta": "Faltante",
        "Sobra": "Sobrante",
        "Mover": "Mover"
    }
    
    # No buscamos productos OK
    
    # Dibujar cada detección con el color correspondiente
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        detection_key = f"{x1}_{y1}_{x2}_{y2}_{detection['class']}"
        
        if detection_key in detection_status:
            estado = detection_status[detection_key]["estado"]
            recomendacion = detection_status[detection_key]["recomendacion"]
            charola = detection_status[detection_key]["charola"]
            producto = detection_status[detection_key]["producto"]
            
            # Dibujar rectángulo semitransparente en el overlay
            overlay_draw.rectangle([x1, y1, x2, y2], 
                                fill=status_colors[estado],
                                outline=border_colors[estado], 
                                width=4)
            
            # Texto para mostrar
            display_text = f"{producto}"
            
            # Calcular posición y tamaño del texto
            text_bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # Dibujar fondo para el texto principal
            overlay_draw.rectangle(
                [x1, y1 - text_height - 10, x1 + text_width + 10, y1],
                fill=status_colors[estado],
                outline=border_colors[estado],
                width=2
            )
            
            # Dibujar texto con sombra para mejor contraste
            overlay_draw.text((x1 + 5 + 1, y1 - text_height - 5 + 1), 
                          display_text, 
                          fill=(0, 0, 0, 255), 
                          font=font)  # sombra
            overlay_draw.text((x1 + 5, y1 - text_height - 5), 
                          display_text, 
                          fill=(255, 255, 255, 255), 
                          font=font)  # texto principal
            
            # Añadir texto de charola y recomendación en la parte inferior
            info_text = f"Ch: {int(charola)}"
            
            # Calcular tamaño del texto de info
            info_bbox = draw.textbbox((0, 0), info_text, font=small_font)
            info_width = info_bbox[2] - info_bbox[0]
            info_height = info_bbox[3] - info_bbox[1]
            
            # Dibujar fondo para el texto de info
            overlay_draw.rectangle(
                [x1, y2, x1 + info_width + 10, y2 + info_height + 6],
                fill=status_colors[estado],
                outline=border_colors[estado],
                width=1
            )
            
            # Dibujar texto de info
            overlay_draw.text((x1 + 5, y2 + 3), 
                          info_text, 
                          fill=(255, 255, 255, 255), 
                          font=small_font)
        else:
            # Si no se encontró en los resultados de comparación, usar color gris
            overlay_draw.rectangle([x1, y1, x2, y2], 
                               outline=(200, 200, 200, 255), 
                               width=3)
    
    # Añadir leyenda en la parte superior derecha
    legend_x = img_width - 180
    legend_y = 20
    legend_height = 35 * len(status_colors) + 10
    
    # Fondo para la leyenda
    overlay_draw.rectangle(
        [legend_x - 10, legend_y - 10, legend_x + 170, legend_y + legend_height],
        fill=(0, 0, 0, 150),  # Negro semitransparente
        outline=(255, 255, 255, 200),
        width=2
    )
    
    # Título de la leyenda
    overlay_draw.text((legend_x, legend_y), 
                  "LEYENDA", 
                  fill=(255, 255, 255, 255), 
                  font=font)
    
    # Elementos de la leyenda
    for i, (status, color) in enumerate(status_colors.items()):
        y_offset = legend_y + 35 + (i * 30)
        # Rectángulo de color
        overlay_draw.rectangle(
            [legend_x, y_offset, legend_x + 20, y_offset + 20],
            fill=color,
            outline=(255, 255, 255, 200),
            width=1
        )
        # Etiqueta
        overlay_draw.text((legend_x + 30, y_offset), 
                      status_labels[status], 
                      fill=(255, 255, 255, 255), 
                      font=font)
    
    # Superponer el overlay en la imagen original
    output_img = Image.alpha_composite(output_img.convert('RGBA'), overlay)
    
    # Convertir de vuelta a RGB para compatibilidad
    return np.array(output_img.convert('RGB'))

# Función para generar gráfico de KPIs y score
def generate_kpi_gauge(metrics):
    """Generar medidor tipo gauge para visualizar el score total"""
    # Crear figura
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics["score"],
        title={'text': "Score de Cumplimiento"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "#dc3545"},
                {'range': [50, 75], 'color': "#ffc107"},
                {'range': [75, 100], 'color': "#28a745"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Configurar sidebar
st.sidebar.header("🛠️ Configuración y Controles")

# Control de confianza del modelo
confidence_threshold = st.sidebar.slider(
    "Umbral de confianza",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
    help="Ajusta el nivel de confianza mínimo para las detecciones"
)

# Desactivar temporalmente el selecionador de modo
st.session_state.mode = "Simple"

# Configuración sencilla para todos los usuarios
st.sidebar.markdown("### Opciones de Visualización")

# Opciones básicas para cajas y etiquetas
box_thickness = st.sidebar.slider("Grosor de cajas", 1, 10, 3)
use_custom_colors = st.sidebar.checkbox("Usar colores por clase", value=True)

# Se han eliminado las opciones de detección de espacios vacíos

# Info de historial
with st.sidebar.expander("Historial de análisis", expanded=False):
    if len(st.session_state.history) > 0:
        for i, item in enumerate(reversed(st.session_state.history)):
            st.write(f"📊 {item['date']} - {item['metrics']['total_productos']} productos")
            if st.button(f"Cargar #{i+1}", key=f"history_{i}"):
                # Cargar datos del historial
                st.session_state.image = item['image']
                st.session_state.csv_data = item['csv_data']
                st.session_state.processed_results = item['processed_results']
                st.session_state.highlighted_image = item['highlighted_image'] if 'highlighted_image' in item else None
                st.session_state.empty_spaces = item['empty_spaces'] if 'empty_spaces' in item else []
                st.session_state.empty_spaces_image = item['empty_spaces_image'] if 'empty_spaces_image' in item else None 
                st.session_state.comparison_results = item['comparison_results']
                st.session_state.metrics = item['metrics']
                st.experimental_rerun()
    else:
        st.info("No hay análisis previos guardados.")

# Sección de uploads de archivos
col1, col2 = st.columns(2)

with col1:
    
    # Cargar imagen del anaquel
    with st.container():
        st.markdown("Selecciona una imagen del anaquel")
        upload_image = st.file_uploader("Arrastra y suelta o haz clic para cargar", type=["jpg", "jpeg", "png"], key="image_uploader")

with col2:
    with st.container():
        st.markdown("Selecciona el archivo CSV del planograma")
        upload_csv = st.file_uploader("Arrastra y suelta o haz clic para cargar", type=["csv"], key="csv_uploader")

# Procesar archivos subidos
if upload_image is not None:
    # Leer y mostrar imagen
    file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir de BGR a RGB
    st.session_state.image = image
    
    # Mostrar imagen original en una sección colapsable
    with st.expander("Ver imagen original", expanded=False):
        st.image(image, caption="Imagen Original", use_column_width=True)

if upload_csv is not None:
    # Leer CSV usando la biblioteca estándar de Python para evitar problemas con pyarrow
    try:
        # Leer el contenido del CSV usando io y csv estándar de Python
        import csv
        import io
        
        # Leer el contenido del archivo como bytes
        csv_content = upload_csv.read()
        
        # Decodificar usando latin-1
        text_content = csv_content.decode('latin-1')
        
        # Crear un StringIO para leer el contenido como archivo
        csv_file = io.StringIO(text_content)
        
        # Usar el lector de CSV para determinar las columnas
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Primera fila como encabezados
        
        # Reiniciar el puntero
        csv_file.seek(0)
        
        # Leer todas las filas
        rows = []
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            rows.append(row)
        
        # Convertir a DataFrame
        csv_data = pd.DataFrame(rows)
        st.session_state.csv_data = csv_data
        
        # Mostrar datos del CSV en una sección colapsable
        with st.expander("Ver datos del planograma", expanded=False):
            st.dataframe(csv_data)
    except Exception as e:
        st.error(f"Error al leer el archivo CSV: {e}")
        import traceback
        st.error(traceback.format_exc())

# Botón para procesar la imagen
if st.session_state.image is not None and st.session_state.csv_data is not None:
    process_button = st.button(
        "🔍 Analizar Planograma", 
        key="process_button",
        help="Procesa la imagen y compara con el planograma esperado",
        type="primary"
    )

    # Inicializar métricas si no existen
    if "metrics" not in st.session_state or st.session_state.metrics is None:
        st.session_state.metrics = {
            "total_productos": 0,
            "ok_productos": 0,
            "falta_productos": 0,
            "sobra_productos": 0,
            "mover_productos": 0,
            "score": 0
        }
    
    if process_button:
        with st.spinner("Analizando imagen y comparando con el planograma... Por favor espere."):
            # Procesar imagen con el modelo
            detections = process_image(st.session_state.image, confidence=confidence_threshold)
            st.session_state.detections = detections
            
            # Dibujar detecciones en la imagen con opciones básicas
            annotated_image = draw_detections(
                st.session_state.image, 
                detections, 
                box_thickness=box_thickness, 
                use_custom_colors=use_custom_colors
            )
                
            st.session_state.processed_results = annotated_image
            
            # Comparar con el planograma esperado usando algoritmo avanzado
            comparison_results = analyze_planogram_advanced(detections, st.session_state.csv_data)
            st.session_state.comparison_results = comparison_results
            
            # Generar imagen con áreas resaltadas (correctas e incorrectas)
            highlighted_image = generate_highlighted_image(st.session_state.image.copy(), detections, comparison_results)
            st.session_state.highlighted_image = highlighted_image
            
            # Generar visualización de espacios faltantes (donde faltan productos)
            missing_spaces_image = visualize_missing_products(st.session_state.image.copy(), comparison_results)
            st.session_state.missing_spaces_image = missing_spaces_image
            
            # Guardar en historial (para todos los modos)
            history_entry = {
                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'image': st.session_state.image.copy(),
                'csv_data': st.session_state.csv_data.copy(),
                'processed_results': st.session_state.processed_results.copy(),
                'highlighted_image': st.session_state.highlighted_image.copy(),
                'empty_spaces': st.session_state.empty_spaces if hasattr(st.session_state, 'empty_spaces') else [],
                'empty_spaces_image': st.session_state.empty_spaces_image.copy() if hasattr(st.session_state, 'empty_spaces_image') and st.session_state.empty_spaces_image is not None else None,
                'comparison_results': st.session_state.comparison_results.copy(),
                'metrics': st.session_state.metrics.copy()
            }
            st.session_state.history.append(history_entry)
            
            # Limitar historial a 5 entradas
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[-5:]

# Mostrar resultados según el modo seleccionado
if st.session_state.processed_results is not None:
    # Sección principal de resultados
    st.header("2. Resultados del Análisis")
    
    # Tabs para diferentes visualizaciones
    visual_tabs = st.tabs(["🏷️ Detecciones Básicas", "✅❌ Áreas Correctas e Incorrectas", "🔍 Espacios Faltantes"])
    
    with visual_tabs[0]:
        st.subheader("Detecciones en el anaquel")
        st.image(st.session_state.processed_results, use_column_width=True)
        
    with visual_tabs[1]:
        st.subheader("Visualización de Áreas Correctas e Incorrectas")
        st.image(st.session_state.highlighted_image, use_column_width=True)
        
        # Leyenda explicativa
        st.markdown("""
        <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 10px;">
          <h4 style="text-align: center;">Guía de Colores</h4>
          <div style="display: flex; justify-content: center; flex-wrap: wrap;">
            <div style="margin: 5px; padding: 5px 10px; background-color: rgba(220, 53, 69, 0.7); color: white; border-radius: 3px;">🔴 Rojo = Faltante</div>
            <div style="margin: 5px; padding: 5px 10px; background-color: rgba(255, 193, 7, 0.7); color: white; border-radius: 3px;">🟡 Amarillo = Sobrante</div>
            <div style="margin: 5px; padding: 5px 10px; background-color: rgba(23, 162, 184, 0.7); color: white; border-radius: 3px;">🔵 Azul = Mover</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    
    with visual_tabs[2]:
        st.subheader("Visualización de Espacios Faltantes en Estanterías")
        st.image(st.session_state.missing_spaces_image, use_column_width=True)
        
        # Explicación de la visualización
        st.markdown("""
        <div style="background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-top: 10px;">
          <h4 style="text-align: center;">Espacios donde faltan artículos</h4>
          <p>Esta visualización muestra los espacios en las estanterías donde deberían colocarse los productos que faltan según el planograma. Cada rectángulo rojo indica un espacio vacío donde debería haber un producto específico.</p>
          <p>Puedes usar esta información para identificar rápidamente dónde colocar los productos faltantes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar tabla de productos faltantes
        missing_products = [item for item in st.session_state.comparison_results if item["Estado"] == "Falta"]
        if missing_products:
            st.subheader("Lista de productos faltantes")
            missing_df = pd.DataFrame([
                {
                    "Charola": item["Charola"],
                    "Producto": item["Clase"],
                    "Posición": item.get("Recomendación", "").replace("Agregar en posición ", "Posición ")
                } for item in missing_products
            ])
            st.dataframe(missing_df, use_container_width=True)
        
    # La pestaña de espacios vacíos ha sido eliminada
    
    # Mostrar métricas básicas en forma de tarjetas
    st.subheader("Métricas de Productos")
    
    # Mostrar métricas en 4 columnas
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Productos", st.session_state.metrics['total_productos'])
    with metric_cols[1]:
        st.metric("Faltantes", st.session_state.metrics['falta_productos'])
    with metric_cols[2]:
        st.metric("Sobrantes", st.session_state.metrics['sobra_productos'])
    with metric_cols[3]:
        st.metric("Por Mover", st.session_state.metrics['mover_productos'])
    
    # Omitimos el resumen por charola con productos correctos

    # Tabla detallada de movimientos
    st.subheader("Tabla Detallada de Movimientos")
    
    if len(st.session_state.comparison_results) > 0:
        # Crear DataFrame para mostrar resultados
        comparison_df = pd.DataFrame([
            {
                "Charola": item["Charola"],
                "Producto": item["Clase"],
                "Estado": item["Estado"],
                "Recomendación": item["Recomendación"]
            } for item in st.session_state.comparison_results
        ])
        
        # Crear un filtro para la charola
        all_levels = sorted(comparison_df['Charola'].unique())
        selected_level = st.multiselect(
            "Filtrar por charola:",
            options=all_levels,
            default=all_levels
        )
        
        # Aplicar filtro
        if selected_level:
            filtered_df = comparison_df[comparison_df['Charola'].isin(selected_level)]
        else:
            filtered_df = comparison_df
        
        # Mostrar dataframe estilizado
        st.dataframe(filtered_df.style.applymap(highlight_status, subset=['Estado']), use_container_width=True)
        
        # Botón para descargar resultados como CSV
        csv = filtered_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="resultados_planograma.csv" class="btn btn-primary">📥 Descargar resultados como CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Presentación de resultados de forma clara y organizada
        st.subheader("Resultados del Planograma por Charola")
        
        # Agrupar resultados por charola
        filtered_df['id'] = range(len(filtered_df))
        
        # Agrupar por charola para facilitar navegación
        charolas = filtered_df['Charola'].unique()
        
        # Crear un acordeon para cada charola 
        for charola in sorted(charolas):
            charola_items = filtered_df[filtered_df['Charola'] == charola]
            
            # Obtener un resumen de estado para esta charola
            total_items = len(charola_items)
            falta = sum(charola_items['Estado'] == 'Falta')
            sobra = sum(charola_items['Estado'] == 'Sobra')
            mover = sum(charola_items['Estado'] == 'Mover')
            
            # Asignar emoji según el estado
            estado_emoji = "⚠️"
            resumen = f"{estado_emoji} Charola {int(charola)}: {falta} Falta, {sobra} Sobra, {mover} A mover"
            
            # Mostrar cada charola en un expandible
            with st.expander(resumen, expanded=True):
                # Crear tabla con estilo condicional
                st.dataframe(charola_items.style.applymap(highlight_status, subset=['Estado']), use_container_width=True)
                
                # Añadir separador entre charolas
                st.markdown("---")

# Mostrar información del código
with st.expander("Información técnica", expanded=False):
    st.markdown("### Información del modelo y la aplicación")
    st.markdown("""
    - **Nombre**: Shell-E
    - **Modelo de detección**: Roboflow Object Detection v8
    - **API**: Roboflow Hosted API
    - **Algoritmo de comparación**: Líneas horizontales y posición relativa
    - **Versión de la aplicación**: Premium 1.0
    """)
    
    st.code("""
    from roboflow import Roboflow
    
    rf = Roboflow(api_key="KkzFysqU16FP7cfNZARz")
    project = rf.workspace("hackathon-cud1g").project("custom-workflow-object-detection-oidmg")
    model = project.version(8).model  # Versión actual del modelo
    """, language="python")

# Pie de página
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p><br> Shell-E | Desarrollado para Hackathon Femsa 2025</p>
</div>
""", unsafe_allow_html=True)
