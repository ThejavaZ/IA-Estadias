import cv2
import numpy as np
import time

def cargar_modelo():
    """Carga el modelo YOLO pre-entrenado para la detección de objetos."""
    # Cargar la red YOLO
    net = cv2.dnn.readNet("/home/javiersg/Proyectos/face_detection/yolov3.weights",
                      "/home/javiersg/Proyectos/face_detection/yolov3.cfg")

    
    # Obtener las capas de salida
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    
    # Cargar las clases
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    return net, output_layers, classes

def detectar_objetos(img, net, output_layers, classes, objetos_interes=None):
    """Detecta objetos en una imagen usando el modelo YOLO."""
    if objetos_interes is None:
        objetos_interes = ["car", "dog", "bird", "person",]
    
    height, width, channels = img.shape
    
    # Preparar la imagen para la red neuronal
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    
    # Realizar la detección
    outs = net.forward(output_layers)
    
    # Información sobre las detecciones
    class_ids = []
    confidences = []
    boxes = []
    
    # Procesar las detecciones
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5 and classes[class_id] in objetos_interes:
                # Coordenadas del objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Coordenadas del rectángulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Aplicar supresión no máxima para eliminar detecciones redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Dibujar las cajas y etiquetas
    font = cv2.FONT_HERSHEY_SIMPLEX
    colores = np.random.uniform(0, 255, size=(len(classes), 3))
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            color = colores[class_ids[i]]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), font, 0.5, color, 2)
    
    return img, class_ids, boxes, confidences

def procesar_video(video_path=0, objetos_interes=None):
    """Procesa un video o webcam para detectar objetos."""
    if objetos_interes is None:
        objetos_interes = ["car", "dog", "bird", "person"]
    
    # Mapeo de clases en español
    mapeo_espanol = {
        "car": "Carro",
        "dog": "Perro",
        "bird": "Paloma",
        "person": "Persona"
    }
    
    # Cargar el modelo
    print("Cargando modelo YOLO...")
    net, output_layers, classes = cargar_modelo()
    print("Modelo cargado correctamente.")
    
    # Iniciar captura de video
    cap = cv2.VideoCapture(video_path)
    
    # Verificar si la cámara se abrió correctamente
    if not cap.isOpened():
        print("Error al abrir la cámara o el archivo de video.")
        return
    
    # Contador de objetos
    contadores = {obj: 0 for obj in objetos_interes}
    ultimo_conteo = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo recibir el frame. Saliendo...")
            break
        
        # Detectar objetos
        frame, class_ids, boxes, confidences = detectar_objetos(frame, net, output_layers, classes, objetos_interes)
        
        # Actualizar contadores cada segundo
        if time.time() - ultimo_conteo >= 1:
            contadores = {obj: 0 for obj in objetos_interes}
            for i in range(len(class_ids)):
                clase = classes[class_ids[i]]
                if clase in objetos_interes:
                    contadores[clase] += 1
            ultimo_conteo = time.time()
        
        # Mostrar contadores en la pantalla
        y_pos = 30
        for obj, count in contadores.items():
            nombre_esp = mapeo_espanol.get(obj, obj)
            cv2.putText(frame, f"{nombre_esp}: {count}", (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
        # Mostrar el resultado
        cv2.imshow("Detector de Objetos", frame)
        
        # Salir con la tecla 'q'
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def procesar_imagen(imagen_path, objetos_interes=None):
    """Procesa una imagen para detectar objetos."""
    if objetos_interes is None:
        objetos_interes = ["car", "dog", "bird", "person"]
    
    # Mapeo de clases en español
    mapeo_espanol = {
        "car": "Carro",
        "dog": "Perro",
        "bird": "Pájaro/Paloma",
        "person": "Persona"
    }
    
    # Cargar el modelo
    print("Cargando modelo YOLO...")
    net, output_layers, classes = cargar_modelo()
    print("Modelo cargado correctamente.")
    
    # Cargar la imagen
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error al cargar la imagen: {imagen_path}")
        return
    
    # Detectar objetos
    img, class_ids, boxes, confidences = detectar_objetos(img, net, output_layers, classes, objetos_interes)
    
    # Contar objetos detectados
    contadores = {obj: 0 for obj in objetos_interes}
    for i in range(len(class_ids)):
        clase = classes[class_ids[i]]
        if clase in objetos_interes:
            contadores[clase] += 1
    
    # Mostrar contadores en la pantalla
    y_pos = 30
    for obj, count in contadores.items():
        nombre_esp = mapeo_espanol.get(obj, obj)
        cv2.putText(img, f"{nombre_esp}: {count}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
    
    # Mostrar la imagen
    cv2.imshow("Detector de Objetos", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Lista de objetos que queremos detectar
    objetos_interes = ["car", "dog", "bird", "person"]

    # Menú simple
    print("Aplicación de Reconocimiento de Objetos")
    print("1. Usar webcam")
    print("2. Procesar video")
    print("3. Procesar imagen")

    
    opcion = input("Seleccione una opción (1-3): ")
    
    match opcion:
        case "1":
            print("Iniciando webcam... Presione 'q' para salir.")
            procesar_video(0, objetos_interes)
        case "2":
            ruta_video = input("Ingrese la ruta del archivo de video: ")
            print("Procesando video... Presione 'q' para salir.")
            procesar_video(ruta_video, objetos_interes)
        case "3":
            ruta_imagen = input("Ingrese la ruta de la imagen: ")
            print("Procesando imagen...")
            procesar_imagen(ruta_imagen, objetos_interes)
        case _:
            print("Opción no válida.")