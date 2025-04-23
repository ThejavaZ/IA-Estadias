import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from descargar_modelos import descargar_modelos_yolo

class AplicacionReconocimiento:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Objetos")
        self.root.geometry("1000x700")
        
        # Variables
        self.cap = None
        self.ejecutando = False
        self.modelo_cargado = False
        self.net = None
        self.output_layers = None
        self.classes = None
        
        # Mapeo de clases en español
        self.mapeo_espanol = {
            "car": "Carro",
            "dog": "Perro",
            "bird": "Pájaro/Paloma",
            "person": "Persona"
        }
        
        # Objetos de interés
        self.objetos_interes = ["car", "dog", "bird", "person"]
        
        # Verificar y descargar modelos
        self.verificar_modelos()
        
        # Crear interfaz
        self.crear_interfaz()
    
    def verificar_modelos(self):
        """Verifica si los modelos YOLO están disponibles y los descarga si es necesario."""
        archivos_necesarios = ["yolov3.weights", "yolov3.cfg", "coco.names"]
        falta_alguno = False
        
        for archivo in archivos_necesarios:
            if not os.path.exists(archivo):
                falta_alguno = True
                break
        
        if falta_alguno:
            respuesta = messagebox.askyesno(
                "Descargar modelos", 
                "Es necesario descargar los modelos YOLO para el reconocimiento de objetos. ¿Desea descargarlos ahora?"
            )
            if respuesta:
                descargar_modelos_yolo()
            else:
                messagebox.showwarning(
                    "Advertencia", 
                    "La aplicación no funcionará correctamente sin los modelos necesarios."
                )
    
    def crear_interfaz(self):
        """Crea la interfaz gráfica de la aplicación."""
        # Frame principal
        frame_principal = tk.Frame(self.root)
        frame_principal.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame izquierdo para controles
        frame_controles = tk.Frame(frame_principal, width=200)
        frame_controles.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Frame derecho para visualización
        frame_video = tk.Frame(frame_principal)
        frame_video.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas para mostrar el video/imagen
        self.canvas = tk.Canvas(frame_video, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Etiqueta de título
        tk.Label(
            frame_controles, 
            text="Reconocimiento de Objetos", 
            font=("Arial", 14, "bold")
        ).pack(pady=10)
        
        # Botones de acción
        tk.Button(
            frame_controles, 
            text="Cargar Modelo", 
            command=self.cargar_modelo,
            width=20
        ).pack(pady=5)
        
        tk.Button(
            frame_controles, 
            text="Iniciar Webcam", 
            command=self.iniciar_webcam,
            width=20
        ).pack(pady=5)
        
        tk.Button(
            frame_controles, 
            text="Cargar Video", 
            command=self.cargar_video,
            width=20
        ).pack(pady=5)
        
        tk.Button(
            frame_controles, 
            text="Cargar Imagen", 
            command=self.cargar_imagen,
            width=20
        ).pack(pady=5)
        
        tk.Button(
            frame_controles, 
            text="Detener", 
            command=self.detener,
            width=20
        ).pack(pady=5)
        
        # Frame para mostrar contadores
        self.frame_contadores = tk.Frame(frame_controles)
        self.frame_contadores.pack(pady=20, fill=tk.X)
        
        tk.Label(
            self.frame_contadores, 
            text="Objetos Detectados:", 
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        # Etiquetas para contadores
        self.etiquetas_contadores = {}
        for obj in self.objetos_interes:
            nombre_esp = self.mapeo_espanol.get(obj, obj)
            self.etiquetas_contadores[obj] = tk.Label(
                self.frame_contadores, 
                text=f"{nombre_esp}: 0",
                font=("Arial", 10)
            )
            self.etiquetas_contadores[obj].pack(anchor=tk.W, pady=2)
        
        # Etiqueta de estado
        self.etiqueta_estado = tk.Label(
            frame_controles, 
            text="Estado: Listo", 
            font=("Arial", 10),
            fg="blue"
        )
        self.etiqueta_estado.pack(side=tk.BOTTOM, pady=10)
    
    def cargar_modelo(self):
        """Carga el modelo YOLO."""
        try:
            self.actualizar_estado("Cargando modelo YOLO...")
            
            # Cargar la red YOLO
            self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
            
            # Obtener las capas de salida
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            
            # Cargar las clases
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            self.modelo_cargado = True
            self.actualizar_estado("Modelo cargado correctamente")
            messagebox.showinfo("Éxito", "Modelo YOLO cargado correctamente")
        except Exception as e:
            self.actualizar_estado(f"Error al cargar el modelo: {e}")
            messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
    
    def iniciar_webcam(self):
        """Inicia la captura desde la webcam."""
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "Primero debe cargar el modelo")
            return
        
        self.detener()  # Detener cualquier proceso en ejecución
        
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("No se pudo acceder a la webcam")
            
            self.ejecutando = True
            self.actualizar_estado("Webcam iniciada")
            
            # Iniciar procesamiento en un hilo separado
            threading.Thread(target=self.procesar_video, daemon=True).start()
        except Exception as e:
            self.actualizar_estado(f"Error al iniciar webcam: {e}")
            messagebox.showerror("Error", f"No se pudo iniciar la webcam: {e}")
    
    def cargar_video(self):
        """Carga un archivo de video para procesarlo."""
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "Primero debe cargar el modelo")
            return
        
        self.detener()  # Detener cualquier proceso en ejecución
        
        ruta_video = filedialog.askopenfilename(
            title="Seleccionar archivo de video",
            filetypes=(("Archivos de video", "*.mp4 *.avi *.mov"), ("Todos los archivos", "*.*"))
        )
        
        if not ruta_video:
            return
        
        try:
            self.cap = cv2.VideoCapture(ruta_video)
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir el archivo de video")
            
            self.ejecutando = True
            self.actualizar_estado(f"Procesando video: {os.path.basename(ruta_video)}")
            
            # Iniciar procesamiento en un hilo separado
            threading.Thread(target=self.procesar_video, daemon=True).start()
        except Exception as e:
            self.actualizar_estado(f"Error al cargar video: {e}")
            messagebox.showerror("Error", f"No se pudo cargar el video: {e}")
    
    def cargar_imagen(self):
        """Carga una imagen para procesarla."""
        if not self.modelo_cargado:
            messagebox.showwarning("Advertencia", "Primero debe cargar el modelo")
            return
        
        self.detener()  # Detener cualquier proceso en ejecución
        
        ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=(("Archivos de imagen", "*.jpg *.jpeg *.png"), ("Todos los archivos", "*.*"))
        )
        
        if not ruta_imagen:
            return
        
        try:
            img = cv2.imread(ruta_imagen)
            if img is None:
                raise Exception("No se pudo abrir la imagen")
            
            self.actualizar_estado(f"Procesando imagen: {os.path.basename(ruta_imagen)}")
            
            # Procesar la imagen
            img_procesada, contadores = self.detectar_objetos(img)
            
            # Actualizar contadores
            self.actualizar_contadores(contadores)
            
            # Mostrar la imagen en el canvas
            self.mostrar_frame(img_procesada)
            
            self.actualizar_estado("Imagen procesada")
        except Exception as e:
            self.actualizar_estado(f"Error al procesar imagen: {e}")
            messagebox.showerror("Error", f"No se pudo procesar la imagen: {e}")
    
    def procesar_video(self):
        """Procesa el video frame por frame."""
        contadores = {obj: 0 for obj in self.objetos_interes}
        ultimo_conteo = time.time()
        
        while self.ejecutando and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                self.actualizar_estado("Fin del video o error en la captura")
                self.detener()
                break
            
            # Detectar objetos
            frame_procesado, nuevos_contadores = self.detectar_objetos(frame)
            
            # Actualizar contadores cada segundo
            if time.time() - ultimo_conteo >= 1:
                contadores = nuevos_contadores
                self.actualizar_contadores(contadores)
                ultimo_conteo = time.time()
            
            # Mostrar el frame procesado
            self.mostrar_frame(frame_procesado)
            
            # Pequeña pausa para no saturar la CPU
            time.sleep(0.01)
    
    def detectar_objetos(self, img):
        """Detecta objetos en una imagen usando el modelo YOLO."""
        height, width, _ = img.shape
        
        # Preparar la imagen para la red neuronal
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        
        # Realizar la detección
        outs = self.net.forward(self.output_layers)
        
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
                
                if confidence > 0.5 and self.classes[class_id] in self.objetos_interes:
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
        colores = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                clase = self.classes[class_ids[i]]
                nombre_esp = self.mapeo_espanol.get(clase, clase)
                label = f"{nombre_esp}: {confidences[i]:.2f}"
                color = colores[class_ids[i]]
                
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), font, 0.5, color, 2)
        
        # Contar objetos detectados
        contadores = {obj: 0 for obj in self.objetos_interes}
        for i in range(len(class_ids)):
            if i in indexes:
                clase = self.classes[class_ids[i]]
                if clase in self.objetos_interes:
                    contadores[clase] += 1
        
        return img, contadores
    
    def mostrar_frame(self, frame):
        """Muestra un frame en el canvas."""
        # Convertir de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para ajustar al canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Asegurarse de que el canvas tenga tamaño
            # Calcular proporciones
            frame_height, frame_width = frame.shape[:2]
            ratio = min(canvas_width/frame_width, canvas_height/frame_height)
            new_width = int(frame_width * ratio)
            new_height = int(frame_height * ratio)
            
            # Redimensionar
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
            
            # Convertir a formato PIL
            img_pil = Image.fromarray(frame_resized)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            
            # Actualizar canvas
            self.canvas.config(width=new_width, height=new_height)
            self.canvas.create_image(new_width//2, new_height//2, image=img_tk)
            self.canvas.image = img_tk  # Mantener referencia
    
    def actualizar_contadores(self, contadores):
        """Actualiza las etiquetas de contadores en la interfaz."""
        for obj, count in contadores.items():
            if obj in self.etiquetas_contadores:
                nombre_esp = self.mapeo_espanol.get(obj, obj)
                self.etiquetas_contadores[obj].config(text=f"{nombre_esp}: {count}")
    
    def actualizar_estado(self, mensaje):
        """Actualiza la etiqueta de estado."""
        self.etiqueta_estado.config(text=f"Estado: {mensaje}")
    
    def detener(self):
        """Detiene la captura y procesamiento de video."""
        self.ejecutando = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.actualizar_estado("Detenido")

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionReconocimiento(root)
    root.mainloop()