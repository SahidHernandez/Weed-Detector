import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Detección con TensorFlow y YOLO")
        self.root.geometry("1200x800")
        
        # ====== CONFIGURACIÓN ======
        self.MODEL_TF_1_PATH = 'C:\\Users\\sahid\\OneDrive\\Escritorio\\Maestria\\Cuatri2\\ProcesamientoImagenes\\U1\\proyecto_final\\weed_detector_40\\saved_model'
        self.MODEL_TF_2_PATH = 'C:\\Users\\sahid\\OneDrive\\Escritorio\\Maestria\\Cuatri2\\ProcesamientoImagenes\\U1\\proyecto_final\\weed_detector_backup\\saved_model'
        self.YOLO_MODEL_PATH = 'best.pt'
        self.LABEL_MAP = {1: 'weed'}
        self.camera_active = False
        self.cap = None
        self.detect_in_camera = False
        self.current_model = "TensorFlow Modelo 1"
        self.detection_result = tk.StringVar()  # Variable para almacenar el resultado
        
        # ====== INTERFAZ ======
        self.create_widgets()
        
        # ====== CARGAR MODELOS ======
        self.load_models()
        
        # Configurar el cierre seguro de la aplicación
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel de resultado de detección
        result_frame = ttk.Frame(main_frame)
        result_frame.pack(fill=tk.X, pady=5)
        
        # Etiqueta para mostrar el resultado
        self.result_label = ttk.Label(
            result_frame, 
            textvariable=self.detection_result,
            font=('Arial', 14),
            foreground='black'
        )
        self.result_label.pack(fill=tk.X, pady=10)
        
        # Panel de control superior
        control_frame = ttk.LabelFrame(main_frame, text="Controles")
        control_frame.pack(fill=tk.X, pady=5)
        
        # Botón para cargar imagen
        self.load_btn = ttk.Button(control_frame, text="Cargar Imagen", command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botón para abrir cámara
        self.camera_btn = ttk.Button(control_frame, text="Abrir Cámara", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Checkbutton para detección en tiempo real
        self.detect_var = tk.BooleanVar()
        self.detect_check = ttk.Checkbutton(
            control_frame, 
            text="Detección en tiempo real", 
            variable=self.detect_var,
            command=self.toggle_detection,
            state='disabled'
        )
        self.detect_check.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Panel de configuración del modelo
        model_frame = ttk.LabelFrame(control_frame, text="Configuración del Modelo")
        model_frame.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Selección de modelo
        self.model_var = tk.StringVar(value="TensorFlow Modelo 1")
        model_options = ["TensorFlow Modelo 1", "TensorFlow Modelo 2", "YOLO"]
        self.model_menu = ttk.OptionMenu(
            model_frame, 
            self.model_var, 
            self.current_model, 
            *model_options,
            command=self.model_changed
        )
        self.model_menu.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botón de detección
        self.detect_btn = ttk.Button(
            control_frame, 
            text="Realizar Detección", 
            command=self.run_detection,
            state='disabled'
        )
        self.detect_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Panel de visualización
        self.image_frame = ttk.LabelFrame(main_frame, text="Imagen")
        self.image_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Canvas para mostrar imágenes
        self.canvas = tk.Canvas(self.image_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
        
        # Mensaje inicial en el canvas
        self.show_initial_message()
    
    def show_initial_message(self):
        """Muestra un mensaje inicial en el canvas vacío"""
        self.canvas.delete("all")
        self.canvas.create_text(
            self.canvas.winfo_width()/2, 
            self.canvas.winfo_height()/2,
            text="Cargue una imagen o active la cámara para comenzar",
            font=('Arial', 14),
            fill='gray'
        )
    
    def model_changed(self, *args):
        """Actualiza el modelo seleccionado"""
        self.current_model = self.model_var.get()
        self.status_var.set(f"Modelo cambiado a: {self.current_model}")
        
        if self.camera_active and self.detect_in_camera:
            self.status_var.set(f"Modelo cambiado a: {self.current_model} - Aplicando en tiempo real...")
    
    def toggle_detection(self):
        """Activa/desactiva la detección en tiempo real"""
        self.detect_in_camera = self.detect_var.get()
        if self.detect_in_camera:
            self.status_var.set("Detección en tiempo real activada")
        else:
            self.status_var.set("Detección en tiempo real desactivada - Mostrando vista previa")
    
    def toggle_camera(self):
        """Activa/desactiva la cámara web"""
        if not self.camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo abrir la cámara. Asegúrese de que esté conectada.")
                return
            
            self.camera_active = True
            self.camera_btn.config(text="Cerrar Cámara")
            self.detect_check.config(state='normal')
            self.detect_btn.config(state='disabled')
            self.load_btn.config(state='disabled')
            self.status_var.set("Cámara activada - Mostrando vista previa")
            self.update_camera()
        else:
            self.camera_active = False
            self.detect_var.set(False)
            self.detect_in_camera = False
            self.detect_check.config(state='disabled')
            
            if self.cap:
                self.cap.release()
            
            self.camera_btn.config(text="Abrir Cámara")
            self.load_btn.config(state='normal')
            
            if hasattr(self, 'original_image'):
                self.detect_btn.config(state='normal')
            
            self.status_var.set("Cámara desactivada")
            self.show_initial_message()
    
    def update_camera(self):
        """Actualiza el frame de la cámara en la interfaz"""
        if self.camera_active:
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if self.detect_in_camera:
                    # Realizar detección y actualizar el mensaje de resultado
                    detected, _ = self.detect_in_frame(frame_rgb)
                    if detected:
                        self.detection_result.set("Resultado: Maleza detectada")
                        self.result_label.config(foreground='red')
                    else:
                        self.detection_result.set("Resultado: No se detectó maleza")
                        self.result_label.config(foreground='green')
                    
                    self.status_var.set(f"Cámara activa - Detectando con {self.current_model}")
                else:
                    self.original_image = Image.fromarray(frame_rgb)
                    self.status_var.set("Cámara activa - Vista previa")
                
                self.display_image(self.original_image)
            
            self.root.after(10, self.update_camera)
    
    def load_models(self):
        """Carga los modelos de TensorFlow y YOLO"""
        self.status_var.set("Cargando modelos...")
        self.root.update()
        
        try:
            self.model_tf1 = tf.saved_model.load(self.MODEL_TF_1_PATH)
            self.model_tf2 = tf.saved_model.load(self.MODEL_TF_2_PATH)
            self.model_yolo = YOLO(self.YOLO_MODEL_PATH)
            self.status_var.set("Modelos cargados correctamente")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar los modelos: {str(e)}")
            self.status_var.set("Error al cargar modelos")
    
    def load_image(self):
        """Carga una imagen desde el sistema de archivos"""
        if self.camera_active:
            self.toggle_camera()
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            try:
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image)
                self.detect_btn.config(state='normal')
                self.status_var.set(f"Imagen cargada: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar la imagen: {str(e)}")
                self.status_var.set("Error al cargar imagen")
                self.show_initial_message()
    
    def display_image(self, image):
        """Muestra una imagen en el canvas"""
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        img_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_image)
        
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width/2, canvas_height/2,
            anchor=tk.CENTER,
            image=self.tk_image
        )
    
    def detect_in_frame(self, frame):
        """Realiza detección en un frame de la cámara o imagen"""
        try:
            image_np = np.array(frame)
            detected = False
            
            if self.current_model.startswith("TensorFlow"):
                modelo_tf = self.model_tf1 if "1" in self.current_model else self.model_tf2
                input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
                detections = modelo_tf(input_tensor)

                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
                scores = detections['detection_scores']

                # Verificar si hay detecciones con confianza > 0.75
                for i in range(num_detections):
                    if scores[i] >= 0.75:
                        detected = True
                        break
                
            else:  # YOLO
                results = self.model_yolo(image_np)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        if box.conf.item() > 0.6:
                            detected = True
                            break
            
            return detected, image_np
            
        except Exception as e:
            self.status_var.set(f"Error en detección: {str(e)}")
            return False, frame
    
    def run_detection(self):
        """Ejecuta el modelo seleccionado en la imagen cargada"""
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("Advertencia", "Primero carga una imagen")
            return
        
        self.status_var.set(f"Procesando imagen con {self.current_model}...")
        self.root.update()
        
        try:
            image_np = np.array(self.original_image)
            detected, _ = self.detect_in_frame(image_np)
            
            if detected:
                self.detection_result.set("Resultado: Maleza detectada")
                self.result_label.config(foreground='red')
            else:
                self.detection_result.set("Resultado: No se detectó maleza")
                self.result_label.config(foreground='green')
            
            self.status_var.set(f"Detección completada con {self.current_model}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la detección: {str(e)}")
            self.status_var.set("Error en detección")
    
    def on_close(self):
        """Maneja el cierre seguro de la aplicación"""
        if self.camera_active and self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()