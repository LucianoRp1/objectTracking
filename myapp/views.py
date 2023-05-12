from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
import cv2

import yolov5, torch 
from yolov5.utils.general import ( check_img_size, non_max_suppression, scale_segments, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from PIL import Image as im
# Create your views here.

def index (request):
    return render(request, 'index.html')
# model = yolov5.load('yolov5s.pt')
# Cargamos el modelo YOLOv5 utilizando la librería torch.hub
# y lo asignamos a la variable model. También se selecciona el dispositivo a utilizar.
model= torch.hub.load('ultralytics/yolov5', 'yolov5s')
device = select_device('cpu')

# Inicializamos el modelo DeepSort con su configuración y lo asignamos a la variable deepsort.
cfg = get_config()
cfg.merge_from_file('deep_sort/configs/deep_sort.yaml')
deepsort = DeepSort('osnet_x0_25',
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

# Obtenemos los nombres de las clases a detectar del modelo YOLOv5.
names = model.module.names if hasattr(model, 'module') else model.names

# Definimos una función para obtener el stream de video en tiempo real desde la cámara.
def stream():
    cap = cv2.VideoCapture(0)
    # Establecemos la confianza mínima y el IoU mínimo para filtrar las detecciones.
    model.conf = 0.45
    model.iou = 0.5
    # Definimos las clases que deseamos detectar (0: persona, 64: perro, 39: bicicleta).
    model.classes = [0,64,39]
    while True:
        # Capturamos el frame de la cámara.
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to capture image")
            break

        # Realizamos la detección de objetos utilizando el modelo YOLOv5.
        results = model(frame, augment=True)
        
        # Procesamos las detecciones utilizando el modelo DeepSort.
        annotator = Annotator(frame, line_width=2, pil=not ascii) 
        det = results.pred[0]
        if det is not None and len(det):   
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
            if len(outputs) > 0:
                for j, (output, conf) in enumerate(zip(outputs, confs)):

                    bboxes = output[0:4]
                    id = output[4]
                    cls = output[5]

                    c = int(cls)  # integer class
                    label = f'{id} {names[c]} {conf:.2f}'
                    annotator.box_label(bboxes, label, color=colors(c, True))

        else:
            deepsort.increment_ages()

        # Mostramos el resultado final con las detecciones y la información de DeepSort.
        im0 = annotator.result()    
        image_bytes = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

# Definimos una vista para transmitir el stream de video a través de HTTP.

def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')   