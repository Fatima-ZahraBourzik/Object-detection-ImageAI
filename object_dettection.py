
#premier code à executer
# from imageai.Detection.Custom import DetectionModelTrainer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# trainer = DetectionModelTrainer()
# trainer.setModelTypeAsYOLOv3()
# trainer.setDataDirectory(data_directory="panneu")
# trainer.setTrainConfig(object_names_array=["panneu"], batch_size=4, num_experiments=100, train_from_pretrained_model="pretrained-yolov3.h5")
# trainer.trainModel()

#deuxième code à executer
# from imageai.Detection.Custom import DetectionModelTrainer
# trainer = DetectionModelTrainer()
# trainer.setModelTypeAsYOLOv3()
# trainer.setDataDirectory(data_directory="panneu")
# trainer.evaluateModel(model_path="panneu\models", json_path="panneu/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)


#troisième code à executé
from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("panneu\models\detection_model-ex-004--loss-0037.174.h5") 
detector.setJsonPath("panneu\json\detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="2.jpg", output_image_path="2_detected.jpg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
