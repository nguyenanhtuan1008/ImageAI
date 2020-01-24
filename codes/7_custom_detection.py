from tmodules.Detection.Custom import CustomObjectDetection

def detect_single_image():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\models_5epocks\detection_model-ex-005--loss-0003.791.h5")
    detector.setJsonPath(r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\json\detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\test\images\20191202IMG_0071.JPG", 
                                                output_image_path=r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\ar.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_single_image_hakkai():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\hakkai\hakkai_config_yolov3\yolov3.h5")
    detector.setJsonPath(r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\hakkai\hakkai_config_yolov3\hk_detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=r"E:\acuity\acuity_projects\hakkai_project\hakkai_datasets\hakkai_test_data\hakkai_test_imgs\Image_C0_0359155.jpg", 
                                                output_image_path=r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\hk4.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

if __name__ == '__main__':
    # detect_single_image()
    detect_single_image_hakkai()