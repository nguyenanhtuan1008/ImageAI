from tmodules.Detection.Custom import CustomObjectDetection

def detect_single_image():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\models_5epocks\detection_model-ex-005--loss-0003.791.h5")
    detector.setJsonPath(r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\json\detection_config.json")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=r"E:\acuity\acuity_projects\AR\datasets\ARmarker_data\_15-21jan\20200115-1IMG_1417.JPG", 
                                                output_image_path=r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\ar1.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def detect_single_image_hakkai():
    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\hakkai\hakkai_config_yolov3\yolov3.h5")
    detector.setJsonPath(r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\hakkai\hakkai_config_yolov3\hk_detection_config.json")

    # Tiny not working now
    # detector.setModelPath(r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\hakkai\hakkai_config_yolov3_tiny\yolov3tiny.h5")
    # detector.setJsonPath(r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\hakkai\hakkai_config_yolov3_tiny\hk_tiny_detection_config.json")

    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=r"E:\acuity\acuity_projects\hakkai_project\hakkai_datasets\hakkai_test_data\hakkai_test_imgs\Image_C0_0359155.jpg", 
                                                output_image_path=r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\hk5.jpg")
    for detection in detections:
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])

def extract_images():

    detector = CustomObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\models_5epocks\detection_model-ex-005--loss-0003.791.h5")
    detector.setJsonPath(r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\json\detection_config.json") 
    detector.loadModel()
    detections, extracted_objects_array = detector.detectObjectsFromImage(input_image=r"E:\acuity\acuity_projects\AR\datasets\ARmarker_data\_15-21jan\20200115-1IMG_1417.JPG", 
                                                        output_image_path=r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\ar2.jpg", 
                                                        extract_detected_objects=True, minimum_percentage_probability=50, 
                                                        display_object_name=False, display_percentage_probability=False)

    for detection, object_path in zip(detections, extracted_objects_array):
        print(object_path)
        print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
        print("---------------")

if __name__ == '__main__':
    # https://github.com/nguyenanhtuan1008/ImageAI/blob/master/imageai/Detection/Custom/CUSTOMDETECTION.md
    # detect_single_image()
    # detect_single_image_hakkai()

    # Extract images for finetune
    extract_images()