from tmodules.Detection import ObjectDetection
import os

def detect_single_images():
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\yolo.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\data-images\image2.jpg"), output_image_path=os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\image2new.jpg"), minimum_percentage_probability=30)
    
    # detections = detector.detectObjectsFromImage(input_type="array", input_image=image_array , output_image_path=os.path.join(execution_path , "image.jpg")) # For numpy array input type
    # detections = detector.detectObjectsFromImage(input_type="stream", input_image=image_stream , output_image_path=os.path.join(execution_path , "test2new.jpg")) # For file stream input type
    # detected_image_array, detections = detector.detectObjectsFromImage(output_type="array", input_image="image.jpg" ) # For numpy array output type

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

def detect_extraction():
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\yolo.h5"))
    detector.loadModel()

    detections, objects_path = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , 
                                                                r"E:\acuity\tuan_experiment\yolo\ImageAI\data-images\image3.jpg"), 
                                                                output_image_path=os.path.join(r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output", "image3new.jpg"),
                                                                minimum_percentage_probability=30,  extract_detected_objects=True)

    for eachObject, eachObjectPath in zip(detections, objects_path):
        print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("Object's image saved in " + eachObjectPath)
        print("--------------------------------")

def custom_obj_detection():
    execution_path = os.getcwd()

    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\yolo.h5"))
    detector.loadModel(detection_speed="fast") #  "normal"(default), "fast", "faster" , "fastest" and "flash"

    custom_objects = detector.CustomObjects(car=True, motorcycle=True)
    detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, 
                                                        input_image=os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\data-images\image3.jpg"), 
                                                        output_image_path=os.path.join(r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output", "image3custom.jpg"), 
                                                        minimum_percentage_probability=30)
    # detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\data-images\image3.jpg"), 
    #                                                     output_image_path=os.path.join(r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output" , "image3new_nodetails.jpg"), 
    #                                                     minimum_percentage_probability=30, 
    #                                                     display_percentage_probability=False, display_object_name=False)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
        print("--------------------------------")

if __name__ == '__main__':
    # Detect
    # detect_single_images()

    # Extract image
    # detect_extraction()

    # Custom object detection
    custom_obj_detection()