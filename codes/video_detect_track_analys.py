from tmodules.Detection import VideoObjectDetection
import os
import cv2

def video_obj_detection():
    execution_path = os.getcwd()

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\detection\yolo.h5"))
    detector.loadModel()

    video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\data-videos\traffic.mp4"),
                                    output_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\traffic_detected_yolov3.mp4")
                                    , frames_per_second=20, log_progress=True)
    print(video_path)

def custom_video_obj_detection():
    execution_path = os.getcwd()

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\detection\yolo.h5"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(person=True, bicycle=True, motorcycle=True)

    video_path = detector.detectCustomObjectsFromVideo(
                    custom_objects=custom_objects,
                    input_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\data-videos\traffic.mp4"),
                    output_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\traffic_custom_detected.mp4"),
                    frames_per_second=20, log_progress=True)
    print(video_path)

def camera_live_video_detection():

    execution_path = os.getcwd()


    camera = cv2.VideoCapture(r"E:\acuity\tuan_experiment\yolo\ImageAI\data-videos\traffic.mp4")

    detector = VideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\detection\yolo.h5"))
    detector.loadModel()


    video_path = detector.detectObjectsFromVideo(
                    camera_input=camera,
                    output_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\camera_detected_video.mp4"),
                    frames_per_second=20, log_progress=True, minimum_percentage_probability=40)

if __name__ == '__main__':
    # Video obj detection
    # video_obj_detection()

    # Video obj detection custom
    # custom_video_obj_detection()

    # Camera
    camera_live_video_detection()