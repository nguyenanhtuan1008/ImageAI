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

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

def forSeconds(second_number, output_arrays, count_arrays, average_output_count):
    print("SECOND : ", second_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last second: ", average_output_count)
    print("------------END OF A SECOND --------------")

def forMinute(minute_number, output_arrays, count_arrays, average_output_count):
    print("MINUTE : ", minute_number)
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)
    print("------------END OF A MINUTE --------------")

def video_analys():
    execution_path = os.getcwd()
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\detection\yolo.h5"))
    video_detector.loadModel(detection_speed="fast")

    video_detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\data-videos\traffic.mp4"),
        output_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\analys_traffic_detected.mp4"),
        frames_per_second=10,
        per_second_function=forSeconds,
        per_frame_function=forFrame,
        per_minute_function=forMinute,
        minimum_percentage_probability=30
    )

def forFull(output_arrays, count_arrays, average_output_count):
    #Perform action on the 3 parameters returned into the function
    print("Array for the outputs of each frame ", output_arrays)
    print("Array for output count for unique objects in each frame : ", count_arrays)
    print("Output average count for unique objects in the last minute: ", average_output_count)

def video_analys_full():
    execution_path = os.getcwd()
    video_detector = VideoObjectDetection()
    video_detector.setModelTypeAsTinyYOLOv3()
    video_detector.setModelPath(os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\detection\yolo-tiny.h5"))
    video_detector.loadModel()

    video_detector.detectObjectsFromVideo(
        input_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\data-videos\traffic.mp4"),
        output_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\analys_full_traffic_detected.mp4"),
        frames_per_second=10,
        video_complete_function=forFull,
        minimum_percentage_probability=30
    )

def camera_detection_timeout():
    execution_path = os.getcwd()
    camera = cv2.VideoCapture(r"E:\acuity\tuan_experiment\yolo\ImageAI\data-videos\traffic.mp4")

    detector = VideoObjectDetection()
    detector.setModelTypeAsTinyYOLOv3()
    detector.setModelPath(os.path.join(execution_path , r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\detection\yolo-tiny.h5"))
    detector.loadModel()


    video_path = detector.detectObjectsFromVideo(camera_input=camera,
                                                output_file_path=os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\codes\output\camera_detected_video_timeout.mp4"),
                                                frames_per_second=20,
                                                log_progress=True,
                                                minimum_percentage_probability=40,
                                                detection_timeout=120)

if __name__ == '__main__':
    # https://github.com/nguyenanhtuan1008/ImageAI/blob/master/imageai/Detection/VIDEO.md
    # Video obj detection
    # video_obj_detection()

    # Video obj detection custom
    # custom_video_obj_detection()

    # Camera
    # camera_live_video_detection()

    # Analys
    # video_analys()

    # Analys
    # video_analys_full()

    #Time out
    camera_detection_timeout()