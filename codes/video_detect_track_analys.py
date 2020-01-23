from tmodules.Detection import VideoObjectDetection
import os
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

if __name__ == '__main__':
    # Video obj detection
    video_obj_detection()