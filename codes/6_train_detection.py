from tmodules.Detection.Custom import DetectionModelTrainer

def training():
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect")
    trainer.setTrainConfig(object_names_array=["fray", "stain"], batch_size=4, num_experiments=50, 
                            train_from_pretrained_model=r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\pretrained-yolov3.h5",)
    # In the above,when training for detecting multiple objects,
    #set object_names_array=["object1", "object2", "object3",..."objectz"]
    trainer.trainModel()

def evaluate():

    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect")
    metrics = trainer.evaluateModel(model_path=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\models_5epocks\detection_model-ex-005--loss-0003.791.h5", 
                                    json_path=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\json\detection_config.json", 
                                    iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)

def evaluate_multi_model():
    trainer = DetectionModelTrainer()
    trainer.setModelTypeAsYOLOv3()
    trainer.setDataDirectory(data_directory=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect")
    metrics = trainer.evaluateModel(model_path=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\models_5epocks",
                                    json_path=r"E:\acuity\acuity_projects\AR\datasets\_imageai_train\images_imageai_detect\json\detection_config.json", 
                                    iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
if __name__ == '__main__':

    # Step 1:
    # training()
    
    # Step 2:
    # Evaluate
    evaluate()

    # Multi model evaluate
    # evaluate_multi_model()
