from tmodules.Prediction.Custom import ModelTraining
model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai")
model_trainer.trainModel(num_objects=2, num_experiments=10, enhance_data=True, batch_size=32, show_network_summary=True, save_full_model=False)