from tmodules.Prediction.Custom import ModelTraining
from tmodules.Prediction.Custom import CustomImagePrediction
import os

def train_custom_model():
    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai")
    model_trainer.trainModel(num_objects=2, num_experiments=10, enhance_data=True, batch_size=32, show_network_summary=True, save_full_model=False)

def train_idenprof():
    from io import open
    import requests
    import shutil
    from zipfile import ZipFile
    import os
    from imageai.Prediction.Custom import ModelTraining

    execution_path = os.getcwd()

    TRAIN_ZIP_ONE = os.path.join(execution_path, "idenprof-train1.zip")
    TRAIN_ZIP_TWO = os.path.join(execution_path, "idenprof-train2.zip")
    TEST_ZIP = os.path.join(execution_path, "idenprof-test.zip")

    DATASET_DIR = os.path.join(execution_path, "idenprof")
    DATASET_TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    DATASET_TEST_DIR = os.path.join(DATASET_DIR, "test")

    if(os.path.exists(DATASET_DIR) == False):
        os.mkdir(DATASET_DIR)
    if(os.path.exists(DATASET_TRAIN_DIR) == False):
        os.mkdir(DATASET_TRAIN_DIR)
    if(os.path.exists(DATASET_TEST_DIR) == False):
        os.mkdir(DATASET_TEST_DIR)

    if(len(os.listdir(DATASET_TRAIN_DIR)) < 10):
        if(os.path.exists(TRAIN_ZIP_ONE) == False):
            print("Downloading idenprof-train1.zip")
            data = requests.get("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-train1.zip", stream = True)
            with open(TRAIN_ZIP_ONE, "wb") as file:
                shutil.copyfileobj(data.raw, file)
            del data
        if (os.path.exists(TRAIN_ZIP_TWO) == False):
            print("Downloading idenprof-train2.zip")
            data = requests.get("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-train2.zip", stream=True)
            with open(TRAIN_ZIP_TWO, "wb") as file:
                shutil.copyfileobj(data.raw, file)
            del data
        print("Extracting idenprof-train1.zip")
        extract1 = ZipFile(TRAIN_ZIP_ONE)
        extract1.extractall(DATASET_TRAIN_DIR)
        extract1.close()
        print("Extracting idenprof-train2.zip")
        extract2 = ZipFile(TRAIN_ZIP_TWO)
        extract2.extractall(DATASET_TRAIN_DIR)
        extract2.close()

    if(len(os.listdir(DATASET_TEST_DIR)) < 10):
        if (os.path.exists(TEST_ZIP) == False):
            print("Downloading idenprof-test.zip")
            data = requests.get("https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-test.zip", stream=True)
            with open(TEST_ZIP, "wb") as file:
                shutil.copyfileobj(data.raw, file)
            del data
        print("Extracting idenprof-test.zip")
        extract = ZipFile(TEST_ZIP)
        extract.extractall(DATASET_TEST_DIR)
        extract.close()

    model_trainer = ModelTraining()
    model_trainer.setModelTypeAsResNet()
    model_trainer.setDataDirectory(DATASET_DIR)
    model_trainer.trainModel(num_objects=10, num_experiments=100, enhance_data=True, batch_size=32, show_network_summary=True)

def continue_model_training():
    
    trainer = ModelTraining()
    trainer.setModelTypeAsResNet()
    trainer.setDataDirectory(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai")
    trainer.trainModel(num_objects=2, num_experiments=5, enhance_data=True, batch_size=8, show_network_summary=True, 
                        continue_from_model= r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai\models\model_ex-010_acc-0.760417.h5", save_full_model=False)

def tranfer_learning():

    trainer = ModelTraining()
    trainer.setModelTypeAsResNet()
    trainer.setDataDirectory(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai")
    trainer.trainModel(num_objects=2, num_experiments=20, enhance_data=True, batch_size=32, show_network_summary=True,
                        transfer_from_model=r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\prediction\resnet50_weights_tf_dim_ordering_tf_kernels.h5", 
                        initial_num_objects=1000)
def custom_images_prediction():
    execution_path = os.getcwd()

    prediction = CustomImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai\models\model_ex-002_acc-0.663194.h5")
    prediction.setJsonPath(os.path.join(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai\json\model_class.json"))
    prediction.loadModel(num_objects=2)

    predictions, probabilities = prediction.predictImage(r"E:\acuity\acuity_projects\AR\datasets\cvat\_stain_fray\images_croped_imageai\test\fray\20191202IMG_0360.jpg", 
                                                        result_count=2)

    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(str(eachPrediction) + " : " + str(eachProbability))


if __name__ == '__main__':

    # Train custom model
    # train_custom_model()

    # Continue traning model
    # continue_model_training()

    # Tranfer learning
    # tranfer_learning()
    
    # Custom image prediction
    custom_images_prediction()