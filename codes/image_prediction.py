from tmodules.Prediction import ImagePrediction
import os

execution_path = os.getcwd()

def prediction_single_image():
    prediction = ImagePrediction()
    prediction.setModelTypeAsResNet()
    prediction.setModelPath(os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\resnet50_weights_tf_dim_ordering_tf_kernels.h5")) # Download the model via this link https://github.com/OlafenwaMoses/ImageAI/releases/tag/1.0
    prediction.loadModel()

    predictions, probabilities = prediction.predictImage(os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\data-images\1.jpg"), result_count=10)
    for eachPrediction, eachProbability in zip(predictions, probabilities):
        print(eachPrediction , " : " , eachProbability)

def predict_folder_images():
    multiple_prediction = ImagePrediction()
    multiple_prediction.setModelTypeAsResNet()
    multiple_prediction.setModelPath(os.path.join(execution_path, r"E:\acuity\tuan_experiment\yolo\ImageAI\weights\resnet50_weights_tf_dim_ordering_tf_kernels.h5"))
    multiple_prediction.loadModel()

    all_images_array = []

    folder_path = r"E:\acuity\tuan_experiment\yolo\ImageAI\data-images"
    all_files = os.listdir(folder_path)
    for each_file in all_files:
        print(folder_path + "\\" + each_file)
        path_file =  folder_path + "\\" + each_file
        if(each_file.endswith(".jpg") or each_file.endswith(".png")):
            all_images_array.append(path_file)

    results_array = multiple_prediction.predictMultipleImages(all_images_array, result_count_per_image=5)

    for each_result in results_array:
        predictions, percentage_probabilities = each_result["predictions"], each_result["percentage_probabilities"]
        for index in range(len(predictions)):
            print(predictions[index] , " : " , percentage_probabilities[index])
        print("-----------------------")
if __name__ == '__main__':
    
    # Prediction single image
    # prediction_single_image()

    # Prediction folder images
    # predict_folder_images()
    