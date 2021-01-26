

from ml.utils import train_application_classification_cnn_model, train_traffic_classification_cnn_model

DATA_PATH = '/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/dataset/temp_pro/application_classification/train.parquet'
MODEL_PATH = 'model/app.cnn.model'
MODE = 'app'
GPU = True


def main(data_path = DATA_PATH, model_path = MODEL_PATH, task = MODE, gpu = GPU):
    if gpu:
        gpu = -1
    else:
        gpu = None
    if task == 'app':
        train_application_classification_cnn_model(data_path, model_path, gpu)
    elif task == 'traffic':
        train_traffic_classification_cnn_model(data_path, model_path, gpu)
    else:
        exit('Not Support')


if __name__ == '__main__':
    main()
