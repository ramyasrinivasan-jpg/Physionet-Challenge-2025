1)Install the dependencies for these scripts by creating a Docker image :  docker build -t image .

2)Train model by running : python train_model.py -d training_data -m model

3)Run trained model by running : python run_model.py -d holdout_data -m model -o holdout_outputs

4)Evaluate your model: python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv


Notes:

1)We required GPU CUDA for running the code

2) Installing torch+cu126 is must
   
3)Training dataset I used for training the model:  SaMi-Trop dataset and 75% of CODE-15% dataset

4) Validation dataset I used for training the model: Remining 25% of CODE-15% dataset
   
