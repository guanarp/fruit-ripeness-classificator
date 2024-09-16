from ultralytics import YOLO
import torch
#from ultralytics.engine import callback

# class CustomCallback(callback.Callback):
#     def on_epoch_end(self, epoch, metrics):
#         print(f"Epoch {epoch} ended with mAP@0.5 = {metrics['mAP50']:.3f}")
# Free unused CUDA memory

torch.cuda.empty_cache()


def main():
    # Path to the YAML file
    data_yaml = 'data.yaml'
    hyp_yaml = None

    # Initialize the YOLO model
    model = YOLO('yolov10x.pt')  # Path to pre-trained YOLOv10 weights, or use 'yolov10s.pt', etc.

    # Prepare the training arguments
    train_args = {
        'data': data_yaml,       # Path to the YAML file
        'epochs': 100,           # Number of epochs to train for
        'imgsz': 640,            # Image size for training
        'batch': 4,             # You could explore 32, originalmente 16
        'device': '0',           # Device to run on ('0' for GPU, 'cpu' for CPU)
        'workers': 2,            # Number of workers for data loading (4 to 8 for GPU training)
        'project': 'model/runs', # Where to save the trained models
        'name': 'yolov10_exp1',  # Name for the training run
        'resume': False,         # Resume training from a checkpoint if applicable
        'patience': 5,            # Early stopping patience
        'verbose':True,
        'amp':True
        #'callbacks':[CustomCallback()]
    }

    #lr0=0.001,              # Initial learning rate
    # lrf=0.01,               # Final learning rate (learning rate decay factor)
    # warmup_epochs=3,        # Number of warmup epochs
    # warmup_bias_lr=0.1      # Initial learning rate for bias layers during warmu


    # Conditionally add hyp.yaml if it's not None
    if hyp_yaml is not None:
        train_args['hyp'] = hyp_yaml

    # Train the model with the conditionally passed arguments
    model.train(**train_args)

if __name__ == '__main__':
    #import multiprocessing
    #multiprocessing.freeze_support()  # Optional if freezing your program for distribution
    main()