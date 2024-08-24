from ultralytics import YOLO

def train_yolov10(model_path, data_yaml, epochs=50, imgsz=640):
    # Load the pre-trained YOLOv10 model
    model = YOLO(model_path)
    
    # Train the model on your dataset
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
    
    # Save the fine-tuned model
    model.save("yolov10_finetuned.pt")

if __name__ == "__main__":
    model_path = "path/to/yolov10.pt"  # Path to your pre-trained YOLOv10 model
    data_yaml = "path/to/data.yaml"  # Path to your data.yaml file
    epochs = 50  # Number of epochs to train

    train_yolov10(model_path, data_yaml, epochs=epochs)
