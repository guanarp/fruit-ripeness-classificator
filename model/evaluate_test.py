from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO('exps/runs/yolov10_exp64/weights/best.pt')  # Provide the path to your trained model

# Path to the dataset configuration file (.yaml)
data_yaml = 'data.yaml'  # Example: 'data.yaml' with dataset info

# Evaluate the model on the test set
results = model.val(data=data_yaml, split='test')

# Print out the results
print(results)