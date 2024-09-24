from ultralytics import YOLO
import sys
import cv2
import os
import torch
from time import time
from torchvision.ops import nms
import torch.nn as nn 
from torchvision import transforms as T
from PIL import Image

# Correr desde root no desde fruit_ripeness

cloned_repo_path = './RT-DETR/rtdetrv2_pytorch/'
sys.path.append(os.path.abspath(cloned_repo_path))

from src.core import YAMLConfig

# Load the YOLO model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    raise ValueError("Cuda not available")


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs

def load_rtdetrv2_model(cfg_path, model_path, device):
    # Load the YAML configuration for the model
    cfg = YAMLConfig(cfg_path, resume=model_path)
    
    # Load the model state dictionary from the .pth file
    state = torch.load(model_path, map_location=device)
    
    if 'ema' in state:
        print('Using EMA weights from the model state')
        state = state['ema']['module']  # Extract EMA weights
    else:
        print('Using standard weights from the model state')
        state = state['model']  # Extract regular weights
    
    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    return cfg

# def save_bounding_boxes(tracks, output_txt_path, frame_id):
#     with open(output_txt_path, 'a') as f:
#         for track in tracks:
#             box = track['bbox']  # Adjust for RT-DETRv2 API
#             x_center, y_center, width, height = box

#             confidence = track['confidence']
#             class_id = track['class']

#             track_id = track.get('id', -1)  # Handle cases where track_id might not exist

#             # Normalize bounding box coordinates
#             x_center /= track['orig_shape'][1]  # width of the image
#             y_center /= track['orig_shape'][0]  # height of the image
#             width /= track['orig_shape'][1]
#             height /= track['orig_shape'][0]
            
#             f.write(f"{frame_id} {track_id} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.4f}\n")

def save_bounding_boxes(labels, boxes, scores, output_txt_path, frame_id, frame_width, frame_height):
    with open(output_txt_path, 'a') as f:
        for label, box, score in zip(labels, boxes, scores):
            x1, y1, x2, y2 = box  # Assuming box is in [x1, y1, x2, y2] format
            width = x2 - x1
            height = y2 - y1

            # Normalize bounding box coordinates
            x_center = (x1 + width / 2) / frame_width
            y_center = (y1 + height / 2) / frame_height
            norm_width = width / frame_width
            norm_height = height / frame_height

            # Write the bounding box info to the file
            f.write(f"{frame_id} -1 {label.item()} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f} {score.item():.4f}\n")

               
def annotate_video(video_path, model, output_dir, output_txt_path, save_every_frame=True):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer for saving the annotated video
    output_video_path = os.path.join(output_dir, 'annotated_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Set up the transformation (Resize to 640x640 and convert to tensor)
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # # Convert frame to tensor and normalize (assuming model input expects normalized values)
        # frame_tensor = torch.from_numpy(frame).float().to(device)
        # frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (batch_size, channels, height, width)

        # orig_size_tensor = torch.tensor([[frame_height, frame_width]]).to(device)

        # Convert frame to PIL image for transformation
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Apply the transformation to resize and convert to tensor
        im_data = transforms(im_pil)[None].to(device)  # Add batch dimension

        # Set the original target size (width, height) as a tensor
        orig_size = torch.tensor([frame_width, frame_height])[None].to(device)  # Shape becomes [1, 2]

        
        print("frame tensor shape",im_data.shape)  # Shape of input frame
        print("orig size tensor",orig_size.shape)  # Shape of orig_target_sizes
        #print("output",outputs.shape)  # Shape of model output
        
        # Perform inference on the current frame using RT-DETRv2
        with torch.no_grad():
            outputs = model(im_data, orig_size)
            labels, boxes, scores = outputs

        # Post-process the output (this part may depend on the exact API of RT-DETRv2)
        #results = post_process_detections(outputs, frame.shape)  # Adjust based on the repo’s post-processing function

        labels = labels.squeeze(0)  # Shape becomes [300]
        boxes = boxes.squeeze(0)  # Shape becomes [300, 4]
        scores = scores.squeeze(0)  # Shape becomes [300]

        score_mask = scores > 0.2
        class_mask = labels != 69
        for class_num in [2, 6, 8, 13, 14, 43, 45, 61, 66, 69, 71]:
            mask = labels != class_num
            class_mask = class_mask & mask
        combined_mask = score_mask & class_mask

        labels = labels[combined_mask]
        boxes = boxes[combined_mask]
        scores = scores[combined_mask]

        #print(f"Labels: {labels.shape if isinstance(labels, torch.Tensor) else len(labels)}")
        #print(f"Boxes: {boxes.shape if isinstance(boxes, torch.Tensor) else len(boxes)}")
        #print(f"Scores: {scores.shape if isinstance(scores, torch.Tensor) else len(scores)}")
        
        # Annotate the frame
        annotated_frame = frame.copy()
        #counter = 0
        for label, box, score in zip(labels, boxes, scores):
            #counter +=1
            #print(counter)
            #print(box)
            x1, y1, x2, y2 = box  # Assuming box is in [x1, y1, x2, y2] format

            # Draw bounding box and label
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Label: {label.item()} Score: {round(score.item(), 2)}", (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #print("out of for loop")
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame: {frame_id}"
        position = (10, 30)
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(annotated_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # # Process the results
        # for track in results:
        #     if 'id' in track:
        #         track_id = track['id']
        #     else:
        #         track_id = '-1'
        #     x, y, w, h = track['bbox']  # Adjust for RT-DETRv2 API
        #     class_id = track['class']

        #     # Draw bounding box and text on the frame
        #     cv2.putText(annotated_frame, f"ID: {track_id}, Class: {class_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #     cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 4)

        if save_every_frame or len(results) > 0:
            frame_filename = os.path.join(output_dir, f"annotated_frame_{frame_id:04d}.png")
            cv2.imwrite(frame_filename, annotated_frame)
            #print("Going to save")
            save_bounding_boxes(labels, boxes, scores, output_txt_path, frame_id, frame_width, frame_height)
            print(f"Saved {frame_filename}")

        # Write the annotated frame to the video
        video_writer.write(annotated_frame)
        frame_id += 1

    cap.release()
    video_writer.release()
    print(f"Annotation complete. {frame_id} frames processed.")
    print(f"Saved video to {output_video_path}")

def annotate_all_videos(folder_path, model, output_dir):
    print("Starting annotation...")
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".h264"):
            video_path = os.path.join(folder_path, file_name)
            print(f"Annotating video: {video_path}")
            
            video_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
            os.makedirs(video_output_dir, exist_ok=True)

            output_txt_path = os.path.join(video_output_dir, "annotations.txt")
            
            start_time = time()
            try:
                annotate_video(video_path, model, video_output_dir, output_txt_path)
            except Exception as e:
                print(f"Failed to annotate video {file_name}: {e}")
                continue
            
            end_time = time()
            print(f"Video {file_name} took {end_time - start_time} s")
        else:
            print(f"{file_name} not a video")

if __name__ == "__main__":
    start_annotation_time = time()
    folder_path = "./videos/naranja_videos3/raw_frames"
    
    # Load the RT-DETRv2 model from the .pth file
    model_path = "./fruit-ripeness-classificator/rtdetrv2_r101vd_6x_coco_from_paddle.pth"  # Replace with the actual path to the .pth file
    cfg_path = "./fruit-ripeness-classificator/rtdetrv2_r101vd_6x_coco.yml"
    
    cfg = load_rtdetrv2_model(cfg_path, model_path, device)
    model = Model(cfg).to(device)

    #output = model(im_data, orig_size)
    #labels, boxes, scores = output

    #draw([im_pil], labels, boxes, scores)
    
    output_dir = "./videos/naranja_videos3/rt-detrv2-frames"
    
    annotate_all_videos(folder_path, model, output_dir)
    end_annotation_time = time()
    print(f"Whole process took {end_annotation_time - start_annotation_time} s")