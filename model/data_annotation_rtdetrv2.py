from ultralytics import YOLO
import cv2
import os
import torch
from time import time
from torchvision.ops import nms

# Load the YOLO model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    raise ValueError("Cuda not available")

def load_rtdetrv2_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def save_bounding_boxes(tracks, output_txt_path, frame_id):
    with open(output_txt_path, 'a') as f:
        for track in tracks:
            box = track['bbox']  # Adjust for RT-DETRv2 API
            x_center, y_center, width, height = box

            confidence = track['confidence']
            class_id = track['class']

            track_id = track.get('id', -1)  # Handle cases where track_id might not exist

            # Normalize bounding box coordinates
            x_center /= track['orig_shape'][1]  # width of the image
            y_center /= track['orig_shape'][0]  # height of the image
            width /= track['orig_shape'][1]
            height /= track['orig_shape'][0]
            
            f.write(f"{frame_id} {track_id} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.4f}\n")
         
               
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to tensor and normalize (assuming model input expects normalized values)
        frame_tensor = torch.from_numpy(frame).float().to(device)
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (batch_size, channels, height, width)
        
        # Perform inference on the current frame using RT-DETRv2
        with torch.no_grad():
            outputs = model(frame_tensor)

        # Post-process the output (this part may depend on the exact API of RT-DETRv2)
        #results = post_process_detections(outputs, frame.shape)  # Adjust based on the repo’s post-processing function

        annotated_frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame: {frame_id}"
        position = (10, 30)
        font_scale = 1
        color = (0, 255, 0)
        thickness = 2
        cv2.putText(annotated_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        # Process the results
        for track in results:
            if 'id' in track:
                track_id = track['id']
            else:
                track_id = '-1'
            x, y, w, h = track['bbox']  # Adjust for RT-DETRv2 API
            class_id = track['class']

            # Draw bounding box and text on the frame
            cv2.putText(annotated_frame, f"ID: {track_id}, Class: {class_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 4)

        if save_every_frame or len(results) > 0:
            frame_filename = os.path.join(output_dir, f"annotated_frame_{frame_id:04d}.png")
            cv2.imwrite(frame_filename, annotated_frame)

            save_bounding_boxes(results, output_txt_path, frame_id)
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

if __name__ == "__main__":
    start_annotation_time = time()
    folder_path = "./model/dataset/naranja_videos3"
    
    # Load the RT-DETRv2 model from the .pth file
    model_path = "rtdetrv2.pth"  # Replace with the actual path to the .pth file
    model = load_rtdetrv2_model(model_path, device)
    
    output_dir = "output/naranja_videos1/annotated_frames_persisted_true"
    
    annotate_all_videos(folder_path, model, output_dir)
    end_annotation_time = time()
    print(f"Whole process took {end_annotation_time - start_annotation_time} s")