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

def save_bounding_boxes(tracks, output_txt_path, frame_id):
    with open(output_txt_path, 'a') as f:
        for track in tracks:
            # Each track already contains xywh and other details in the detection results
            box = track.xywh  # Access xywh format
            x_center, y_center, width, height = box.tolist()[0]  # Convert to list for better handling
            confidence = track.conf.item()  # Confidence score
            class_id = track.cls.item()  # Class ID

            if track.id:
                track_id = track.id.item()
            else:
                track_id ='-1'
            
            # Normalize bounding box coordinates
            x_center /= track.orig_shape[1]  # width of the image
            y_center /= track.orig_shape[0]  # height of the image
            width /= track.orig_shape[1]
            height /= track.orig_shape[0]
            
            f.write(f"{frame_id} {track_id} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence:.4f}\n")
         
               
def annotate_video(video_path, model_name, output_dir, output_txt_path, save_every_frame=True):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the YOLOv10 model
    model = YOLO(model_name)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define video writer for saving the annotated video
    output_video_path = os.path.join(output_dir,'annotated_video.avi')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for mp4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame
        results = model.track(source=frame, device=device, persist=True, tracker="bytetrack.yaml", show=False, classes=49)#[32,46,49])
        #results_track = model.track(source=frame, device=device, persist=True, tracker="bytetrack.yaml")
        #print(results)

        #asdasdsa

        # for idx, result in enumerate(results):
        #     dets = result.boxes.data.cpu().detach()  # Convert to CPU
        #     boxes = dets[:, :4]  # x1, y1, x2, y2
        #     scores = dets[:, 4]  # confidence
        #     class_ids = dets[:, 5]  # class IDs

        #     # Apply NMS
        #     indices = nms(boxes, scores, iou_threshold=0.5)
        #     #result.boxes.data = dets[indices] #ojo con esta parte
        #     boxes = boxes[indices]
        #     scores = scores[indices]
        #     class_ids = class_ids[indices]

        #     # Prepare detections for ByteTrack
        #     detections = []
        #     for i in range(len(boxes)):
        #         x1, y1, x2, y2 = boxes[i]
        #         tlwh = [x1.item(), y1.item(), (x2 - x1).item(), (y2 - y1).item()]  # top-left width-height
        #         score = scores[i].item()
        #         cls = int(class_ids[i].item())
        #         detection = STrack(tlwh, score, cls)
        #         detection.frame_width = frame_width
        #         detection.frame_height = frame_height
        #         detections.append(detection)

        
        # Annotate the frame with bounding boxes
        #annotated_frame = results[0].plot()
        annotated_frame = results[0].orig_img

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Frame: {frame_id}"
        position = (10, 30)  # Adjust the position as needed
        font_scale = 1
        color = (0, 255, 0)  # Green color in BGR
        thickness = 2

        # Add text to the frame
        cv2.putText(annotated_frame, text, position, font, font_scale, color, thickness, cv2.LINE_AA)

        for track in results[0].boxes:
            if track.id:
                track_id = track.id.item()
            else:
                track_id = '-1'
            x, y, w, h = track.xywh[0].tolist()
            cv2.putText(annotated_frame, f"ID: {track_id}, Class Id: {track.cls.item()}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(annotated_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 4)


        # Save the annotated frame as an image
        if save_every_frame or results.xyxy[0].size(0) > 0:  # Optionally save only frames with detections
            frame_filename = os.path.join(output_dir, f"annotated_frame_{frame_id:04d}.png")
            cv2.imwrite(frame_filename, annotated_frame)

            #save_bounding_boxes(results, output_txt_path, frame_id)
            # Save bounding boxes and tracks
            save_bounding_boxes(results[0].boxes, output_txt_path, frame_id)
            print(f"Saved {frame_filename}")
        
        # Write the annotated frame to the video
        video_writer.write(annotated_frame)
        frame_id += 1

    cap.release()
    video_writer.release()  # Make sure to release the video writer
    print(f"Annotation complete. {frame_id} frames processed.")
    print(f"Saved video to {output_video_path}")

def annotate_all_videos(folder_path, model_name, output_dir):
    print("Starting to anotate")
    # Iterate through all files in the given folder
    for file_name in os.listdir(folder_path):
        #print(f"Checking {file_name}")
        if file_name.endswith(".h264"):  # Check if the file is a video
            video_path = os.path.join(folder_path, file_name)
            print(f"Annotating video: {video_path}")
            
            # Create a specific output directory for each video
            video_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
            os.makedirs(video_output_dir, exist_ok=True)

            output_txt_path = os.path.join(video_output_dir, "annotations.txt")
            
            start_time = time()
            # Call your existing annotate_video function
            annotate_video(video_path, model_name, video_output_dir, output_txt_path)
            
            end_time = time()
            print(f"Video {file_name} took {end_time - start_time} s")

if __name__ == "__main__":
    start_anotation_time = time()
    folder_path = "./model/dataset/naranja_videos2"
    model_name = "yolov10x.pt"  # Path to your YOLOv10 model
    #model_name = "yolov9e.pt"
    output_dir = "output/naranja_videos2/annotated_frames_persisted_true"
    
    annotate_all_videos(folder_path, model_name, output_dir)
    end_anotation_time = time()
    print(f"Whole process took {end_anotation_time - start_anotation_time} s")