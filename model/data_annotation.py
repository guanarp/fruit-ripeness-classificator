from ultralytics import YOLO
import cv2
import os

def save_bounding_boxes(results, output_txt_path):
    with open(output_txt_path, 'w') as f:
        for result in results.xywh[0]:  # Accessing results in xywh format
            class_id = int(result[5])  # Assuming class ID is the 6th element
            x_center, y_center, width, height = result[0:4]
            # Normalize bounding box coordinates
            x_center /= results.orig_shape[1]  # width of the image
            y_center /= results.orig_shape[0]  # height of the image
            width /= results.orig_shape[1]
            height /= results.orig_shape[0]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

def annotate_video(video_path, model_path, output_dir, save_every_frame=True):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the YOLOv10 model
    model = YOLO(model_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference on the current frame
        results = model(frame)

        # Annotate the frame with bounding boxes
        annotated_frame = results.render()[0]

        # Save the annotated frame as an image
        if save_every_frame or results.xyxy[0].size(0) > 0:  # Optionally save only frames with detections
            frame_filename = os.path.join(output_dir, f"annotated_frame_{frame_id:04d}.jpg")
            txt_filename = os.path.join(output_dir, f"annotated_frame_{frame_id:04d}.txt")
            cv2.imwrite(frame_filename, annotated_frame)
            save_bounding_boxes(results, txt_filename)
            print(f"Saved {frame_filename} and {txt_filename}")
        
        frame_id += 1

    cap.release()
    print(f"Annotation complete. {frame_id} frames processed.")

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"
    model_path = "path/to/yolov10.pt"  # Path to your YOLOv10 model
    output_dir = "output/annotated_frames"
    
    annotate_video(video_path, model_path, output_dir)
