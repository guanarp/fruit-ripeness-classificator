from time import time
import cv2
import os
from ultralytics import YOLO, solutions
import torch


# Load the YOLO model and use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    raise ValueError("Cuda not available")

def count_all_videos(folder_path, model_name, output_dir, classes_to_count=[49]):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".h264"):  # Check if the file is a video
            video_path = os.path.join(folder_path, file_name)
            print(f"Counting video: {video_path}")

            # Create a specific output directory for each video
            video_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
            #os.makedirs(video_output_dir, exist_ok=True)
            
            start_time = time()

            model = YOLO(model_name)
            cap = cv2.VideoCapture(video_path)
            w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

            # Define region points
            region_points = [(364, 0), (365, 1088)]

            # Video writer
            video_writer = cv2.VideoWriter(video_output_dir + file_name[:-4] + "mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Init Object Counter
            counter = solutions.ObjectCounter(
                view_img=True,
                reg_pts=region_points,
                names=model.names,
                draw_tracks=True,
                line_thickness=2
            )

            while cap.isOpened():
                success, im0 = cap.read()
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.")
                    break
                tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)

                im0 = counter.start_counting(im0, tracks)
                video_writer.write(im0)

            cap.release()
            video_writer.release()
            
            end_time = time()
            print(f"Video {file_name} took {end_time - start_time} s")
    


if __name__ == "__main__":
    start_anotation_time = time()
    folder_path = "./model/dataset/naranja_videos2"
    model_name = "yolov10x.pt"  # Path to your YOLOv10 model
    #model_name = "yolov9e.pt"
    output_dir = "output2/naranja_videos2/counting_videos"
    
    count_all_videos(folder_path, model_name, output_dir)

    cv2.destroyAllWindows()

    end_anotation_time = time()
    print(f"Whole process took {end_anotation_time - start_anotation_time} s")





