import os
import cv2

class_color_map = {
    1: (0, 255, 0),    # Green (unripe)
    2: (0, 165, 255),  # Yellow (ripening)
    3: (0, 140, 255),  # Orange (optimal ripeness)
    4: (0, 69, 255),   # Darker orange (overripe)
    5: (42, 42, 165),   # Brown (rotten)
    -1: (255,0,0)       # Red (default class)
}

class_name_map = {
    1: "Unripe",
    2: "Ripening",
    3: "Optimal",
    4: "Overripe",
    5: "Rotten"
}

def render_detections(video_path, annotation_file, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    #total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

    # Load annotations
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            frame_number = int(parts[0])
            track_id = int(float(parts[1]))
            try:
                class_id = int(float(parts[2]))  # You can use this to color the boxes by class
            except:
                class_id = -1
            try:
                x_center, y_center, width, height = map(float, parts[3:7])
            except:
                print(parts)
                print(annotation_file)
                asdadasda

            if frame_number not in annotations:
                annotations[frame_number] = []
            annotations[frame_number].append((x_center, y_center, width, height, class_id))

    # Process each frame
    current_frame = 0
    print(annotations)
    while cap.isOpened():
        #print(current_frame)
        ret, frame = cap.read()
        if not ret:
            break  # No more frames to read

        # Draw detections on the current frame
        #
        if current_frame in annotations:
            print("entered")
            print(annotations[current_frame])
            for detection in annotations[current_frame]:
                x_center, y_center, width, height, class_id = detection

                # Convert YOLO coordinates to pixel coordinates
                x1 = int((x_center - width / 2) * frame_width)
                y1 = int((y_center - height / 2) * frame_height)
                x2 = int((x_center + width / 2) * frame_width)
                y2 = int((y_center + height / 2) * frame_height)

                # Draw a rectangle and label the class
                color = class_color_map.get(class_id, (255, 255, 255))  # Default to white if class_id not found
                class_name = class_name_map.get(class_id, "Unknown")  # Default to "Unknown" if class_id not found
                # Draw a rectangle and label the class
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f'{class_name} (Ripeness {class_id})'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Write the frame with detections into the output video
        out.write(frame)

        current_frame += 1
        if current_frame >= max(annotations.keys()):
            break

    # Release video objects
    cap.release()
    out.release()

    print(f"Finished rendering detections for {video_path}. Output saved as {output_video_path}")


def process_videos_with_detections(video_folder, annotations_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    for video_file in os.listdir(video_folder):
        if video_file.endswith(".h264"):  # Process video files (you can adjust for other formats if needed)
            video_path = os.path.join(video_folder, video_file)
            
            # Assuming the annotation file is named 'annotations.txt' and located in a folder with the same name as the video
            video_name = os.path.splitext(video_file)[0]
            #annotation_file = os.path.join(annotations_folder, video_name, 'annotations.txt')
            annotation_file = os.path.join(annotations_folder,f'{video_name}_modified.txt')
            
            if not os.path.exists(annotation_file):
                print(f"Annotation file {annotation_file} not found for video {video_file}. Skipping...")
                continue

            output_video_path = os.path.join(output_folder, f"{video_name}_with_ripeness_detections.avi")

            # Render detections on the video
            render_detections(video_path, annotation_file, output_video_path)


if __name__ == "__main__":
    for i in range(1,4):
        video_folder = f'./videos/naranja_videos{i}/raw_frames'  # Folder with original videos
        annotations_folder = f'./fruit-ripeness-classificator/output/new_class_annotationsv2/naranja_videos{i}'  # Folder with updated annotations
        output_folder = './fruit-ripeness-classificator/output/videos_with_new_detections_classv2'  # Folder to save the output videos
    
        # Process all videos in the folder and render detections
        process_videos_with_detections(video_folder, annotations_folder, output_folder)
