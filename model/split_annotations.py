import os
import shutil  # To move files

# Paths
annotation_folder = "new_class_annotations/naranja_videos2/"
image_folder = "model/dataset/images/train/"
label_output_folder = "model/dataset/labels/train/"
image_output_folder = "model/dataset/images/train/"


# Create the output directory if it doesn't exist
os.makedirs(label_output_folder, exist_ok=True)

# Loop through each annotation file (one for each video)
for annotation_file in os.listdir(annotation_folder):
    annotation_path = os.path.join(annotation_folder, annotation_file)

    video_name = annotation_path.split('/')[2].replace("_modified.txt",'')
    
    # Read the corresponding annotation file
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    # Create a dictionary to hold annotations per frame
    frame_annotations = {}

    # Loop through each line in the annotation file
    for line in annotations:
        # Parse the annotation line (frame_id track_id class_id ...)
        parts = line.strip().split()
        frame_id = int(parts[0])  # Assuming frame_id is the first element in the line
        track_id = parts[1]       # Track ID (if needed, might not be used)
        class_id = parts[2]       # Class ID
        bbox = parts[3:]          # The rest of the line contains the bounding box information (YOLO format)

        # Convert bbox to the YOLO format (class_id x_center y_center width height)
        annotation = f"{class_id} {' '.join(bbox)}\n"

        # Append the annotation to the frame's list
        if frame_id not in frame_annotations:
            frame_annotations[frame_id] = []
        frame_annotations[frame_id].append(annotation)

    # For each frame in the annotations, create a separate txt file
    for frame_id, annotations in frame_annotations.items():
        # Construct the corresponding image filename (assuming frame_{frame_id}.jpg format)
        frame_filename = f"{video_name}_frame_{frame_id}.jpg"
        frame_txt_filename = f"{video_name}_frame_{frame_id}.txt"
        

        original_image_path = f'{image_folder}{video_name}_frame_{frame_id}.jpg'

        # Check if the corresponding image file exists
        if os.path.exists(original_image_path):
            output_dir = os.path.join(label_output_folder, video_name)

            # Write the annotations for this frame to a new .txt file
            output_txt_path = os.path.join(label_output_folder, frame_txt_filename)
            with open(output_txt_path, 'w') as label_file:
                label_file.writelines(annotations)

            print(f"Created label file for {frame_filename}: {output_txt_path}")

            # Move the image if it's not already in the model/dataset/images/train/ directory
            final_image_path = os.path.join(image_output_folder, frame_filename)
            if not os.path.exists(final_image_path):
                print(f"Moving image {frame_filename} to {image_output_folder}")
                shutil.move(original_image_path, final_image_path)
            else:
                print(f"Image {frame_filename} already exists in {image_output_folder}, skipping move.")
        else:
            print(f"{original_image_path} not found. Skipping annotation.")

            # After processing all frames for the video, delete the folder along with its contents
