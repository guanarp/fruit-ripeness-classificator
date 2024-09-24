import os
import pandas as pd

# Load Annotations.csv
annotations_df = pd.read_csv('./fruit-ripeness-classificator/Annotations.csv')

det_annotations_path = './fruit-ripeness-classificator/output/corrected_annotations'

# Create an output directory for modified files
output_dir = './fruit-ripeness-classificator/output/new_class_annotationsv2'
os.makedirs(output_dir, exist_ok=True)

# Group the annotations by Video, Folder, and frame range to process each video separately
grouped_annotations = annotations_df.groupby(['Video', 'Folder'])

# Iterate through each group in Annotations.csv
for (video_name, folder_name), group in grouped_annotations:
    # Build the path to the corresponding annotations.txt file
    txt_file_path = os.path.join(det_annotations_path, video_name, 'annotations.txt')
    
    # Build the path to the new modified file
    new_txt_file_path = os.path.join(output_dir, folder_name)
    os.makedirs(new_txt_file_path, exist_ok=True)  # Ensure subfolders exist
    new_txt_file_path = os.path.join(new_txt_file_path, f'{video_name}_modified.txt')

    if os.path.exists(txt_file_path):
        # Read the annotations.txt file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()

        # Create a dictionary to track processed objects for each frame
        processed_objects_in_frame = {}

        # Create a new list to store modified lines
        modified_lines = []
        
        # Process lines in the annotations.txt
        counter=0
        for line in lines:
            
            parts = line.strip().split()

            # if len(parts) < 2:  # Skip invalid lines
            #     modified_lines.append(line)
            #     continue

            frame_id = int(parts[0])  # Assuming the first value in the line is the frame number

            # Initialize the object count for the frame if not present
            if frame_id not in processed_objects_in_frame:
                processed_objects_in_frame[frame_id] = []

            # Check if the frame is within the range specified in the CSV
            frame_annotations = group[(group['Starting_frame'] <= frame_id) & (group['Ending_frame'] >= frame_id)]
            # if len(frame_annotations) > 1:
            #     counter+=1
            #     print(frame_annotations)
            #     print("#################################")
            #     asdsadasdasdsad

            if not frame_annotations.empty:
                # # Get the current object number for this frame
                # object_count = objects_in_frame[frame_id]

                # # Check if the object number is within the range of available annotations
                # if object_count < len(frame_annotations):
                    
                #     # Update the class ID based on the current object in the CSV
                #     class_value = frame_annotations.iloc[object_count]['Class']
                #     parts[2] = str(class_value)  # Assuming the class_id is the third part in the line
                    
                #     # Increment the object count for the current frame
                #     objects_in_frame[frame_id] += 1

                #     #print(f"updating frame {frame_id}, with class {class_value} for {video_name}")
                #     #print(parts)
                
                # Find the first unprocessed annotation for the current frame
                available_annotations = frame_annotations[~frame_annotations.index.isin(processed_objects_in_frame[frame_id])]
                # print(available_annotations)
                # print("#################################")
                
                if not available_annotations.empty:
                    # Get the first available annotation and update the object
                    annotation = available_annotations.iloc[0]
                    class_value = annotation['Class']
                    parts[2] = str(class_value)  # Assuming the class_id is the third part in the line

                    # Mark this annotation as processed
                    processed_objects_in_frame[frame_id].append(annotation.name)

                    # Rebuild the modified line and add it to the output
                    if len(parts) == 8:
                        modified_line = ' '.join(parts[:-1])
                    elif len(parts) == 7:
                        modified_line = ' '.join(parts)
                    else:
                        asdadd
                    modified_lines.append(modified_line + '\n')
                    # if counter>0:
                    #     print(modified_line)
            # if counter==2:
            #     asdadsada

            # else:
            #     # If the frame is not in the CSV range, leave it unchanged
            #     modified_lines.append(line)

        # Filter out lines where class is 49 right before writing to the file
        modified_lines = [line for line in modified_lines if line.strip().split()[2] != '49']
        # Write the modified content to a new annotations.txt file
        print(*modified_lines)
        with open(new_txt_file_path, 'a') as file:
            file.writelines(modified_lines)

        print(f"Modified annotations saved to {new_txt_file_path}")
    else:
        print(f"File {txt_file_path} does not exist.")
