import os
import pandas as pd


# Create an output folder to save the corrected files
output_folder = './fruit-ripeness-classificator/output/corrected_annotations'
os.makedirs(output_folder, exist_ok=True)

# Function to load all annotation files in a folder recursively
def load_annotation_files(folder):
    annotation_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".txt"):
                annotation_files.append(os.path.join(root, file))
    return annotation_files

# Custom function to read annotations, handling cases with missing 'track_id'
def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 7:  # If track_id is missing
                frame, class_id, x, y, w, h, score = parts
                track_id = -1  # Default value for missing track_id
            elif len(parts) == 8:  # If track_id is present
                frame, track_id, class_id, x, y, w, h, score = parts
            else:
                raise ValueError(f"Unexpected number of columns in line: {line}")
            
            # Convert frame and class_id to int, handling float values like '49.0'
            frame = int(float(frame))
            class_id = int(float(class_id))
            track_id = track_id if track_id == -1 else int(float(track_id))  # Convert track_id only if present
            annotations.append([int(frame), track_id, class_id, float(x), float(y), float(w), float(h), float(score)])
    return pd.DataFrame(annotations, columns=['frame', 'track_id', 'class_id', 'x', 'y', 'w', 'h', 'score'])

for i in range(1,4):
    # Define the paths for both sets of annotations
    first_annotations_folder = f'./videos/naranja_videos{i}/annotated_frames_persisted'
    second_annotations_folder = f'./videos/naranja_videos{i}/annotated_frames_persisted_true'
    
    # Load the list of annotation files for both sets
    first_annotation_files = load_annotation_files(first_annotations_folder)
    print(first_annotation_files)
    second_annotation_files = load_annotation_files(second_annotations_folder)
    
    # Process each pair of annotation files (assuming the same structure and filenames)
    for first_file in first_annotation_files:
        # Find the corresponding second annotation file (based on filename)
        relative_path = os.path.relpath(first_file, first_annotations_folder)
        second_file = os.path.join(second_annotations_folder, relative_path)
        
        
        if not os.path.exists(second_file):
            print(f"Warning: Corresponding file not found for {first_file}")
            continue
    
        # Load the first set of annotations using the custom function
        first_annotations = read_annotations(first_file)
        first_annotations['frame'] = first_annotations['frame'].astype(int)
        first_annotations['class_id'] = first_annotations['class_id'].astype(int)
        
        # Load the second set of annotations
        second_annotations = read_annotations(second_file)
        second_annotations['frame'] = second_annotations['frame'].astype(int)
        second_annotations['class_id'] = second_annotations['class_id'].astype(int)
        print(second_annotations['track_id'].unique())

        first_annotations['score'] = first_annotations['score'].round(5)
        second_annotations['score'] = second_annotations['score'].round(5)

        # assert len(first_annotations) == len(second_annotations), "DataFrames are not of equal length."
        
        # Merge the two datasets to get the correct track_id
        merged_annotations = pd.merge(first_annotations.drop(columns=['track_id']), 
                                      second_annotations[['frame', 'track_id','score']], 
                                      on=['frame', 'score'], how='left')
        # merged_annotations = pd.merge(first_annotations.drop(columns=['track_id']), 
        #                       second_annotations[['track_id']], 
        #                       left_index=True, right_index=True, how='left')
        
        # Reorder columns to match the original structure
        merged_annotations = merged_annotations[['frame', 'track_id', 'class_id', 'x', 'y', 'w', 'h', 'score']]
        #print(merged_annotations.head(1))
        merged_annotations['track_id'].fillna(-1, inplace=True)
        
        # Save the corrected annotations to the output folder
        output_file = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        merged_annotations.to_csv(output_file, sep=' ', index=False, header=False)
        
        print(f"Corrected annotations saved for {first_file} as {output_file}")
