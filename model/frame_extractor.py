import cv2
import os

for i in range(3,4):
    # Paths to your annotations and videos
    annotation_folder = f"./fruit-ripeness-classificator/output/new_class_annotationsv2/naranja_videos{i}/"
    video_folder = f"./videos/naranja_videos{i}/raw_frames/"
    output_frame_folder = "./fruit-ripeness-classificator/dataset/images/train/"
    
    # Make sure the output folder exists
    os.makedirs(output_frame_folder, exist_ok=True)
    
    # Loop through each annotation file
    for annotation_file in os.listdir(annotation_folder):
        annotation_path = os.path.join(annotation_folder, annotation_file)
        
        # Read the corresponding video
        video_filename = annotation_file.replace('_modified.txt', '.h264')  # Assuming videos are .mp4, change as needed
        video_path = os.path.join(video_folder, video_filename)
        
    
        if not os.path.exists(video_path):
            print(f"Video {video_path} not found for annotation {annotation_file}")
            continue
        
        cap = cv2.VideoCapture(video_path)
    
        frames_to_extract = []
        
        with open(annotation_path, 'r') as f:
            #counter = 0
            for line in f.readlines():
                # Parse the annotation file to get frame numbers (modify based on your format)
                parts = line.strip().split()
                frame_number = int(parts[0])  # Assuming the annotation contains frame numbers
                frames_to_extract.append(frame_number)
    
                # Set the video to the specific frame
                #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                # # Read the frame
                # ret, frame = cap.read()
                # if frame_number == counter:
                #     # Save the frame as an image file
                #     frame_filename = f"frame_{frame_number}.jpg"
                #     output_path = os.path.join(output_frame_folder, video_filename, frame_filename)
                #     cv2.imwrite(output_path, frame)
                # else:
                #     print(f"Failed to capture frame {frame_number} from {video_filename}")
                # counter += 1
            counter = 0
            while cap.isOpened():
                success, im0 = cap.read()
                #print(success)
                if not success:
                    print("Video frame is empty or video processing has been successfully completed.", counter)
                    break
                if counter in frames_to_extract:
                    # Display the frame for debugging (press any key to close the window)
                    #cv2.imshow(f"Frame {counter}", im0)
                    #cv2.waitKey(100)  # Adjust the delay as needed, press any key to continue
    
                    frame_filename = f"frame_{str(counter)}.jpg"
                    output_path = output_frame_folder +  video_filename.split('.')[0] + "_" + frame_filename
                    write_success = cv2.imwrite(output_path, im0)
                    if write_success:
                        print(f"Saved frame {counter} to {output_path}")
                    else:
                        print(f"Failed to save frame {counter} to {output_path}")
                counter += 1
    
        cap.release()
    
print("Frame extraction complete.")
