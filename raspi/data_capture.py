import cv2
from picamera2 import Picamera2, Preview
from picamera2.encoders import H264Encoder, Quality
from datetime import datetime
import time
import os
#from pydrive2.auth import GoogleAuth
#from pydrive2.drive import GoogleDrive

encoder = H264Encoder()

def record_video(file_path=None, resolution=None, framerate=30):
    picam2 = Picamera2()
    
    # Use the camera's default maximum resolution if none is specified
    if resolution is None:
        camera_info = picam2.sensor_modes[0]
        resolution = (camera_info['size'][0], camera_info['size'][1])
    
    video_config = picam2.create_video_configuration(
        main={"size":resolution, "format": "BGR888"},
        controls={"FrameRate": framerate},
        display="main"
    )
    picam2.configure(video_config)
    
    if file_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"video_{timestamp}.mp4"
    
    try:
        picam2.start_recording(encoder, file_path, quality=Quality.VERY_HIGH)
        print(f"Recording started. Saving video to {file_path}")
        counter = 0
        while True:
            # Keep recording until interrupted by user (Ctrl+C)
            #picam2.wait_recording(1)  # Wait for 1 second intervals
            print(counter,"\n")
            counter += 1
            time.sleep(1)
        
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user.")
    except Exception as e:
        print(e)
    
    finally:
        picam2.stop_recording()
        picam2.close()
        print(f"Recording finished. Video saved to {file_path}")
        return True

def upload_to_drive(file_path):
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile("credentials.json")  # Load the saved credentials

    if not gauth.credentials or gauth.access_token_expired:
        print("[INFO] Credentials not found or expired, refreshing.")
        gauth.LocalWebserverAuth()  # Authenticate if needed
        gauth.SaveCredentialsFile("credentials.json")  # Save the new credentials

    drive = GoogleDrive(gauth)
    file_metadata = {'title': os.path.basename(file_path)}
    file_metadata['parents'] = [{"id": folder_id}]
    file = drive.CreateFile(file_metadata)
    file.SetContentFile(file_path)
    file.Upload()
    print(f"Uploaded {file_path} to Google Drive")

    # Delete the file after upload
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted local file: {file_path}")
    else:
        print(f"File not found: {file_path}")

if __name__ == "__main__":
    folder_id = '1gGmNlZXm_09bzUwqOuw2vP9VN3yXMr9D'
    
    # Step 1: Record video until Ctrl+C is pressed
    
    
    local_save_directory = "/home/pi/videos"
    os.makedirs(local_save_directory, exist_ok=True)
    
    file_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h264"
    file_path = os.path.join(local_save_directory, file_name)
    
    recording_success = record_video(file_path=file_path, framerate=30)
    
    if recording_success:
        # Step 2: Upload the recorded video to Google Drive and delete it locally
        #upload_to_drive(file_name)
        print("Sucessfull recording")
    else:
        print("[INFO] No video to upload since recording was interrupted.")
