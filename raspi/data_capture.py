import cv2
from picamera2 import Picamera2, Preview
from datetime import datetime
import os
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

def record_video(file_path="recorded_video.mp4", resolution=None, framerate=30):
    picam2 = Picamera2()
    
    # Use the camera's default maximum resolution if none is specified
    if resolution is None:
        camera_info = picam2.sensor_modes[0]
        resolution = (camera_info['size'][0], camera_info['size'][1])
    
    video_config = picam2.create_video_configuration(
        main={"size": resolution, "format": "BGR888"},
        controls={"FrameRate": framerate},
        display="main"
    )
    picam2.configure(video_config)
    
    try:
        picam2.start_recording(file_path)
        print(f"Recording started. Saving video to {file_path}")
        
        while True:
            # Keep recording until interrupted by user (Ctrl+C)
            picam2.wait_recording(1)  # Wait for 1 second intervals
        
    except KeyboardInterrupt:
        print("\n[INFO] Recording interrupted by user.")
        return False
    
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
    file = drive.CreateFile({'title': os.path.basename(file_path)})
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
    # Step 1: Record video until Ctrl+C is pressed
    file_name = f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    
    recording_success = record_video(file_path=file_name, framerate=30)
    
    if recording_success:
        # Step 2: Upload the recorded video to Google Drive and delete it locally
        upload_to_drive(file_name)
    else:
        print("[INFO] No video to upload since recording was interrupted.")
