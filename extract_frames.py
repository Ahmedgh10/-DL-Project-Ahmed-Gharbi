import cv2
import os

def extract_and_downsample(video_path, hr_dir, lr_dir, target_lr_size=(960, 540)):
    # Create directories if they don't exist
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    print("Starting frame extraction... This might take a few minutes.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # 1. Save the original 1080p frame (HR)
        hr_filename = os.path.join(hr_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(hr_filename, frame)

        # 2. Downscale the frame to 540p (LR)
        # Using INTER_AREA is mathematically the best method for downsampling
        lr_frame = cv2.resize(frame, target_lr_size, interpolation=cv2.INTER_AREA)
        
        # 3. Save the downscaled 540p frame
        lr_filename = os.path.join(lr_dir, f"frame_{frame_count:05d}.png")
        cv2.imwrite(lr_filename, lr_frame)

        frame_count += 1
        
        if frame_count % 500 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"Done! Successfully extracted {frame_count} HR and LR frame pairs.")

# --- RUN THE SCRIPT ---
# Replace 'spiderman_gameplay.mp4' with the actual name of your video file
video_file = "Marvels Spider-Man 2 gameplay.mp4" 
extract_and_downsample(video_file, "dataset/HR", "dataset/LR")