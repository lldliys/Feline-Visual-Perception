import cv2
import numpy as np
import math
import os

# ================= CONFIGURATION =================
INPUT_PATH = "data/Butterfly.mp4"
OUTPUT_DIR = "output/"

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= CORE ALGORITHM MODULES =================

def apply_color_transformation(frame, matrix):
    """
    Simulates feline dichromatic vision by applying a color-space 
    transformation matrix derived from feline opsin sensitivity.
    """
    # Convert OpenCV BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Scale pixel values to [0, 1] for matrix multiplication
    rgb = frame_rgb.astype(np.float32) / 255.0
    h, w, _ = rgb.shape
    
    # Flatten image for efficient matrix operation
    rgb_flat = rgb.reshape(-1, 3)
    rgb_cvd_flat = rgb_flat @ matrix.T
    
    # Clip values to ensure they remain within [0, 1] and reshape back
    rgb_cvd = np.clip(rgb_cvd_flat.reshape(h, w, 3), 0, 1)
    
    # Convert back to uint8 and return to BGR for OpenCV compatibility
    rgb_cvd_uint8 = (rgb_cvd * 255).astype(np.uint8)
    return cv2.cvtColor(rgb_cvd_uint8, cv2.COLOR_RGB2BGR)

def create_radial_mask(height, width, center_power=2.5):
    """
    Generates a radial gradient mask to simulate feline visual acuity, 
    where central vision is clearer than the periphery.
    """
    Y, X = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    max_dist = np.sqrt(center_x**2 + center_y**2)
    
    # Normalize distance and apply power function for radial falloff
    normalized_dist = dist_from_center / max_dist
    mask = 1.0 - np.power(normalized_dist, center_power)
    mask = np.clip(mask, 0, 1)
    
    # Stack into 3 channels to match image shape
    return np.dstack([mask] * 3).astype(np.float32)

def apply_spatial_blur(frame, mask, sigma_fovea, sigma_periphery):
    """
    Applies differential spatial resolution (blur) based on the radial mask.
    Foveal region uses light blur, while periphery uses heavy blur.
    """
    frame_float = frame.astype(np.float32) / 255.0
    
    # Generate central vision layer (light blur/foveal acuity)
    if sigma_fovea > 0.1:
        center_view = cv2.GaussianBlur(frame_float, (0, 0), sigmaX=sigma_fovea, sigmaY=sigma_fovea)
    else:
        center_view = frame_float
        
    # Generate peripheral vision layer (heavy blur)
    periphery_view = cv2.GaussianBlur(frame_float, (0, 0), sigmaX=sigma_periphery, sigmaY=sigma_periphery)
    
    # Blend layers using the radial mask
    final_frame = (center_view * mask) + (periphery_view * (1.0 - mask))
    return (np.clip(final_frame, 0, 1) * 255).astype(np.uint8)

def apply_fisheye_distortion(frame):
    """
    Simulates the panoramic field of view (~200 degrees) using 
    fisheye camera distortion coefficients.
    """
    h, w = frame.shape[:2]
    # Define virtual camera matrix K and distortion coefficients D
    # Parameters adjusted to emulate feline wide-angle perspective
    K = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32)
    D = np.array([-0.3, 0.1, 0, 0], dtype=np.float32)
    
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    return cv2.undistort(frame, K, D, None, new_K)

# ================= MAIN PROCESSING PIPELINE =================

def run_simulation():
    """
    Executes the full feline vision simulation pipeline on the input video.
    Steps: Color Shift -> Spatial Blur -> FOV Distortion -> Temporal Interleaving.
    """
    # 1. Initialize video capture
    cap = cv2.VideoCapture(INPUT_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open input video at {INPUT_PATH}")
        return

    # Retrieve video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 2. Define biological and environmental parameters
    # CVD Matrix derived from feline spectral sensitivity (dichromatic)
    CVD_MATRIX = np.array([
        [0.152286, 1.052583, -0.204868], 
        [0.114503, 0.786281, 0.099216], 
        [-0.003882, -0.048116, 1.051998]
    ])
    
    # Calculate blur sigmas based on physical screen and viewing distance
    CLARITY_FACTOR = 0.2
    PPD = 60 # Estimated Pixels Per Degree
    SIGMA_FOVEA = 0.5 * PPD * CLARITY_FACTOR
    SIGMA_PERIPHERY = 1.5 * PPD * CLARITY_FACTOR
    
    # 3. Pre-generate reusable components
    radial_mask = create_radial_mask(h, w)
    black_frame = np.zeros((h, w, 3), dtype=np.uint8) # For temporal interleaving

    # 4. Initialize video writer (Final output)
    # Note: Output FPS is doubled due to black-frame interleaving
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(OUTPUT_DIR, "feline_view_final.mp4")
    out = cv2.VideoWriter(out_path, fourcc, fps * 2, (w, h))

    print(f"Starting feline vision processing pipeline...")
    print(f"Input: {INPUT_PATH} | Output: {out_path}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        # Step A: Color Transformation
        processed = apply_color_transformation(frame, CVD_MATRIX)
        
        # Step B: Spatial Resolution Adjustment
        processed = apply_spatial_blur(processed, radial_mask, SIGMA_FOVEA, SIGMA_PERIPHERY)
        
        # Step C: Field of View (FOV) Distortion
        processed = apply_fisheye_distortion(processed)
        
        # Step D: Temporal Resolution Simulation 
        # Insert black frames to simulate flicker fusion perception
        out.write(processed)
        out.write(black_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frames processed: {frame_count}...", end='\r')

    # Cleanup
    cap.release()
    out.release()
    print(f"\nProcessing complete! Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_simulation()