import os
import numpy as np
import pydicom
from skimage import io, color, measure, exposure, morphology
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from vesselness2D import vesselness2D  # Python version we converted

# Paths
main_file_path = "/workspaces/Frame_Selection_Methods/input"
output_folder = "/workspaces/Frame_Selection_Methods/output"
os.makedirs(output_folder, exist_ok=True)

# Adaptive threshold for vesselness peaks
VESSEL_PEAK_HEIGHT = 0.2  # fraction of max vesselness to consider a peak
VESSEL_PEAK_DISTANCE = 3  # minimal distance between peaks

# Helper: process all frames and detect peaks
def process_frames(image_data, sigmas=range(2, 6), tau=1.0, brightondark=False):
    num_frames = image_data.shape[0]
    vesselness_means = []

    for i in range(num_frames):
        frame = image_data[i]
        if frame.ndim == 3:
            frame = color.rgb2gray(frame)

        vessel = vesselness2D(frame, sigmas, spacing=(1, 1), tau=tau, brightondark=brightondark)
        vesselness_means.append(np.mean(vessel))
        print(f"Frame {i+1}/{num_frames}: vesselness mean={np.mean(vessel):.4f}, max={np.max(vessel):.4f}")

    vesselness_means = np.array(vesselness_means)
    norm_curve = vesselness_means / (vesselness_means.max() if vesselness_means.max() > 0 else 1)

    peaks, _ = find_peaks(norm_curve, height=VESSEL_PEAK_HEIGHT, distance=VESSEL_PEAK_DISTANCE)
    return peaks, norm_curve

# Main loop
for sub_filename in os.listdir(main_file_path):
    sub_fullpath = os.path.join(main_file_path, sub_filename)
    year = sub_filename.split('_')[0]
    if not os.path.isdir(sub_fullpath):
        continue

    for patient_name in os.listdir(sub_fullpath):
        code = ''.join([c for c in patient_name if c.isdigit()])
        ID_number = str(code)
        dicom_video_path = os.path.join(sub_fullpath, patient_name, "IMAGE")
        if not os.path.exists(dicom_video_path):
            continue

        for dicom_file in os.listdir(dicom_video_path):
            full_path = os.path.join(dicom_video_path, dicom_file)
            if os.path.isdir(full_path):
                continue

            try:
                ds = pydicom.dcmread(full_path)
                image_data = ds.pixel_array
                # Convert to (num_frames, height, width)
                if image_data.ndim == 3:
                    if image_data.shape[0] < image_data.shape[1]:
                        # already (frames, h, w)
                        num_frames = image_data.shape[0]
                    else:
                        # maybe (h, w, frames)
                        image_data = np.transpose(image_data, (2, 0, 1))
                        num_frames = image_data.shape[0]
                elif image_data.ndim == 4:
                    # (h, w, channels, frames)
                    image_data = np.transpose(image_data, (3, 0, 1))
                    num_frames = image_data.shape[0]
                else:
                    num_frames = 1
                    image_data = np.expand_dims(image_data, axis=0)

                if num_frames < 10:
                    print(f"Skipping {full_path}: not enough frames ({num_frames})")
                    continue

                print(f"Processing {full_path} with {num_frames} frames")

                # Vesselness peak detection
                peaks, norm_curve = process_frames(image_data, sigmas=range(2, 6))
                if len(peaks) == 0:
                    print(f"No vesselness peaks detected for {full_path}")
                    continue

                # Pick the peak with minimal SSIM to first frame
                reference_frame = image_data[0]
                ssim_vals = [ssim(image_data[p], reference_frame) for p in peaks]
                best_idx = peaks[np.argmin(ssim_vals)]
                selected_frame = image_data[best_idx]

                # Save
                vessel_rescaled = exposure.rescale_intensity(selected_frame, out_range=(0, 255))
                vessel_rescaled = vessel_rescaled.astype(np.uint8)

                acquis_time = getattr(ds, 'AcquisitionTime', 'unknown')
                output_filename = f"{year}_{ID_number}_{acquis_time}_{dicom_file}_Selected_Frame{best_idx}_vessel.png"
                output_path = os.path.join(output_folder, output_filename)
                io.imsave(output_path, vessel_rescaled)
                print(f"Saved selected frame: {output_path}")

            except Exception as e:
                print(f"Error processing {full_path}: {e}")
