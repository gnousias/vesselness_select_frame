import os
import numpy as np
import pydicom
import streamlit as st
from skimage import io, color, exposure
from skimage.metrics import structural_similarity as ssim
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from vesselness2D import vesselness2D

# Parameters
VESSEL_PEAK_HEIGHT = 0.2  # fraction of max vesselness to consider a peak
VESSEL_PEAK_DISTANCE = 3  # minimal distance between peaks

# --- Helper function to process frames ---
def process_frames(image_data, sigmas=range(2, 6), tau=1.0, brightondark=False):
    num_frames = image_data.shape[0]
    vesselness_means = []

    for i in range(num_frames):
        frame = image_data[i]
        if frame.ndim == 3:
            frame = color.rgb2gray(frame)

        vessel = vesselness2D(frame, sigmas, spacing=(1, 1), tau=tau, brightondark=brightondark)
        vesselness_means.append(np.mean(vessel))

    vesselness_means = np.array(vesselness_means)
    norm_curve = vesselness_means / (vesselness_means.max() if vesselness_means.max() > 0 else 1)
    peaks, _ = find_peaks(norm_curve, height=VESSEL_PEAK_HEIGHT, distance=VESSEL_PEAK_DISTANCE)
    return peaks, norm_curve


# --- Streamlit UI ---
st.title("DICOM Frame Selection Tool (Vesselness Peak Detection)")

uploaded_files = st.file_uploader("Upload DICOM files", type=["dcm"], accept_multiple_files=True)

output_folder = st.text_input("Output folder", "./output")
os.makedirs(output_folder, exist_ok=True)

if uploaded_files:
    for dicom_file in uploaded_files:
        try:
            ds = pydicom.dcmread(dicom_file)
            image_data = ds.pixel_array

            # Normalize frame dimension ordering
            if image_data.ndim == 3:
                if image_data.shape[0] < image_data.shape[1]:
                    num_frames = image_data.shape[0]
                else:
                    image_data = np.transpose(image_data, (2, 0, 1))
                    num_frames = image_data.shape[0]
            elif image_data.ndim == 4:
                image_data = np.transpose(image_data, (3, 0, 1, 2))
                num_frames = image_data.shape[0]
            else:
                num_frames = 1
                image_data = np.expand_dims(image_data, axis=0)

            if num_frames < 10:
                st.warning(f"Skipping {dicom_file.name}: not enough frames ({num_frames})")
                continue

            st.write(f"Processing {dicom_file.name} with {num_frames} frames...")

            # Vesselness peak detection
            peaks, norm_curve = process_frames(image_data, sigmas=range(2, 6))

            if len(peaks) == 0:
                st.warning(f"No vesselness peaks detected for {dicom_file.name}")
                continue

            # Pick the peak with minimal SSIM to first frame
            reference_frame = image_data[0]
            ssim_vals = [ssim(image_data[p], reference_frame) for p in peaks]
            best_idx = peaks[np.argmin(ssim_vals)]
            selected_frame = image_data[best_idx]

            # Rescale and save
            vessel_rescaled = exposure.rescale_intensity(selected_frame, out_range=(0, 255)).astype(np.uint8)
            acquis_time = getattr(ds, 'AcquisitionTime', 'unknown')
            output_filename = f"{acquis_time}_{dicom_file.name}_Selected_Frame{best_idx}.png"
            output_path = os.path.join(output_folder, output_filename)
            io.imsave(output_path, vessel_rescaled)

            # Display results
            st.image(vessel_rescaled, caption=f"Selected Frame (Peak {best_idx})", use_container_width=True)

            fig, ax = plt.subplots()
            ax.plot(norm_curve, label="Vesselness Curve")
            ax.plot(peaks, norm_curve[peaks], "x", label="Peaks")
            ax.axvline(best_idx, color="r", linestyle="--", label="Selected Frame")
            ax.legend()
            st.pyplot(fig)

            st.success(f"Saved selected frame: {output_path}")

        except Exception as e:
            st.error(f"Error processing {dicom_file.name}: {e}")
