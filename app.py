import streamlit as st
import tempfile
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# Load YOLOv8 model (custom)
@st.cache_resource
def load_model():
    model = YOLO(r"best.pt")  
    return model


def detect_image(image, model, original_filename="detected_image"):
    results = model(image)
    annotated_image = results[0].plot()

    # Convert back to RGB before saving to match original image colors
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Save image to output folder
    output_path = os.path.join("output", f"{original_filename}_detected.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))  # Convert to BGR for saving

    return annotated_image, results[0], output_path



def detect_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)

    # Set output filename based on input name
    base_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join("output", f"{base_name}_detected.mp4")

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    cap.release()
    out.release()
    return output_path


# Streamlit UI
st.title("Personal Protective Equipment (PPE)Detection")
st.sidebar.header("Choose Input Type")
input_type = st.sidebar.radio("Select input type", ["Image", "Video"])

model = load_model()

if input_type == "Image":
    
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="Original Image", use_container_width=True)

        st.subheader("Detection Results:")
        filename = os.path.splitext(uploaded_image.name)[0]
        annotated_image, results, image_output_path = detect_image(image, model, original_filename=filename)
        st.image(annotated_image, caption="Detected Image", use_container_width=True)

        st.write("Detected Objects:")
        st.dataframe(results.boxes.data.cpu().numpy())

        with open(image_output_path, "rb") as f:
            st.download_button("Download Detected Image", f, file_name=os.path.basename(image_output_path))


elif input_type == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        video_bytes = uploaded_video.getvalue()
        st.video(video_bytes, format=uploaded_video.type or "video/mp4")

        with st.spinner("Running object detection on video..."):
            video_suffix = os.path.splitext(uploaded_video.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=video_suffix) as temp_input:
                temp_input.write(video_bytes)
                video_path = temp_input.name

            output_video_path = detect_video(video_path, model)
            with open(output_video_path, "rb") as output_video:
                processed_video_bytes = output_video.read()
            st.download_button(
                "Download Processed Video",
                processed_video_bytes,
                file_name=os.path.basename(output_video_path),
            )


        st.success("Done! Here's the processed video:")
        st.video(processed_video_bytes, format="video/mp4")


