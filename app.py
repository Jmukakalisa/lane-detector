
import streamlit as st
import torch
import numpy as np
from PIL import Image
from laneDetection import ENet, LaneDetector 

@st.cache(allow_output_mutation=True)
def load_model():
    enet_model = ENet(2, 4)  
    enet_model.load_state_dict(torch.load('lane_detection_model.pth', map_location=torch.device('cpu')))
    enet_model.eval()
    lane_detector = LaneDetector(enet_model)  
    return lane_detector

# Streamlit UI
def main():
    st.title("Lane Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = np.array(image)

        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image
        if st.button('Detect Lanes'):
            lane_detector = load_model()
            try:
                processed_image, lanes = lane_detector(image)
                st.image(processed_image, caption='Processed Image', use_column_width=True)
                st.write(f"Detected {len(lanes)} lanes")
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()