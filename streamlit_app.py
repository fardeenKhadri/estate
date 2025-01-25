import streamlit as st
import os
import numpy as np
from PIL import Image
from argparse import Namespace
import torch
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath("src"))

from config.defaults import get_config
from utils.logger import get_logger
from inference import preprocess, run_one_inference
from models.build import build_model


# Function to load the model for inference
def load_inference_model():
    logger = get_logger()
    # Get the configuration for inference
    config = get_config()
    config.defrost()
    config.MODE = 'inference'
    config.MODEL.NAME = 'LGT_Net'
    config.MODEL.ARGS = [{
        "decoder_name": "Transformer",  # Options: 'Transformer', 'LSTM'
        "output_name": "LGT",          # Options: 'LGT', 'LED', 'Horizon'
        "backbone": "resnet50",        # Backbone model
        "dropout": 0.1,                # Dropout rate
        "win_size": 8,                 # Transformer window size
        "depth": 6                     # Transformer depth
    }]
    config.freeze()

    model = build_model(config, logger)
    model.eval()
    return model, logger, config


# Streamlit interface
st.title("LGT-Net: Room Layout Prediction")
st.markdown("""
This app predicts the 3D room layout of an RGB panorama using the Geometry-Aware Transformer Network.
""")

# File uploader for input panorama
uploaded_file = st.file_uploader("Upload an RGB panorama image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("Processing the image...")

    # Load the model
    model, logger, config = load_inference_model()

    # Preprocess the image
    img_name = "uploaded_image"
    img = np.array(img.resize((1024, 512), Image.Resampling.BICUBIC))[..., :3]
    vp_cache_path = os.path.join('src/output', f'{img_name}_vp.txt')

    img, vp = preprocess(img, vp_cache_path=vp_cache_path)
    img = (img / 255.0).astype(np.float32)

    # Run inference
    args = Namespace(output_dir="src/output")
    run_one_inference(img, model, args, img_name, logger=logger, show=False)
    st.success("Inference Complete!")

    # Display the predicted result
    result_img_path = os.path.join('src/output', f"{img_name}_pred.png")
    if os.path.exists(result_img_path):
        st.image(result_img_path, caption="Predicted Result", use_column_width=True)

    # Download button for the predicted image
    st.markdown("Download Outputs:")
    with open(result_img_path, "rb") as file:
        st.download_button(label="Download Predicted Image", data=file, file_name="predicted_result.png")

if __name__ == "__main__":
    import os
    os.system("streamlit run streamlit_app.py --server.address 192.168.10.1 --server.port 5110")
