import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from pipeline import UNet

# ---------- Page configuration ----------
st.set_page_config(page_title="CityScapes Segmentation", layout="wide")

# ---------- Load model (cached) ----------
@st.cache_resource
def load_model():
    from pipeline import UNet   # if not imported globally
    model = UNet(in_channels=3, out_channels=23)
    model.load_state_dict(torch.load('unet_cityscapes.pth', map_location='cuda'))
    model.eval()
    return model

model = load_model()

# ---------- Helper functions ----------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def predict_mask(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).numpy()
    return pred

def visualize_prediction(original_image, true_mask, pred_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(true_mask, cmap='tab20', vmin=0, vmax=22)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=22)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.tight_layout()
    return fig

# ---------- Page 1: Training Metrics ----------
def page1():
    st.title("📊 Training Metrics")
    st.markdown("### Loss Curve, mIoU, and mDice during Training")

    # Load the saved plots (assumes you saved them as images)
    if os.path.exists("training_plots.png"):
        st.image("training_plots.png", caption="Training curves")
    else:
        st.warning("Plot image not found. Please run training first.")

    # Display final test metrics
    st.markdown("### Final Test Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean IoU (mIoU)", "0.523")   # replace with your actual value
    with col2:
        st.metric("Mean Dice (mDice)", "0.571") # replace with your actual value

# ---------- Page 2: Upload and Segment ----------
def page2():
    st.title("🖼️ Segment Your Own Images")
    st.markdown("Upload up to 4 images from the test set to see ground‑truth and predicted masks.")

    uploaded_files = st.file_uploader("Choose images", type=['png', 'jpg', 'jpeg'],
                                      accept_multiple_files=True)

    if uploaded_files:
        # Limit to 4 images
        uploaded_files = uploaded_files[:4]
        for i, file in enumerate(uploaded_files):
            st.markdown(f"### Image {i+1}")
            col1, col2, col3 = st.columns(3)
            with col1:
                original = Image.open(file).convert('RGB')
                st.image(original, caption="Original", use_column_width=True)
            with col2:
                # For ground truth, we need to load the corresponding mask file.
                # This assumes the mask file has the same name as the image.
                # Adjust the path logic as needed.
                mask_filename = file.name
                mask_path = os.path.join("CameraMask", mask_filename)
                if os.path.exists(mask_path):
                    true_mask = np.array(Image.open(mask_path))
                    st.image(true_mask, caption="Ground Truth", use_column_width=True)
                else:
                    st.warning("Ground truth mask not found.")
                    true_mask = None
            with col3:
                # Predict using the model
                tensor = preprocess_image(original)
                pred_mask = predict_mask(tensor)
                st.image(pred_mask, caption="Prediction", use_column_width=True)

# ---------- Navigation ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Training Metrics", "Segment Images"])

if page == "Training Metrics":
    page1()
else:
    page2()