import streamlit as st
import cv2
import numpy as np
from PIL import Image
import yaml
from ocr_utils import run_ic_ocr
from matcher import load_oem_db, match_text

# Load config
with open("config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

oem_db = load_oem_db(config["paths"]["oem_db"])
threshold = config["matching"]["threshold"]

st.title("🔍 AOI-Based IC Marking Verifier (CRAFT OCR)")
st.markdown("Upload an IC image to verify against OEM database")

uploaded_file = st.file_uploader("Upload IC image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    st.image(image, caption="Uploaded IC Image", use_column_width=True)

    ocr_text = run_ic_ocr(
        img_array,
        craft_model_path=config["ocr"]["craft_model_path"],
        whitelist=config["ocr"]["whitelist"]
    )

    st.subheader("Extracted Marking Text")
    st.code(ocr_text if ocr_text else "No text detected")

    result = match_text(ocr_text, oem_db, threshold)

    st.subheader("Verification Result")
    if result["match_found"]:
        st.success(f"✅ Match Found: {result['name']} ({result['score']:.2f}%)")
        st.write(f"**Manufacturer:** {result['manufacturer']}")
        st.write(f"**Description:** {result['description']}")
    else:
        st.error(f"❌ No good match found (best score: {result['score']:.2f}%)")
