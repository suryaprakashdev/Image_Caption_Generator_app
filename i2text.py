import streamlit as st
import os
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
import pandas as pd
import io
import zipfile
from typing import List, Tuple

# Page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to avoid reloading on every run
@st.cache_resource
def load_model():
    """Load and cache the pre-trained model and processors"""
    try:
        model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        return model, feature_extractor, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def generate_caption(image: Image.Image, model, feature_extractor, tokenizer, device, max_length: int = 16) -> str:
    """Generate caption for a single image"""
    try:
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Process image
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)
        
        # Generate caption
        gen_kwargs = {"max_length": max_length, "num_beams": 4, "early_stopping": True}
        outputs = model.generate(pixel_values, **gen_kwargs)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def process_multiple_images(uploaded_files: List, model, feature_extractor, tokenizer, device, max_length: int) -> List[Tuple[str, str]]:
    """Process multiple uploaded images"""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}... ({i+1}/{len(uploaded_files)})")
            
            # Open and process image
            image = Image.open(uploaded_file)
            caption = generate_caption(image, model, feature_extractor, tokenizer, device, max_length)
            results.append((uploaded_file.name, caption))
            
        except Exception as e:
            results.append((uploaded_file.name, f"Error: {str(e)}"))
    
    progress_bar.empty()
    status_text.empty()
    return results

def main():
    # Title and description
    st.title("🖼️ AI Image Caption Generator")
    st.markdown("""
    Generate descriptive captions for your images using a pre-trained Vision Transformer (ViT) + GPT-2 model.
    Upload single or multiple images to get AI-generated captions.
    """)
    
    # Load model
    with st.spinner("Loading AI model... This may take a moment on first run."):
        model, feature_extractor, tokenizer, device = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please refresh the page and try again.")
        return
    
    # Model info
    st.success(f"✅ Model loaded successfully! Running on: {device}")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Caption length slider
    st.sidebar.markdown("### Caption Length")
    st.sidebar.info("""
    **ViT-GPT2 Model Specifications:**
    - **Minimum length:** 5 tokens
    - **Maximum length:** 50 tokens (recommended)
    - **Optimal range:** 10-25 tokens
    - **Note:** Longer captions may become repetitive
    """)
    
    max_length = st.sidebar.slider(
        "Maximum caption length (in tokens)",
        min_value=5,
        max_value=50,
        value=16,
        step=1,
        help="Number of tokens (words/subwords) in the generated caption"
    )
    
    # Input method selection
    st.sidebar.markdown("### Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Single Image Upload", "Multiple Images Upload", "Camera Capture"],
        help="Select how you want to provide images"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    if input_method == "Single Image Upload":
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
                help="Supported formats: PNG, JPG, JPEG, BMP, GIF"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)
                
                # Generate caption button
                if st.button("🔮 Generate Caption", type="primary"):
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(image, model, feature_extractor, tokenizer, device, max_length)
                    
                    with col2:
                        st.subheader("📝 Generated Caption")
                        st.success(caption)
                        
                        # Download option
                        caption_data = pd.DataFrame([[uploaded_file.name, caption]], 
                                                  columns=["Image", "Caption"])
                        csv_data = caption_data.to_csv(index=False)
                        st.download_button(
                            "💾 Download as CSV",
                            csv_data,
                            file_name="image_caption.csv",
                            mime="text/csv"
                        )
    
    elif input_method == "Multiple Images Upload":
        with col1:
            st.subheader("📤 Upload Multiple Images")
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
                accept_multiple_files=True,
                help="You can upload multiple images at once"
            )
            
            if uploaded_files:
                st.write(f"📊 **{len(uploaded_files)} images uploaded**")
                
                # Show thumbnails
                if st.checkbox("🖼️ Show image previews"):
                    cols = st.columns(3)
                    for i, uploaded_file in enumerate(uploaded_files[:9]):  # Show max 9 previews
                        with cols[i % 3]:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=uploaded_file.name, use_column_width=True)
                    
                    if len(uploaded_files) > 9:
                        st.info(f"Showing first 9 images. Total: {len(uploaded_files)} images")
                
                # Process all images button
                if st.button("🔮 Generate All Captions", type="primary"):
                    with st.spinner("Processing all images..."):
                        results = process_multiple_images(uploaded_files, model, feature_extractor, 
                                                        tokenizer, device, max_length)
                    
                    with col2:
                        st.subheader("📝 Generated Captions")
                        
                        # Display results
                        df = pd.DataFrame(results, columns=["Image", "Caption"])
                        st.dataframe(df, use_container_width=True)
                        
                        # Download options
                        csv_data = df.to_csv(index=False)
                        st.download_button(
                            "💾 Download as CSV",
                            csv_data,
                            file_name="batch_image_captions.csv",
                            mime="text/csv"
                        )
                        
                        # Success message
                        st.success(f"✅ Successfully generated captions for {len(results)} images!")
    
    elif input_method == "Camera Capture":
        with col1:
            st.subheader("📷 Camera Capture")
            camera_image = st.camera_input("Take a picture")
            
            if camera_image is not None:
                # Display captured image
                image = Image.open(camera_image)
                st.image(image, caption="Captured Image", use_column_width=True)
                
                # Generate caption button
                if st.button("🔮 Generate Caption", type="primary"):
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(image, model, feature_extractor, tokenizer, device, max_length)
                    
                    with col2:
                        st.subheader("📝 Generated Caption")
                        st.success(caption)
                        
                        # Download option
                        caption_data = pd.DataFrame([["camera_capture.jpg", caption]], 
                                                  columns=["Image", "Caption"])
                        csv_data = caption_data.to_csv(index=False)
                        st.download_button(
                            "💾 Download as CSV",
                            csv_data,
                            file_name="camera_caption.csv",
                            mime="text/csv"
                        )
    
    # Footer with additional information
    st.markdown("---")
    with st.expander("ℹ️ About this Application"):
        st.markdown("""
        ### Model Information
        - **Model:** nlpconnect/vit-gpt2-image-captioning
        - **Architecture:** Vision Transformer (ViT) + GPT-2
        - **Capabilities:** Generates natural language descriptions of images
        
        ### Technical Details
        - **Input:** RGB images (automatically converted if needed)
        - **Output:** Natural language captions
        - **Token Length:** Each token represents roughly 0.75 words
        - **Processing:** GPU acceleration when available
        
        ### Tips for Best Results
        - Use clear, well-lit images
        - Avoid heavily distorted or blurry images
        - Optimal caption length: 10-25 tokens
        - Model works best with common objects and scenes
        """)

if __name__ == "__main__":
    main()