import sys
import pickle
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd
import nltk


# Add the project root to sys.path so imports work seamlessly
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from database import Database
from imagecaption import ImageCaptioner
from models.LSTM import LSTM_Model


MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_FILES = {"BiLSTM": MODEL_DIR / "bilstm.pt"}

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multimodal Content Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Database & Model Caching
# ---------------------------------------------------------
@st.cache_resource
def get_database() -> Database:
    """Initializes and returns the database handler."""
    db_file = BASE_DIR / "DB" / "database.csv"
    return Database(db_path=str(db_file))


@st.cache_resource(show_spinner=False)
def load_captioner() -> ImageCaptioner:
    """Loads and caches the BLIP image captioning model."""
    captioner = ImageCaptioner()
    captioner.load_model()
    return captioner

@st.cache_resource(show_spinner=False)
def load_text_classifier(model_name: str) -> LSTM_Model:
    model_path = saved_models_dir / model_filename
    model_path = MODEL_FILES.get(model_name)
    if model_path is None:
        raise ValueError(f"Unsupported model: {model_name}")

    vocab_path = saved_models_dir / "vocab.pkl"
    label_encoder_path = saved_models_dir / "label_encoder.pkl"

    # Validate before instantiating or caching
    missing_files = [p for p in [model_path, vocab_path, label_encoder_path] if not p.exists()]
    if missing_files:
        missing_str = ", ".join([f"`{f.name}`" for f in missing_files])
        raise FileNotFoundError(f"Missing required model assets in `{saved_models_dir}`: {missing_str}")

    config_path = saved_models_dir / "config.pkl"
    max_len = 40
    embed_dim = 300
    hidden_dim = 128
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                cfg = pickle.load(f)
                max_len = cfg.get("max_len", max_len)
                embed_dim = cfg.get("embed_dim", 300)
                hidden_dim = cfg.get("hidden_dim", hidden_dim)
        except Exception:
            pass

    classifier = LSTM_Model(
        model_path=str(model_path),
        vocab_path=str(vocab_path),
        label_encoder_path=str(label_encoder_path),
        max_len=max_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    )
    classifier.load_assets()
    return classifier

@st.cache_resource
def prepare_nltk_resources() -> bool:
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
    }

    for resource, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(resource, quiet=True):
                raise RuntimeError(f"Unable to download NLTK resource: {resource}")

    return True



# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def determine_input_type(has_text: bool, has_image: bool) -> str:
    if has_text and has_image:
        return "Multimodal (Text + Image)"
    elif has_image:
        return "Image Only"
    elif has_text:
        return "Text Only"
    return "Unknown"


# ---------------------------------------------------------
# Main Application UI
# ---------------------------------------------------------
def main():
    db = get_database()
    prepare_nltk_resources()
    # --- Sidebar Configuration ---
    st.sidebar.title("⚙️ Settings")
    st.sidebar.markdown("Configure your inference parameters below.")

    # Model Dropdown Selection
    model_options = [name for name, path in MODEL_FILES.items() if path.exists()]
    if not model_options:
        st.error(f"No trained model files were found in `{MODEL_DIR}`.")
        st.stop()

    selected_model_name = st.sidebar.selectbox(
        "Choose Classification Model:",
        options=model_options,
        index=0,
        help="Select the model architecture used for classification.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Query History")
    history_df = db.get_history()
    st.sidebar.metric("Total Logged Queries", len(history_df))

    if st.sidebar.button("🗑️ Clear Database History"):
        db.clear_history()
        st.sidebar.success("Database history cleared!")
        st.rerun()

    # --- Header ---
    st.title("🛡️ Multimodal Content Classifier")
    st.markdown(
        """
        Analyze text, images, or multimodal inputs (both text and image) to detect 
        toxic or sensitive content. Upload an image, type text, or do both!
        """
    )
    st.markdown("---")

    # --- Tabs for Predict and History ---
    tab_predict, tab_history = st.tabs(["🚀 Predict", "📜 Database Logs"])

    with tab_predict:
        col_left, col_right = st.columns([1, 1], gap="large")

        with col_left:
            st.subheader("1. Provide Input")
            
            # Text Input
            user_text = st.text_area(
                "Enter Text:",
                placeholder="Type or paste your text query here...",
                height=130,
            )

            # Image Input
            uploaded_image = st.file_uploader(
                "Upload Image (optional):",
                type=["png", "jpg", "jpeg", "webp"],
                help="Upload an image to extract visual context via BLIP image captioning.",
            )

            image_obj = None
            if uploaded_image is not None:
                try:
                    image_obj = Image.open(uploaded_image).convert("RGB")
                    st.image(image_obj, caption="Uploaded Image Preview", use_container_width=True)
                except Exception as e:
                    st.error(f"Error opening image: {e}")

            # Predict Button
            predict_button = st.button("🔍 Run Prediction", type="primary", use_container_width=True)

        with col_right:
            st.subheader("2. Analysis & Output")

            if predict_button:
                has_text = bool(user_text and user_text.strip())
                has_image = image_obj is not None

                if not has_text and not has_image:
                    st.warning("⚠️ Please provide at least text input, an image, or both.")
                else:
                    input_type = determine_input_type(has_text, has_image)
                    generated_caption = ""
                    combined_text = ""

                    # --- Step 1: Image Captioning (if image provided) ---
                    if has_image:
                        with st.spinner("🤖 Generating image caption using BLIP..."):
                            try:
                                captioner = load_captioner()
                                generated_caption = captioner.generate_caption(image_obj)
                                st.info(f"**Generated Image Caption:** {generated_caption}")
                            except Exception as e:
                                st.error(f"Failed to generate image caption: {e}")
                                return

                    # --- Step 2: Combine Inputs ---
                    if input_type == "Multimodal (Text + Image)":
                        combined_text = f"{user_text.strip()} {generated_caption}".strip()
                    elif input_type == "Image Only":
                        combined_text = generated_caption
                    else:  # Text Only
                        combined_text = user_text.strip()

                    st.markdown(f"**Final Input to Classifier:** `{combined_text}`")

                    # --- Step 3: Classification ---
                    with st.spinner(f"⚡ Running inference with {selected_model_name}..."):
                        try:
                            classifier = load_text_classifier(selected_model_name)
                            
                            # Check if model has weights loaded
                            saved_model_file = MODEL_FILES[selected_model_name]
                            if not saved_model_file.exists():
                                st.error(
                                    f"⚠️ Model weights not found at `{saved_model_file}`. "
                                    "Please run `python train.py` first to generate model weights."
                                )
                                return

                            result = classifier.predict(combined_text)
                            predicted_label = result.get("label", "Unknown")
                            confidence = float(result.get("confidence", 0.0))
                            probabilities = result.get("probabilities", {})

                        except Exception as e:
                            st.error(f"Inference error: {e}")
                            return

                    # --- Step 4: Display Output ---
                    st.success("✅ Prediction Completed!")

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Predicted Label", str(predicted_label).capitalize())
                    with res_col2:
                        st.metric("Confidence", f"{confidence * 100:.2f}%")

                    # Probability distribution bar
                    st.progress(min(max(confidence, 0.0), 1.0))

                    if probabilities:
                        with st.expander("📊 View Class Probabilities"):
                            prob_df = pd.DataFrame(
                                list(probabilities.items()),
                                columns=["Class", "Probability"]
                            ).sort_values(by="Probability", ascending=False)
                            st.dataframe(
                                prob_df.style.format({"Probability": "{:.4f}"}),
                                use_container_width=True
                            )

                    # --- Step 5: Save Query to Database ---
                    try:
                        db.log_entry(
                            input_type=input_type,
                            user_text=user_text.strip() if user_text else "",
                            generated_caption=generated_caption,
                            combined_text=combined_text,
                            predicted_label=str(predicted_label),
                            confidence=confidence,
                            model_used=selected_model_name,
                        )
                        st.toast("💾 Query logged to database successfully!", icon="✅")
                    except Exception as e:
                        st.warning(f"Failed to log query to database: {e}")

            else:
                st.info("👈 Enter text, upload an image, or both, then click **Run Prediction**.")

    # --- Tab 2: Database Logs ---
    with tab_history:
        st.subheader("📋 Query Log History")
        updated_history = db.get_history()

        if updated_history.empty:
            st.info("No queries logged yet. Run a prediction to see records here.")
        else:
            col_search, col_dl = st.columns([3, 1])
            with col_search:
                search_term = st.text_input("Filter by keyword:", "")
            with col_dl:
                csv_data = updated_history.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download CSV",
                    data=csv_data,
                    file_name="query_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if search_term:
                filtered_df = updated_history[
                    updated_history.apply(
                        lambda row: row.astype(str).str.contains(search_term, case=False).any(),
                        axis=1
                    )
                ]
                st.dataframe(filtered_df, use_container_width=True)
            else:
                st.dataframe(updated_history, use_container_width=True)


if __name__ == "__main__":
    main()
