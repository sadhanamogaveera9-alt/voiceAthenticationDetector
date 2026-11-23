import streamlit as st
import librosa
import numpy as np
import tempfile
import os

st.set_page_config(
    page_title="Voice Authenticity Detector",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Voice Authenticity Detector")

st.markdown("""
### AI-Powered Voice Analysis

Upload an audio file to analyze basic audio signal characteristics.

This prototype demonstrates audio analysis and experimental
voice authenticity screening.
""")

st.info(
    "Experimental demo: This tool does not reliably determine "
    "whether a voice is real or AI-generated."
)

uploaded_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "mp3", "ogg", "flac"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    if st.button("Analyze Audio"):

        with st.spinner("Analyzing audio..."):

            temp_path = None

            try:
                # Save uploaded audio temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=os.path.splitext(uploaded_file.name)[1]
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                # Load audio
                audio, sr = librosa.load(
                    temp_path,
                    sr=22050,
                    mono=True
                )

                # Basic signal features
                duration = librosa.get_duration(
                    y=audio,
                    sr=sr
                )

                rms = librosa.feature.rms(
                    y=audio
                ).mean()

                zero_crossings = librosa.feature.zero_crossing_rate(
                    y=audio
                ).mean()

                spectral_centroid = librosa.feature.spectral_centroid(
                    y=audio,
                    sr=sr
                ).mean()

                mfcc = librosa.feature.mfcc(
                    y=audio,
                    sr=sr,
                    n_mfcc=13
                )

                mfcc_variation = np.std(mfcc)

                # Display results
                st.success("Audio analysis completed!")

                st.subheader("Analysis Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Duration",
                        f"{duration:.2f} seconds"
                    )

                    st.metric(
                        "RMS Energy",
                        f"{rms:.4f}"
                    )

                with col2:
                    st.metric(
                        "Zero Crossing Rate",
                        f"{zero_crossings:.4f}"
                    )

                    st.metric(
                        "Spectral Centroid",
                        f"{spectral_centroid:.2f} Hz"
                    )

                st.metric(
                    "MFCC Variation",
                    f"{mfcc_variation:.2f}"
                )

                st.subheader("Experimental Screening")

                # Demo rule, NOT a real voice classifier
                if duration < 2:
                    st.warning(
                        "Audio is very short. More audio is "
                        "needed for meaningful analysis."
                    )

                else:
                    st.warning(
                        "Audio signal features have been measured. "
                        "No reliable real-versus-AI classification "
                        "is available in this demo."
                    )

                st.markdown("---")

                st.subheader("Safety Recommendation")

                st.write(
                    "Do not rely on voice alone to verify a person. "
                    "Confirm important requests through a separate, "
                    "trusted communication channel."
                )

            except Exception as e:

                st.error(
                    f"Could not analyze audio: {e}"
                )

            finally:

                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

st.markdown("---")

st.caption(
    "Prototype for Build 2036 | "
    "Voice Authenticity Detector | Experimental demonstration"
)