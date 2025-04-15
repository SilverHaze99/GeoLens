# GeoLens - Contextual Geo-OSINT Analysis ![Status: WIP](https://img.shields.io/badge/Status-WIP-orange.svg)

GeoLens is a local Python tool for identifying potential location clues in social media posts by analyzing images and text. Unlike traditional methods relying on EXIF data, GeoLens uses AI-powered image and text analysis (optimized for local execution on hardware like an RTX 3060Ti) to uncover subtle geographical indicators. It emphasizes user privacy by running all analysis locally and aims to raise awareness about unintentional location disclosures.

## Overview

GeoLens offers a simple graphical interface to analyze images and text. It uses a two-stage image recognition process with a pre-trained CLIP model to identify scenes and landmarks, alongside text analysis with spaCy to detect locations. The tool correlates image and text data to estimate locations and flags privacy risks.

**Key Features:**

- **Image Analysis:** Identifies objects and scenes to infer locations, even without EXIF data.
  - **Two-Stage Recognition:** General scene detection followed by specific landmark identification.
  - **Local AI Processing:** Uses your GPU (RTX 3060Ti recommended) for efficient, private analysis.
- **Text Analysis:** Extracts geographical locations from text, handling hashtags and varying cases.
- **Contextual Awareness:** Correlates image and text to estimate locations and highlight discrepancies.
- **Privacy-Focused:** Runs entirely locally within a virtual environment.
- **User-Friendly GUI:** Drag-and-drop interface for images and text input.
- **Warning System:** Flags privacy risks like street signs and location conflicts.

## Getting Started

Run GeoLens in a virtual environment to manage dependencies effectively.

### Prerequisites

- **Python 3.8 or higher**
- **NVIDIA GPU with CUDA (e.g., RTX 3060Ti):** Recommended for performance; CPU fallback possible but slower.

### Installation

1.  **Clone the repository** (if you are using Git):
    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```
4.  **Install Required Python Libraries:** Install the necessary libraries using pip within the activated virtual environment:

    ```bash
    pip install Pillow spacy transformers torch torchvision torchaudio tkinter tkinterdnd2
    python -m spacy download en_core_web_sm
    ```
5.  **Download the `landmarks.json` file:** This file contains the definitions of city landmarks used for specific image recognition. Ensure it is in the same directory as the Python script. You might need to create this file yourself with your desired locations and labels (see the "Data Structure" section for more details).

### Running the Tool

Ensure your virtual environment is still activated, then simply execute the Python script:

```bash
python geolens.py
