# GeoLens2 - Advanced Privacy-Focused Geo-OSINT Analysis ![Status: Stable](https://img.shields.io/badge/Status-Stable-green.svg)

GeoLens2 is a powerful, privacy-centric Python tool designed to identify and highlight potential location-disclosing elements in images before they are shared online. Built from the ground up as an evolution of the original GeoLens, it leverages advanced AI models for image and text analysis, running entirely locally to ensure user privacy. Optimized for modern GPUs (e.g., RTX 3060Ti), GeoLens2 offers a modern, user-friendly GUI and supports multiple performance modes for flexibility.

![Sample](/Sample.png)

## Overview

GeoLens2 analyzes images to detect privacy risks such as street signs, license plates, and geographic indicators using a combination of CLIP for scene recognition, YOLO for object detection, and OCR for text extraction. It provides detailed risk assessments and generates masked images to visualize sensitive areas, helping users make informed decisions about sharing content.

**Key Features:**

- **Advanced Image Analysis:** Uses CLIP for scene and landmark recognition, supplemented by YOLO for precise object detection.
- **Text Extraction & Analysis:** Supports multiple OCR engines (EasyOCR, PaddleOCR, Tesseract) to extract and analyze text for location clues.
- **Privacy Risk Detection:** Identifies high-risk elements like street signs, license plates, and personal identifiers with bounding box precision.
- **Performance Modes:** Offers Fast, Balanced, and Precision modes to balance speed and accuracy based on hardware capabilities.
- **Masked Image Output:** Generates images with sensitive areas highlighted or masked in red for high-risk elements.
- **Local Processing:** All analysis runs locally, ensuring no data leaves the user's device.
- **Modern GUI:** Built with PySide6, featuring drag-and-drop functionality and a sleek, dark-themed interface.
- **Report Export:** Saves analysis results as JSON for documentation and review.

## Getting Started

GeoLens2 requires a Python environment and specific dependencies to run effectively. It is optimized for NVIDIA GPUs but supports CPU fallback.

### Prerequisites

- **Python 3.8 or higher**
- **NVIDIA GPU with CUDA (e.g., RTX 3060Ti):** Recommended for optimal performance; CPU fallback available but slower.
- **Operating System:** Windows, macOS, or Linux.
- **Optional:** Install OCR engines (Tesseract, EasyOCR, or PaddleOCR) and YOLO for enhanced functionality.

### Installation

1. **Clone the repository** (if using Git):
    ```bash
    git clone https://github.com/SilverHaze99/GeoLens
    cd GeoLens2
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    - **On macOS and Linux:**
        ```bash
        source venv/bin/activate
        ```

4. **Install required Python libraries:**
    ```bash
    pip install Pillow transformers torch torchvision torchaudio PySide6 opencv-python numpy
    ```

5. **Install optional dependencies for enhanced features:**
    - For OCR support (choose one or more):
        ```bash
        pip install easyocr  # Recommended for Balanced/Precision modes
        pip install paddleocr  # Alternative OCR engine
        pip install pytesseract  # Basic OCR fallback
        ```
    - For YOLO object detection:
        ```bash
        pip install ultralytics
        ```

6. **Create an `assets` folder** for icons (optional, if not already cloned):
    - Place SVG or PNG icons (e.g., `zap.svg`, `search.png`) in an `assets` folder in the project directory for GUI enhancements.

### Running the Tool

Activate the virtual environment and run the script:

```bash
python GeoLens2.py
```

The GUI will launch, allowing you to select a performance mode, drag-and-drop images, and analyze them for privacy risks.

## Usage

1. **Select Performance Mode:**
   - **Fast:** CPU-based, uses Tesseract OCR, lightweight but less accurate.
   - **Balanced:** Light GPU usage, uses EasyOCR and YOLO for better accuracy.
   - **Precision:** Full GPU power, uses large CLIP model and EasyOCR for maximum accuracy.
   - Click "Apply Mode" to initialize the analyzer.

2. **Analyze Images:**
   - Drag and drop an image into the GUI or click to browse.
   - Click "Analyze Privacy Risks" to start the analysis.
   - View results in the GUI, including a risk level indicator and detailed findings.

3. **Review Results:**
   - The tool highlights critical, moderate, and low risks with descriptions.
   - A masked image is generated, showing sensitive areas in red (high risk) or orange (potential risk).

4. **Save Outputs:**
   - Save the masked image to visualize privacy risks.
   - Export a JSON report for detailed analysis results.

## Models Used

GeoLens2 leverages the following AI models:

- **CLIP ([`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32) or `clip-vit-large-patch14`):** For scene and landmark recognition. Licensed under the [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE) by OpenAI.
- **YOLO (yolov8m.pt):** For precise object detection (e.g., signs, vehicles). Requires the `ultralytics` package.
- **OCR Engines:**
  - **EasyOCR:** Default for Balanced/Precision modes, supports multiple languages.
  - **PaddleOCR:** Alternative OCR engine for high accuracy.
  - **Tesseract:** Fallback for Fast mode, requires separate installation.

> **Copyright:**  
> CLIP model: Copyright (c) 2021 OpenAI, licensed under the MIT License.

## Performance Modes

GeoLens2 offers three performance modes to suit different hardware and accuracy needs:

- **Fast:** Uses CPU, Tesseract OCR, and a lightweight CLIP model. Ideal for low-end systems.
- **Balanced:** Uses GPU (if available), EasyOCR, YOLO, and a standard CLIP model for good accuracy and speed.
- **Precision:** Uses GPU, EasyOCR, YOLO, and a larger CLIP model for maximum accuracy, recommended for high-end GPUs.

## Privacy and Security

GeoLens2 runs all processing locally, ensuring no data is sent to external servers. It is designed to help users identify and mitigate unintentional location disclosures in images, making it a valuable tool for privacy-conscious individuals.

## Limitations

- Requires a capable GPU for optimal performance in Balanced/Precision modes.
- OCR accuracy depends on the chosen engine and image quality.
- YOLO object detection requires the `ultralytics` package and may download large model weights on first use.
- No internet connectivity is required, but some dependencies may need to be downloaded during installation.

## Contributing

Contributions are welcome! Please submit issues or pull requests to the [GitHub repository](https://github.com/SilverHaze99/GeoLens). Ensure you test changes in a virtual environment and follow the coding style in the provided script.

## License

GeoLens2 is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
