from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import sys
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *
import cv2
import numpy as np
import re
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum
import threading
from datetime import datetime
import json
import os

# Optional imports based on performance mode
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Nach den Imports, vor "class PerformanceMode(Enum):"
class IconManager:
    """Manages SVG and PNG icons for the application"""
    
    @staticmethod
    def get_icon(icon_name: str, size: int = 16) -> QIcon:
        """Load icon from assets folder"""
        icon_path = os.path.join("assets", f"{icon_name}.svg")
        png_path = os.path.join("assets", f"{icon_name}.png")
        
        # Try SVG first, then PNG, then fallback
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        elif os.path.exists(png_path):
            return QIcon(png_path)
        else:
            # Create empty fallback icon
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.GlobalColor.transparent)
            return QIcon(pixmap)
    
    @staticmethod
    def get_icon_html(icon_name: str, size: int = 16) -> str:
        """Get HTML representation of icon for rich text"""
        icon_path = os.path.join("assets", f"{icon_name}.svg")
        png_path = os.path.join("assets", f"{icon_name}.png")
        
        if os.path.exists(icon_path):
            return f'<img src="{icon_path}" width="{size}" height="{size}" style="vertical-align: middle;">'
        elif os.path.exists(png_path):
            return f'<img src="{png_path}" width="{size}" height="{size}" style="vertical-align: middle;">'
        else:
            return "🔍"  # Emoji fallback

class PerformanceMode(Enum):
    FAST = "fast"
    BALANCED = "balanced"
    PRECISION = "precision"

@dataclass
class RiskDetection:
    category: str
    risk_level: int
    description: str
    bbox: Tuple[int, int, int, int] = None

class AnalysisWorker(QThread):
    finished = Signal(list, object)
    progress = Signal(str)
    
    def __init__(self, analyzer, image_path):
        super().__init__()
        self.analyzer = analyzer
        self.image_path = image_path
    
    def run(self):
        try:
            loading_text = IconManager.get_icon_html("loading", 16) + " Loading models..."
            self.progress.emit(loading_text)
            risks, masked_image = self.analyzer.analyze_image_risks(self.image_path)
            self.finished.emit(risks, masked_image)
        except Exception as e:
            error_text = IconManager.get_icon_html("error", 16) + f" Analysis failed: {e}"
            self.finished.emit([RiskDetection("error", 1, error_text)], None)

class MultiImageAnalysisWorker(QThread):
    finished = Signal(list)
    progress = Signal(str)
    
    def __init__(self, analyzer, image_paths):
        super().__init__()
        self.analyzer = analyzer
        self.image_paths = image_paths
    
    def run(self):
        try:
            results = self.analyzer.analyze_multiple_images(self.image_paths)
            self.finished.emit(results)
        except Exception as e:
            self.progress.emit(f"Batch analysis failed: {e}")
            self.finished.emit([])

class PrivacyRiskAnalyzer:
    def __init__(self, performance_mode: PerformanceMode = PerformanceMode.BALANCED):
        self.performance_mode = performance_mode
        self.device = self._get_device()
        self.models = {}
        self.config = self._get_config_for_mode()
        
        self.RISK_PATTERNS = {
            "street_infrastructure": {
                "patterns": [
                    "street sign with text", "road sign with location", "traffic sign with city name",
                    "highway sign with destination", "street name plate", "address number plate"
                ],
                "risk_level": 4,
                "description": "Street signage can reveal exact location"
            },
            "personal_identifiers": {
                "patterns": [
                    "license plate on vehicle", "car registration plate", "number plate with region code",
                    "name tag", "ID card", "badge with name"
                ],
                "risk_level": 5,
                "description": "Personal identifiers directly reveal information"
            },
            "geographic_indicators": {
                "patterns": [
                    "palm trees in urban setting", "snow-capped mountains", "desert landscape",
                    "tropical vegetation", "deciduous forest", "coastal scenery"
                ],
                "risk_level": 2,
                "description": "Geographic features suggest region/climate"
            },
            "cultural_elements": {
                "patterns": [
                    "traditional architecture", "religious building", "cultural monument",
                    "local cuisine", "regional clothing", "ethnic signage"
                ],
                "risk_level": 3,
                "description": "Cultural elements indicate specific regions"
            },
            "public_transport": {
                "patterns": [
                    "bus with route number", "train with destination", "subway sign",
                    "bus stop sign", "metro station"
                ],
                "risk_level": 3,
                "description": "Public transport shows specific routes/areas"
            },
            "business_signage": {
                "patterns": [
                    "store sign with text", "restaurant sign", "shop name", 
                    "business logo", "storefront with name"
                ],
                "risk_level": 3,
                "description": "Business names can be searched/located online"
            }
        }
        
        self.TEXT_RISK_PATTERNS = {
            "location_indicators": {
                "patterns": [
                    r'\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Lane|Ln\.?|Boulevard|Blvd\.?|Drive|Dr\.?)\b',
                    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*(?:Germany|France|Italy|Spain|UK|USA|Canada|Australia)\b',
                    r'#(?:Berlin|Paris|London|NewYork|LA|Munich|Hamburg|Cologne|Frankfurt|Vienna|Madrid)\b',
                    r'\b(?:DE-|FR-|IT-|ES-)[A-Z]{1,3}\s*\d{1,4}\b',
                    r'\b\w+straße\b|\b\w+gasse\b|\bRue\s+\w+\b|\bVia\s+\w+\b',
                    r'\b[A-Z][a-zA-Z-äöüß]+(?:-[A-Z][a-zA-Z-äöüß]+)*\b',
                    r'\b\d{3}\s*-\s*\d{3}\b'
                ],
                "risk_level": 4,
                "description": "Location-specific text detected"
            },
            "personal_info": {
                "patterns": [
                    r'\b\d{4,5}\s*[A-Z]{2}\b',
                    r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{3,4}\b',
                    r'\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b',
                    r'\b[A-Za-z\s]+\s+\d{1,5}\b',
                ],
                "risk_level": 3,
                "description": "Personal/sensitive information detected"
            }
        }

    def _get_device(self):
        if self.performance_mode == PerformanceMode.FAST:
            return "cpu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _get_config_for_mode(self):
        configs = {
            PerformanceMode.FAST: {
                "clip_model": "openai/clip-vit-base-patch32",
                "clip_threshold": 0.25,
                "ocr_engine": "tesseract",
                "use_object_detection": False,
                "image_size": (224, 224)
            },
            PerformanceMode.BALANCED: {
                "clip_model": "openai/clip-vit-base-patch32", 
                "clip_threshold": 0.15,
                "ocr_engine": "easyocr",
                "use_object_detection": True,
                "image_size": (224, 224)
            },
            PerformanceMode.PRECISION: {
                "clip_model": "openai/clip-vit-large-patch14",
                "clip_threshold": 0.12,
                "ocr_engine": "easyocr",
                "use_object_detection": True,
                "image_size": (336, 336)
            }
        }
        return configs[self.performance_mode]
    
    def _get_ocr_engine(self):
        preferred_engine = self.config["ocr_engine"]
        
        if preferred_engine == "paddleocr" and PADDLEOCR_AVAILABLE:
            if not hasattr(self, '_paddle_ocr'):
                try:
                    # Try new PaddleOCR API first
                    self._paddle_ocr = paddleocr.PaddleOCR(
                        use_textline_orientation=True,
                        lang='en',
                        show_log=False
                    )
                    return "paddle"
                except Exception as e:
                    try:
                        # Fallback to older API
                        self._paddle_ocr = paddleocr.PaddleOCR(
                            use_angle_cls=True,
                            lang='en',
                            show_log=False
                        )
                        return "paddle"
                    except Exception as e2:
                        print(f"PaddleOCR initialization failed: {e2}")
            
            if hasattr(self, '_paddle_ocr'):
                return "paddle"
        
        if preferred_engine == "easyocr" and EASYOCR_AVAILABLE:
            if not hasattr(self, '_easy_ocr'):
                self._easy_ocr = easyocr.Reader(['en', 'de', 'fr'], gpu=self.device == "cuda")
            return "easy"
        
        if TESSERACT_AVAILABLE:
            return "tesseract"
        
        return None
    
    def _load_clip_model(self):
        """Load CLIP model with caching"""
        model_name = self.config["clip_model"]
        
        if model_name not in self.models:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            model = model.to(self.device)
            
            self.models[model_name] = {
                "model": model,
                "processor": processor
            }
        
        return self.models[model_name]["model"], self.models[model_name]["processor"]

    def _detect_objects_with_yolo(self, image_path: str) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Use YOLO to detect objects and get bounding boxes"""
        if not YOLO_AVAILABLE or not self.config["use_object_detection"]:
            return []
        
        try:
            # Load YOLO model (will download first time)
            if not hasattr(self, '_yolo_model'):
                self._yolo_model = YOLO('yolov8m.pt')  # Medium version for better accuracy
            
            # Load original image size for scaling
            original_image = Image.open(image_path)
            original_size = original_image.size
            
            # Run inference
            results = self._yolo_model(image_path, verbose=False)
            detections = []
            
            # YOLO class names that could indicate privacy risks
            risky_classes = {
                'stop sign': 4, 'traffic light': 9, 'car': 3, 'truck': 7, 'bus': 5,
                'person': 1, 'bicycle': 1, 'motorcycle': 3
            }
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = self._yolo_model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Scale bounding box to match original image size
                        scale_x = original_size[0] / result.orig_shape[1]
                        scale_y = original_size[1] / result.orig_shape[0]
                        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        # Only include objects that might be privacy-relevant
                        if (class_name in risky_classes and confidence > 0.3) or 'sign' in class_name.lower():
                            detections.append((class_name, (x1, y1, x2, y2), confidence))
            
            return detections
            
        except Exception as e:
            print(f"YOLO detection failed: {e}")
            return []

    def _perform_ocr(self, image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Perform OCR using best available engine"""
        engine = self._get_ocr_engine()
        results = []
        
        if engine == "paddle":
            try:
                ocr_results = self._paddle_ocr.ocr(image, cls=True)
                for line in ocr_results[0] or []:
                    bbox_points = np.array(line[0])
                    x1, y1 = bbox_points.min(axis=0).astype(int)
                    x2, y2 = bbox_points.max(axis=0).astype(int)
                    text = line[1][0]
                    confidence = line[1][1]
                    results.append((text, (x1, y1, x2, y2), confidence))
            except Exception as e:
                print(f"PaddleOCR failed: {e}")
                return []
                
        elif engine == "easy":
            try:
                ocr_results = self._easy_ocr.readtext(image)
                for (bbox, text, confidence) in ocr_results:
                    points = np.array(bbox)
                    x1, y1 = points.min(axis=0).astype(int)
                    x2, y2 = points.max(axis=0).astype(int)
                    results.append((text, (x1, y1, x2, y2), confidence))
            except Exception as e:
                print(f"EasyOCR failed: {e}")
                return []
                
        elif engine == "tesseract":
            try:
                import pytesseract
                text = pytesseract.image_to_string(image)
                if text.strip():
                    # Fake bbox for entire image
                    h, w = image.shape[:2]
                    results.append((text, (0, 0, w, h), 0.8))
            except Exception as e:
                print(f"Tesseract failed: {e}")
                return []
        
        return results

    def analyze_image_risks(self, image_path: str) -> Tuple[List[RiskDetection], np.ndarray]:
        """Analyze image for privacy risks and return detections + masked image"""
        risks = []
        
        try:
            image = Image.open(image_path).convert("RGB")

            # Pre-check for blank/monochromatic images
            image_np = np.array(image)
            # Check if the standard deviation of pixel values is very low
            if image_np.std() < 5: #Threshold
                masked_image = self._create_masked_image(image_path, [])
                return [RiskDetection("info", 1, "Image is blank or monochromatic, no risks detected.")], masked_image


            original_size = image.size
            target_size = self.config["image_size"]
            image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        except Exception as e:
            return [RiskDetection("error", 1, f"Image load error: {e}")], None
        
        # Load CLIP model
        model, processor = self._load_clip_model()
        
        # Prepare all risk pattern labels
        all_labels = []
        label_to_category = {}

        no_risk_labels = ["a blank image", "a solid color background", "a blurry photo with no details", "abstract shapes"]
        all_labels.extend(no_risk_labels)
        for label in no_risk_labels:
            label_to_category[label] ="no_risk"


        for category, data in self.RISK_PATTERNS.items():
            for pattern in data["patterns"]:
                all_labels.append(pattern)
                label_to_category[pattern] = category
        
        # CLIP analysis
        inputs = processor(text=all_labels, images=image_resized, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs.to(self.device))
        
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

        highest_prob_index = np.argmax(probs)
        highest_prob_label = all_labels[highest_prob_index]

        if label_to_category[highest_prob_label] == "no_risk":
            masked_image = self._create_masked_image(image_path, [])
            return [RiskDetection ("info", 1, "Image content does not suggest any privacy risks.")], masked_image
        
        # Filter detections based on mode-specific threshold
        threshold = self.config["clip_threshold"]
        for label, prob in zip(all_labels, probs):
            if prob > threshold:
                category = label_to_category[label]
                risk_data = self.RISK_PATTERNS[category]
                
                risk = RiskDetection(
                    category=category,
                    risk_level=risk_data["risk_level"],
                    description=f"{risk_data['description']} (Confidence: {prob:.1%})",
                    bbox=None  # CLIP doesn't give bounding boxes
                )
                risks.append(risk)
        
        # YOLO Object Detection for precise masking
        yolo_detections = []
        if self.config["use_object_detection"]:
            yolo_detections = self._detect_objects_with_yolo(image_path)
            
            # Convert YOLO detections to risks with bounding boxes
            for obj_name, bbox, confidence in yolo_detections:
                # Map YOLO classes to risk categories
                if 'sign' in obj_name.lower() or obj_name == 'stop sign':
                    risks.append(RiskDetection(
                        category="detected_signage",
                        risk_level=4,
                        description=f"Detected {obj_name} (YOLO: {confidence:.1%})",
                        bbox=bbox
                    ))
                elif obj_name in ['car', 'truck', 'bus', 'motorcycle']:
                    risks.append(RiskDetection(
                        category="vehicle_with_plates", 
                        risk_level=3,
                        description=f"Vehicle detected: {obj_name} (potential license plate)",
                        bbox=bbox
                    ))
        
        # OCR Analysis
        if self._get_ocr_engine():
            try:
                ocr_results = self._perform_ocr(np.array(image))
                
                for text, bbox, confidence in ocr_results:
                    if confidence > 0.5:  # Only high-confidence text
                        # Scale bbox to original image size
                        x1, y1, x2, y2 = bbox
                        if original_size != image.size:
                            scale_x = original_size[0] / image.size[0]
                            scale_y = original_size[1] / image.size[1]
                            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                        
                        scaled_bbox = (x1, y1, x2, y2)
                        
                        # Check OCR text against patterns
                        matched = False
                        for category, data in self.TEXT_RISK_PATTERNS.items():
                            for pattern in data["patterns"]:
                                matches = re.finditer(pattern, text, re.IGNORECASE)
                                for match in matches:
                                    risk = RiskDetection(
                                        category=f"text_{category}",
                                        risk_level=data["risk_level"],
                                        description=f"{data['description']}: '{match.group()}'",
                                        bbox=scaled_bbox
                                    )
                                    risks.append(risk)
                                    matched = True
                        if not matched:
                            # Potential risk for unrecognized text

                            street_suffixes = [
                                r'straße\b', r'gasse\b', r'weg\b', r'platz\b', r'allee\b', r'ufer\b',  # Deutsch
                                r'street\b', r'road\b', r'avenue\b', r'lane\b', r'drive\b', r'court\b', # Englisch
                                r'rue\b', r'boulevard\b', # Französisch
                                r'via\b', r'piazza\b' # Italienisch
                            ]
                            street_pattern = r'.*(?:' + '|'.join(street_suffixes) + ')'

                            if re.search(street_pattern, text, re.IGNORECASE) and len(text) > 5:
                                risk = RiskDetection(
                                    category="text_street_name",
                                    risk_level=4,
                                    description=f"Detected lilkey street name: '{text}'",
                                    bbox=scaled_bbox
                                )
                                risks.append(risk)

                            elif len(text) > 4 and any(c.isalpha() for c in text) and any(isdigit() for c in text):
                                risk = RiskDetection(
                                    category="text_potenial_identifier",
                                    risk_level=3,
                                    description=f"Potential identifying text (letter & numbers): '{text}'",
                                    bbox=scaled_bbox
                                )
                                risks.append(risk)

                            elif (len(text) > 4 and (text.isupper() or (text[0].isupper() and text[1:].islower()))):
                                risk = RiskDetection(
                                    category="text_potential_proper_noun",
                                    risk_level=2,  # Geringeres Risiko, da dies auch ein normales Wort am Satzanfang sein könnte
                                    description=f"Potential proper noun / location name: '{text}'",
                                    bbox=scaled_bbox
                                )
                                risks.append(risk)

            except Exception as e:
                risks.append(RiskDetection("ocr_error", 1, f"OCR analysis failed: {e}"))
        else:
            risks.append(RiskDetection("ocr_warning", 1, "No OCR engine available - install tesseract, easyocr, or paddleocr"))
        
        # Create masked image
        masked_image = self._create_masked_image(image_path, risks)
        
        # Clean up GPU memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
            
        return risks, masked_image

    def _create_masked_image(self, image_path: str, risks: List[RiskDetection]) -> np.ndarray:
        """Create image with privacy risks masked in red with precise object masking"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        overlay = image.copy()
        has_precise_masks = False
        
        # Mask detected objects with bounding boxes (YOLO + OCR)
        for risk in risks:
            if risk.bbox and risk.risk_level >= 3:
                x1, y1, x2, y2 = risk.bbox
                
                # Choose color based on risk level
                if risk.risk_level >= 4:
                    color = (0, 0, 255)  # Red for high risk
                else:
                    color = (0, 165, 255)  # Orange for potential risk
                
                # Draw semi-transparent rectangle over the object
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Add warning label above the box
                label = risk.category.replace('_', ' ').replace('text ', '').upper()
                label_y = max(y1 - 10, 15)
                cv2.putText(overlay, f"⚠️ {label}", (x1, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Add border around the masked area
                cv2.rectangle(overlay, (x1-2, y1-2), (x2+2, y2+2), color, 3)
                
                has_precise_masks = True
        
        # For CLIP risks without bounding boxes, add corner indicators
        clip_only_risks = [r for r in risks if not r.bbox and r.risk_level >= 3]
        if clip_only_risks and not has_precise_masks:
            # Add corner warnings since we can't mask precisely
            h, w = image.shape[:2]
            
            # Top corners with warning triangles
            triangle_size = 50
            triangle = np.array([[0, 0], [triangle_size, 0], [0, triangle_size]], np.int32)
            cv2.fillPoly(overlay, [triangle], (0, 0, 255))  # Top-left
            
            triangle_tr = np.array([[w, 0], [w-triangle_size, 0], [w, triangle_size]], np.int32)  
            cv2.fillPoly(overlay, [triangle_tr], (0, 0, 255))  # Top-right
            
            # Warning text
            cv2.putText(overlay, "⚠️ PRIVACY RISKS - LOCATION UNCLEAR", 
                       (triangle_size + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            has_precise_masks = True
        
        if has_precise_masks:
            # Blend overlay with original image
            alpha = 0.6  # More opaque for better visibility
            masked = cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)
        else:
            # No risks found - add green checkmark
            masked = image.copy()
            cv2.putText(masked, "✅ LOW PRIVACY RISK", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)
        
        return masked

    def calculate_overall_risk(self, risks: List[RiskDetection]) -> Tuple[int, str]:
        """Calculate overall privacy risk score"""
        if not risks:
            return 0, "No privacy risks detected"
        
        max_risk = max(risk.risk_level for risk in risks)
        risk_count = len([r for r in risks if r.risk_level >= 3])
        
        if max_risk >= 4 or risk_count >= 3:
            return 5, "🔴 CRITICAL: High location disclosure risk!"
        elif max_risk >= 3 or risk_count >= 2:
            return 3, "🟡 MODERATE: Some location indicators present"
        elif max_risk >= 2:
            return 2, "🟢 LOW: Minor location clues detected"
        else:
            return 1, "✅ MINIMAL: Very low privacy risk"

    def analyze_multiple_images(self, image_paths: List[str]) -> List[Tuple[List[RiskDetection], np.ndarray]]:
        """Analyze multiple images in sequence and return results"""
        results = []
        for path in image_paths:
            self.progress.emit(f"Analyzing {path.split('/')[-1]}...")
            risks, masked_image = self.analyze_image_risks(path)
            results.append((risks, masked_image))
        return results

    def export_analysis_report(self, risks: List[RiskDetection], filename: str):
        """Export analysis results to JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "risks": [
                {
                    "category": r.category,
                    "risk_level": r.risk_level,
                    "description": r.description,
                    "bbox": r.bbox if r.bbox else None
                } for r in risks
            ]
        }
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Failed to export report: {e}")
            return False

    def cleanup_models(self):
        """Clean up loaded models to free memory"""
        for model_data in self.models.values():
            del model_data["model"]
            del model_data["processor"]
        self.models.clear()
        
        if hasattr(self, '_yolo_model'):
            del self._yolo_model
        if hasattr(self, '_easy_ocr'):
            del self._easy_ocr
        if hasattr(self, '_paddle_ocr'):
            del self._paddle_ocr
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class ImageDropWidget(QLabel):
    """Custom widget for drag & drop functionality"""
    imageDropped = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                background-color: #1e1e1e;
                color: #666;
                font-size: 14px;
            }
            QLabel:hover {
                border-color: 00bcd4;
                background-color: #2e2e2e;
            }
        """)
        self.setText("Drag & Drop image here or click to browse")
        self.setMinimumHeight(100)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #00bcd4;
                    border-radius: 10px;
                    padding: 20px;
                    background-color: #1e1e1e;
                    color: #00bcd4;
                    font-size: 14px;
                }
            """)
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                background-color: #1e1e1e;
                color: #2e2e2e;
                font-size: 14px;
            }
        """)
    
    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files and any(files[0].lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
            self.imageDropped.emit(files[0])
        self.dragLeaveEvent(event)
    
    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", 
            "Image Files (*.jpg *.jpeg *.png *.bmp *.gif)"
        )
        if file_path:
            self.imageDropped.emit(file_path)

class RiskLevelWidget(QFrame):
    """Custom widget to display risk level with color coding"""
    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
        """)
        
        layout = QHBoxLayout(self)
        self.icon_label = QLabel()
        self.icon_label.setFixedSize(30, 30)
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label = QLabel("No analysis performed yet")
        self.text_label.setFont(QFont("", 12, QFont.Weight.Bold))
        
        layout.addWidget(self.icon_label)
        layout.addWidget(self.text_label)
        layout.addStretch()
    
    def set_risk_level(self, level: int, message: str):
        colors = {
            0: ("#3e3e3e", "#ff0000", "❓"),
            1: ("#3e3e3e", "#0bbd17", "✅"), 
            2: ("#3e3e3e", "#c9c012", "🟡"),
            3: ("#3e3e3e", "#c9c012", "🟡"),
            4: ("#3e3e3e", "#f20017", "🔴"),
            5: ("#3e3e3e", "#f20017", "🔴")
        }
        
        bg_color, text_color, icon = colors.get(level, colors[0])
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {text_color};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        
        self.icon_label.setText(icon)
        self.icon_label.setFont(QFont("", 16))
        self.text_label.setText(message)
        self.text_label.setStyleSheet(f"color: {text_color};")

class GeoLensPrivacyGuard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.current_image_path = None
        self.current_masked_image = None
        self.current_risks = None
        self.analysis_worker = None
        
        self.setWindowTitle("🛡️ GeoLens Privacy Guard")
        self.setMinimumSize(900, 800)
        self.setup_ui()
        self.apply_modern_style()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header_layout = QVBoxLayout()
        title_label = QLabel("GeoLens Privacy Guard")
        title_label.setFont(QFont("", 18, QFont.Weight.Bold))
        subtitle_label = QLabel("Analyze images for location privacy risks before sharing")
        subtitle_label.setStyleSheet("color: #00bcd4; font-size: 12px;")
        
        header_layout.addWidget(title_label)
        header_layout.addWidget(subtitle_label)
        main_layout.addLayout(header_layout)
        
        # Performance Mode Selection
        perf_group = QGroupBox("Performance Mode")
        perf_layout = QVBoxLayout(perf_group)
        
        self.mode_buttons = QButtonGroup()
        self.fast_radio = QRadioButton("Fast: CPU-based, lightweight (Tesseract OCR)")
        self.fast_radio.setIcon(IconManager.get_icon("zap",16))
        self.balanced_radio = QRadioButton("Balanced: Light GPU usage, good accuracy (EasyOCR)")
        self.balanced_radio.setIcon(IconManager.get_icon("scale",16))
        self.precision_radio = QRadioButton("Precision: Full GPU power, best accuracy (EasyOCR + Large CLIP)")
        self.precision_radio.setIcon(IconManager.get_icon("crosshair",16))
        
        self.balanced_radio.setChecked(True)  # Default
        
        self.mode_buttons.addButton(self.fast_radio, 0)
        self.mode_buttons.addButton(self.balanced_radio, 1)
        self.mode_buttons.addButton(self.precision_radio, 2)
        
        perf_layout.addWidget(self.fast_radio)
        perf_layout.addWidget(self.balanced_radio)
        perf_layout.addWidget(self.precision_radio)
        
        apply_button = QPushButton("Apply Mode")
        apply_button.setIcon(IconManager.get_icon("refresh",16))
        apply_button.clicked.connect(self.update_analyzer)
        perf_layout.addWidget(apply_button)
        
        main_layout.addWidget(perf_group)
        
        # Status Bar
        self.status_label = QLabel("Select performance mode and click 'Apply Mode'")
        self.status_label.setStyleSheet("""
            QLabel {
                border: 1px solid #00bcd4;
                padding: 5px;
                background-color: #1e1e1e;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                background-color: #121212;
            }
            QProgressBar::chunk {
                background-color: #00bcd4;
                border-radius: 3px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Image Drop Area
        self.drop_widget = ImageDropWidget()
        self.drop_widget.imageDropped.connect(self.on_image_selected)
        main_layout.addWidget(self.drop_widget)
        
        # Analysis Button
        self.analyze_button = QPushButton("Analyze Privacy Risks")
        self.analyze_button.setIcon(IconManager.get_icon("search", 16))
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self.analyze_image)
        self.analyze_button.setStyleSheet("""
            QPushButton {
                background-color: #1e1e1e;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00bcd4;
            }
            QPushButton:disabled {
                background-color: #3e3e3e;
                color: #666;
            }
        """)
        main_layout.addWidget(self.analyze_button)
        
        # Risk Level Display
        self.risk_widget = RiskLevelWidget()
        main_layout.addWidget(self.risk_widget)
        
        # Results Area
        results_group = QGroupBox("Privacy Risk Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setFont(QFont("Consolas", 10))
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        # Action Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Masked Image")
        self.save_button.setIcon(IconManager.get_icon("save",16))
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save_masked_image)
        
        self.export_button = QPushButton("Export Report")
        self.export_button.setIcon(IconManager.get_icon("document",16))
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_report)
        
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.setIcon(IconManager.get_icon("trash",16))
        self.clear_button.clicked.connect(self.clear_results)
        
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.clear_button)
        button_layout.addStretch()
        
        results_layout.addLayout(button_layout)
        main_layout.addWidget(results_group)
    
    def apply_modern_style(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #00bcd4;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #1e1e1e;
                border: 1px solid #00bcd4;
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2e2e2e;
            }
            QPushButton:pressed {
                background-color: #1e1e1e;
            }
            QRadioButton {
                spacing: 8px;
                font-size: 12px;
            }
            QTextEdit {
                border: 1px solid #00bcd4;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
    
    def update_analyzer(self):
        mode_map = {0: PerformanceMode.FAST, 1: PerformanceMode.BALANCED, 2: PerformanceMode.PRECISION}
        selected_mode = mode_map[self.mode_buttons.checkedId()]
        
        self.analyzer = PrivacyRiskAnalyzer(selected_mode)
        self.status_label.setText(f"Mode: {selected_mode.value.upper()} | Device: {self.analyzer.device.upper()}")
        
        if self.current_image_path:
            self.analyze_button.setEnabled(True)
    
    def on_image_selected(self, file_path: str):
        self.current_image_path = file_path
        filename = file_path.split('/')[-1] if '/' in file_path else file_path.split('\\')[-1]
        self.drop_widget.setText(f"Selected: {filename}")
        
        if self.analyzer:
            self.analyze_button.setEnabled(True)
    
    def analyze_image(self):
        if not self.current_image_path or not self.analyzer:
            return
        
        # Disable UI during analysis
        self.analyze_button.setEnabled(False)
        self.results_text.clear()
        self.results_text.append("Analyzing image for privacy risks...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        mode_name = ["FAST", "BALANCED", "PRECISION"][self.mode_buttons.checkedId()]
        self.results_text.append(f"Using {mode_name} mode with {self.analyzer.device.upper()} processing...\n")
        
        self.analysis_worker = AnalysisWorker(self.analyzer, self.current_image_path)
        self.analysis_worker.progress.connect(self.on_analysis_progress)
        self.analysis_worker.finished.connect(self.on_analysis_finished)
        self.analysis_worker.start()
    
    def on_analysis_progress(self, message: str):
        self.results_text.append(message)
    
    def on_analysis_finished(self, risks: List[RiskDetection], masked_image):
        self.current_masked_image = masked_image
        self.current_risks = risks
        
        self.progress_bar.setVisible(False)
        
        # Check for errors in risks
        errors = [r for r in risks if r.category.startswith("error") or r.category.endswith("_error")]
        if errors:
            self.results_text.clear()
            self.results_text.append("❌ Analysis Errors:")
            for error in errors:
                self.results_text.append(f"  • {error.description}")
            self.risk_widget.set_risk_level(0, "Analysis failed due to errors")
            self.analyze_button.setEnabled(True)
            return
        
        # Calculate overall risk
        overall_risk, risk_msg = self.analyzer.calculate_overall_risk(risks)
        self.risk_widget.set_risk_level(overall_risk, risk_msg)
        
        # Display detailed results
        self.results_text.clear()
        self.results_text.append(f"Analysis Complete - {risk_msg}")
        self.results_text.append("=" * 60 + "\n")
        
        if risks:
            # Group by risk level
            critical_risks = [r for r in risks if r.risk_level >= 4]
            moderate_risks = [r for r in risks if r.risk_level == 3]
            low_risks = [r for r in risks if r.risk_level <= 2]
            
            if critical_risks:
                self.results_text.append("CRITICAL RISKS:")
                for risk in critical_risks:
                    self.results_text.append(f"  • {risk.description}")
                self.results_text.append("")
            
            if moderate_risks:
                self.results_text.append("MODERATE RISKS:")
                for risk in moderate_risks:
                    self.results_text.append(f"  • {risk.description}")
                self.results_text.append("")
            
            if low_risks:
                self.results_text.append("LOW RISKS:")
                for risk in low_risks:
                    self.results_text.append(f"  • {risk.description}")
                self.results_text.append("")
            
            self.results_text.append("RECOMMENDATIONS:")
            if critical_risks or len(moderate_risks) >= 2:
                self.results_text.append("⚠️  Consider editing out sensitive elements before sharing")
                self.results_text.append("⚠️  Use the masked version to see what needs attention")
            else:
                self.results_text.append("✅ Image appears relatively safe for sharing")
            
            if self.current_masked_image is not None:
                self.save_button.setEnabled(True)
                self.export_button.setEnabled(True)
        else:
            self.results_text.append("✅ No significant privacy risks detected!")
            self.results_text.append("Image appears safe for sharing.")
        
        # Re-enable UI
        self.analyze_button.setEnabled(True)
    
    def save_masked_image(self):
        if self.current_masked_image is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Masked Image", "", 
            "JPEG Files (*.jpg);;PNG Files (*.png);;All Files (*)"
        )
        
        if file_path:
            cv2.imwrite(file_path, self.current_masked_image)
            self.results_text.append(f"\n💾 Masked image saved to: {file_path}")
    
    def export_report(self):
        if not self.current_risks:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis Report", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            success = self.analyzer.export_analysis_report(self.current_risks, file_path)
            if success:
                self.results_text.append(f"\n📄 Report saved to: {file_path}")
            else:
                self.results_text.append("\n❌ Failed to save report")
    
    def clear_results(self):
        self.results_text.clear()
        self.risk_widget.set_risk_level(0, "No analysis performed yet")
        self.current_masked_image = None
        self.current_risks = None
        self.save_button.setEnabled(False)
        self.export_button.setEnabled(False)
        if self.analyzer:
            self.analyzer.cleanup_models()

def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("GeoLens Privacy Guard")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("GeoLens")
    
    # Modern Windows 11 style
    app.setStyle('Fusion')
    
    window = GeoLensPrivacyGuard()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()