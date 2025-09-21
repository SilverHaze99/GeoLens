from PIL import Image
import spacy
from transformers import CLIPProcessor, CLIPModel
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
import tkinterdnd2 as tkdnd
import json

# Load location definitions from JSON file
with open("landmarks.json", "r", encoding="utf-8") as f:
    CITY_LANDMARKS = json.load(f)

def analyze_image(image_path):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
    except Exception as e:
        return f"Image could not be loaded: {e}"
    
    # Stage 1: General recognition
    general_labels = [city["general_label"] for city in CITY_LANDMARKS.values()] + ["a street sign", "a generic cityscape"]
    inputs = processor(text=general_labels, images=image, return_tensors="pt", padding=True)
    model = model.to("cuda")
    with torch.no_grad():
        outputs = model(**inputs.to("cuda"))
    
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    general_results = [(label, prob) for label, prob in zip(general_labels, probs) if prob > 0.2]
    
    # Stage 2: Specific recognition
    specific_results = []
    for city, data in CITY_LANDMARKS.items():
        if any(city in label for label, _ in general_results):
            specific_labels = data["specific_labels"]
            inputs = processor(text=specific_labels, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs.to("cuda"))
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
            specific_results = [(label, prob) for label, prob in zip(specific_labels, probs) if prob > 0.2]
            break
    
    detected_objects = general_results + specific_results if specific_results else general_results
    
    torch.cuda.empty_cache()
    return detected_objects

def analyze_text(text):
    text = text.replace("#", "").strip().lower()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    locations = []
    for ent in doc.ents:
        if ent.label_ == "GPE":
            locations.append(ent.text)
    return locations

def geo_osint_analyzer(image_path, text, output_text_widget):
    output_text_widget.delete(1.0, tk.END)
    output_text_widget.insert(tk.END, "Analyzing image...\n")
    
    objects = analyze_image(image_path)
    if isinstance(objects, str):
        output_text_widget.insert(tk.END, objects + "\n")
        return
    
    output_text_widget.insert(tk.END, "Detected objects and probabilities:\n")
    for obj, prob in objects:
        output_text_widget.insert(tk.END, f"{obj}: {prob:.2%}\n")
    
    output_text_widget.insert(tk.END, "\nAnalyzing text...\n")
    locations = analyze_text(text)
    output_text_widget.insert(tk.END, f"Detected locations: {locations}\n")
    
    result = {
        "objects": [obj for obj, _ in objects],
        "locations": locations,
        "estimated_location": "No location detected"
    }
    
    image_location = None
    for city, data in CITY_LANDMARKS.items():
        if any(city in obj for obj, _ in objects):
            image_location = city
            result["estimated_location"] = [city]
            key_landmarks = data["key_landmarks"]
            if any(label in obj for obj, _ in objects for label in key_landmarks):
                landmark_label = next(obj for obj, _ in objects if any(label in obj for label in key_landmarks))
                result["warning"] = f"Image shows {landmark_label}, location {city} very likely!"
            else:
                result["warning"] = f"Image shows {city} landmark, location {city} likely!"
            break
    
    if any("street sign" in obj for obj, _ in objects):
        result["warning"] = "Image contains street signs, possible privacy risk!"
    
    if image_location and locations and not any(image_location.lower() == loc.lower() for loc in locations):
        result["conflict_warning"] = f"Conflict: Image shows {image_location}, text mentions {locations[0]}"
    
    output_text_widget.insert(tk.END, "\nResults:\n")
    output_text_widget.insert(tk.END, f"Objects: {result['objects']}\n")
    output_text_widget.insert(tk.END, f"Locations: {result['locations']}\n")
    output_text_widget.insert(tk.END, f"Estimated location: {result['estimated_location']}\n")
    if "warning" in result:
        output_text_widget.insert(tk.END, f"Warning: {result['warning']}\n")
    if "conflict_warning" in result:
        output_text_widget.insert(tk.END, f"Conflict warning: {result['conflict_warning']}\n")

def create_gui():
    root = tkdnd.Tk()
    root.title("Geo-OSINT Context Analyzer")
    root.geometry("600x400")

    drop_label = ttk.Label(root, text="Drag and drop an image here or click to select", relief="solid", padding=10)
    drop_label.pack(pady=10, fill=tk.X, padx=10)

    text_label = ttk.Label(root, text="Enter text:")
    text_label.pack(pady=5)
    text_entry = ttk.Entry(root)
    text_entry.pack(fill=tk.X, padx=10)
    text_entry.insert(0, "Having coffee in Paris, love this city!")

    output_label = ttk.Label(root, text="Results:")
    output_label.pack(pady=5)
    output_text = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD)
    output_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    analyze_button = ttk.Button(root, text="Analyze", command=lambda: geo_osint_analyzer(image_path.get(), text_entry.get(), output_text))
    analyze_button.pack(pady=10)

    image_path = tk.StringVar(value="")

    def drop(event):
        path = event.data
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        image_path.set(path)
        drop_label.config(text=f"Image: {path}")

    drop_label.drop_target_register(tkdnd.DND_FILES)
    drop_label.dnd_bind('<<Drop>>', drop)

    def select_file():
        file_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            image_path.set(file_path)
            drop_label.config(text=f"Image: {file_path}")

    drop_label.bind("<Button-1>", lambda e: select_file())

    root.mainloop()

if __name__ == "__main__":
    create_gui()
