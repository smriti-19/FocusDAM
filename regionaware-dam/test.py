import torch
from PIL import Image
import json
from pathlib import Path
import sys
import os

# Add DAM repo
DAM_REPO_PATH = "/home/saksham/Desktop/GenAI/FocusDAM/external/describe-anything"
if os.path.exists(DAM_REPO_PATH):
    sys.path.insert(0, DAM_REPO_PATH)

# Load DAM
from dam import load_pretrained_model, get_model_name_from_path, disable_torch_init

def main():
    dataset_json = "/home/saksham/Desktop/GenAI/Kiru-Project/data/small_objects_dataset.json"
    images_dir = "/home/saksham/Desktop/GenAI/Kiru-Project/data/ref-l4"
    model_path = "/home/saksham/Desktop/GenAI/Kiru-Project/models/DAM-3B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading dataset…")
    with open(dataset_json, "r") as f:
        samples = json.load(f)["samples"]

    # Pick first sample
    sample = samples[0]
    print("Testing sample:", sample)

    img_path = Path(images_dir) / sample["file_name"]
    bbox = sample["bbox"]

    print("Loading image:", img_path)
    img = Image.open(img_path).convert("RGB")

    # Apply crop
    x, y, w, h = bbox
    img = img.crop((x, y, x+w, y+h))

    print("Loading model…")
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_name, model_base=None
    )
    model.to(device)
    model.eval()

    # Prepare image tensor
    processed = image_processor(img)
    # Pixel values come inside a list 
    pixel_tensor_np = processed["pixel_values"][0] 
    pixel_tensor = torch.from_numpy(pixel_tensor_np) 

    # Add batch dimension for DAM
    img_tensor = pixel_tensor.unsqueeze(0).to(device)



    # Prepare text
    prompt = "Describe this region."
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    print("\nRunning Generate()\n")

    # ***** CRUCIAL: DAM-style image injection *****
    model.image = img_tensor

    try:
        out = model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            max_new_tokens=30,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        print("\nSUCCESS!")
        print("Generated sequence:", tokenizer.decode(out.sequences[0]))
        print("Hidden states available:", hasattr(out, "decoder_hidden_states"))
    except Exception as e:
        print("\nError Occurred")
        print(type(e).__name__, ":", str(e))
        print("\nTraceback:\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
