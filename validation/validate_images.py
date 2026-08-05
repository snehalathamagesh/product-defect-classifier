import os
import sys
from PIL import Image

RAW_DATA_DIR = "data/raw"
VALID_CLASSES = ["ok", "defective"]

def validate_image_dataset(data_dir=RAW_DATA_DIR):
    print("🔍 Starting Image Data Validation Gate...")
    
    if not os.path.exists(data_dir):
        print(f"❌ Error: Raw data path '{data_dir}' does not exist.")
        sys.exit(1)

    total_images = 0
    corrupt_images = 0
    class_counts = {c: 0 for c in VALID_CLASSES}

    for category in VALID_CLASSES:
        cat_dir = os.path.join(data_dir, category)
        if not os.path.exists(cat_dir):
            print(f"❌ Error: Required class directory '{cat_dir}' missing.")
            sys.exit(1)

        for img_name in os.listdir(cat_dir):
            img_path = os.path.join(cat_dir, img_name)
            total_images += 1
            
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image integrity
                
                # Re-open after verify() to check attributes
                with Image.open(img_path) as img:
                    if img.mode not in ["RGB", "L"]:
                        print(f"⚠️ Warning: {img_name} has non-standard mode: {img.mode}")
                    class_counts[category] += 1

            except Exception as e:
                print(f"❌ Corrupt image detected: {img_path} | Error: {e}")
                corrupt_images += 1

    print("\n--- Validation Summary ---")
    print(f"Total Images Checked: {total_images}")
    print(f"Class Counts: {class_counts}")
    print(f"Corrupt Files: {corrupt_images}")

    # Pipeline failure rule
    if corrupt_images > 0:
        print("\n❌ Data Validation FAILED: Found corrupt images. Pipeline halted.")
        sys.exit(1)
        
    print("✅ Image Data Validation Passed! Ready for training pipeline.\n")

if __name__ == "__main__":
    validate_image_dataset()