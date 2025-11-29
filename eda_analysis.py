import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from collections import Counter

# Set style for better plots
plt.style.use('default')  # Use default instead of seaborn-v0_8 which might not exist
sns.set_palette("husl")

# Dataset paths - Fixed based on your structure
BASE_DIR = os.path.join(os.getcwd(), 'Dataset')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation')  
TEST_DIR = os.path.join(BASE_DIR, 'test')

def count_images_in_folders(base_path):
    """Count images in real and fake folders"""
    counts = {}
    dataset_dir = os.path.join(base_path, 'Dataset')

    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(dataset_dir, split)
        if os.path.exists(split_path):
            counts[split] = {}
            for class_name in ['real', 'fake']:
                class_path = os.path.join(split_path, class_name)
                if os.path.exists(class_path):
                    image_files = [f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
                    counts[split][class_name] = len(image_files)
                    print(f"Found {len(image_files)} images in {class_path}")
                else:
                    counts[split][class_name] = 0
                    print(f"Directory not found: {class_path}")
        else:
            counts[split] = {'real': 0, 'fake': 0}
            print(f"Split directory not found: {split_path}")
    return counts

def plot_dataset_distribution(counts):
    """Plot dataset distribution across splits and classes - handles empty datasets"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Images per split
    splits = list(counts.keys())
    real_counts = [counts[split]['real'] for split in splits]
    fake_counts = [counts[split]['fake'] for split in splits]

    x = np.arange(len(splits))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, real_counts, width, label='Real', color='#2E8B57', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, fake_counts, width, label='Fake', color='#CD5C5C', alpha=0.8)

    axes[0].set_xlabel('Dataset Split')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_title('Dataset Distribution by Split')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(splits)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    max_count = max(max(real_counts) if real_counts else 1, max(fake_counts) if fake_counts else 1)
    for i, (real, fake) in enumerate(zip(real_counts, fake_counts)):
        if max_count > 0:
            axes[0].text(i - width/2, real + max_count*0.01, str(real), 
                        ha='center', va='bottom', fontweight='bold')
            axes[0].text(i + width/2, fake + max_count*0.01, str(fake), 
                        ha='center', va='bottom', fontweight='bold')

    # Plot 2: Overall class distribution
    total_real = sum(real_counts)
    total_fake = sum(fake_counts)

    if total_real > 0 or total_fake > 0:
        labels = ['Real', 'Fake']
        sizes = [total_real, total_fake]
        colors = ['#2E8B57', '#CD5C5C']

        # Only create pie chart if we have data
        if sum(sizes) > 0:
            # Remove zero values for pie chart
            non_zero_labels = []
            non_zero_sizes = []
            non_zero_colors = []

            for label, size, color in zip(labels, sizes, colors):
                if size > 0:
                    non_zero_labels.append(label)
                    non_zero_sizes.append(size)
                    non_zero_colors.append(color)

            if non_zero_sizes:
                axes[1].pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, 
                           autopct='%1.1f%%', startangle=90, 
                           textprops={'fontsize': 12, 'fontweight': 'bold'})
            else:
                axes[1].text(0.5, 0.5, 'No Data Available', ha='center', va='center', 
                           transform=axes[1].transAxes, fontsize=14)
        else:
            axes[1].text(0.5, 0.5, 'No Images Found', ha='center', va='center', 
                       transform=axes[1].transAxes, fontsize=14)
    else:
        axes[1].text(0.5, 0.5, 'No Images Found\nCheck Dataset Path', ha='center', va='center', 
                   transform=axes[1].transAxes, fontsize=14)

    axes[1].set_title('Overall Class Distribution')

    plt.tight_layout()
    plt.savefig('dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    return total_real, total_fake

def check_dataset_structure():
    """Check and display the actual dataset structure"""
    print("\n🔍 Checking dataset structure...")
    base_paths_to_check = [
        os.path.join(os.getcwd(), 'Dataset'),
        os.path.join(os.getcwd(), 'data'),
        os.path.join(os.getcwd(), 'dataset'),
        os.getcwd()
    ]

    found_structure = False

    for base_path in base_paths_to_check:
        if os.path.exists(base_path):
            print(f"\nChecking: {base_path}")
            try:
                contents = os.listdir(base_path)
                if contents:
                    print(f"Contents: {contents[:10]}...")  # Show first 10 items

                    # Look for common dataset folder names
                    dataset_folders = [f for f in contents if f.lower() in ['train', 'test', 'validation', 'val']]
                    if dataset_folders:
                        print(f"Found dataset folders: {dataset_folders}")
                        found_structure = True

                        # Check deeper structure
                        for folder in dataset_folders:
                            folder_path = os.path.join(base_path, folder)
                            if os.path.isdir(folder_path):
                                subcontents = os.listdir(folder_path)[:5]
                                print(f"  {folder}/: {subcontents}...")
                else:
                    print("Empty directory")
            except PermissionError:
                print("Permission denied")
            except Exception as e:
                print(f"Error: {e}")

    if not found_structure:
        print("\n❌ No standard dataset structure found!")
        print("Expected structure:")
        print("Dataset/")
        print("├── train/")
        print("│   ├── real/")
        print("│   └── fake/") 
        print("├── validation/")
        print("│   ├── real/")
        print("│   └── fake/")
        print("└── test/")
        print("    ├── real/")
        print("    └── fake/")

    return found_structure

def analyze_image_properties(data_dir, max_samples=100):
    """Analyze image properties like dimensions, file sizes, etc."""
    properties = {
        'widths': [], 'heights': [], 'file_sizes': [], 
        'aspect_ratios': [], 'classes': []
    }

    for class_name in ['real', 'fake']:
        class_path = os.path.join(data_dir, class_name)
        if os.path.exists(class_path):
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

            print(f"Processing {len(image_files)} {class_name} images...")

            # Sample random images to avoid processing too many
            if len(image_files) > max_samples:
                image_files = np.random.choice(image_files, max_samples, replace=False)

            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Get image dimensions
                    with Image.open(img_path) as img:
                        width, height = img.size
                        properties['widths'].append(width)
                        properties['heights'].append(height)
                        properties['aspect_ratios'].append(width/height)
                        properties['classes'].append(class_name)

                    # Get file size
                    file_size = os.path.getsize(img_path) / 1024  # KB
                    properties['file_sizes'].append(file_size)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    return properties

def main():
    """Main EDA function"""
    print("🔍 Starting Exploratory Data Analysis for Deepfake Dataset...")
    print("=" * 60)

    # First check the dataset structure
    structure_found = check_dataset_structure()

    # 1. Count images in all folders
    print("\n📊 Counting images in dataset...")
    counts = count_images_in_folders(os.getcwd())

    # Print counts
    total_images = 0
    for split, class_counts in counts.items():
        split_total = sum(class_counts.values())
        total_images += split_total
        print(f"{split.title()}: Real={class_counts['real']}, Fake={class_counts['fake']}, Total={split_total}")

    print(f"\nTotal images found: {total_images}")

    if total_images == 0:
        print("\n❌ No images found! Please check:")
        print("1. Dataset path is correct")
        print("2. Images are in the right folders (train/real, train/fake, etc.)")
        print("3. Image files have valid extensions (.jpg, .png, etc.)")
        return

    print("\n📈 Creating dataset distribution plots...")
    total_real, total_fake = plot_dataset_distribution(counts)

    # 2. Analyze training set properties if we found images
    train_dir = os.path.join(os.getcwd(), 'Dataset', 'train')
    if os.path.exists(train_dir) and (total_real > 0 or total_fake > 0):
        print("\n🔬 Analyzing image properties (sampling from training set)...")
        properties = analyze_image_properties(train_dir, max_samples=50)  # Reduced sample size

        if properties['widths']:  # Check if we have data
            print("\n📊 Creating basic statistics...")
            print(f"Sample size: {len(properties['widths'])} images")
            print(f"Width range: {min(properties['widths'])} - {max(properties['widths'])} px")
            print(f"Height range: {min(properties['heights'])} - {max(properties['heights'])} px")
            print(f"File size range: {min(properties['file_sizes']):.1f} - {max(properties['file_sizes']):.1f} KB")
        else:
            print("❌ No image properties could be analyzed!")
    else:
        print(f"❌ Training directory not found or empty: {train_dir}")

    print("\n✅ EDA Complete! Generated files:")
    print("   • dataset_distribution.png")

if __name__ == "__main__":
    main()
