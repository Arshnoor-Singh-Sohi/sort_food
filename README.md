# Food Image Filter

A Python tool that automatically sorts your image collection by identifying which photos contain food. Perfect for organizing photos from restaurants, cooking, or mixed photo collections.

## What It Does

The tool scans through a folder of images and sorts them into three categories:
- **Food Images** - Photos clearly containing food
- **Non-Food Images** - Photos with no food detected  
- **Uncertain Images** - Photos that might contain food but need manual review

## Quick Start

### Installation
```bash
pip install ultralytics torch torchvision pillow tqdm matplotlib opencv-python numpy
```

### Basic Usage
1. Put your images in a folder called `images`
2. Run the tool:
```bash
python optimized_food_filter.py --source images
```

That's it! Check the three output folders that get created automatically.

## Understanding Confidence Levels

The tool uses a confidence threshold to decide how certain it needs to be before classifying an image as food. Think of it like this:

- **High confidence (0.5+)**: Very strict, only obvious food photos
- **Medium confidence (0.35)**: Balanced approach (recommended)  
- **Low confidence (0.25)**: More inclusive, catches more food but creates more uncertain cases

### Common Commands
```bash
# More inclusive (catches more food)
python optimized_food_filter.py --source images --confidence 0.25

# More selective (higher accuracy)  
python optimized_food_filter.py --source images --confidence 0.45

# Analyze why some images are uncertain
python optimized_food_filter.py --source images --analyze
```

## Key Features

**Smart Detection**: Uses AI object detection plus contextual clues like plates, dining tables, and utensils to identify food scenes even when the food itself is partially hidden.

**Multiple Detection Methods**: Combines direct food recognition, context analysis, and color patterns for better accuracy.

**Batch Processing**: Handles large image collections efficiently with progress tracking.

**Detailed Logging**: Keeps track of what was detected and why, helping you understand the results.

## File Structure After Processing
```
your-images-folder/
├── food_images/          # Photos with food detected
├── non_food_images/      # Photos without food
├── uncertain_images/     # Photos needing manual review
├── processing_stats.json # Detailed statistics
└── food_filter_log.txt   # Processing log
```

## Common Issues & Solutions

**"Too many uncertain images"**: Lower the confidence threshold to 0.25 or 0.20

**"Missing obvious food photos"**: The images might have unusual angles or lighting. Try the analyze option to see what's happening.

**"Out of memory errors"**: Reduce batch size with `--batch-size 25`

**"No images found"**: Make sure images are directly in the source folder, not in subfolders

## When to Review Uncertain Images

The uncertain folder typically contains:
- Food photos with unusual presentation
- Images with partial food visibility
- Photos where context suggests food but it's not clearly visible
- Borderline cases that benefit from human judgment

Spending a few minutes reviewing this folder usually helps you catch any missed food photos and understand how the tool interprets different types of images.

## Advanced Options

```bash
# Disable context detection (dining tables, utensils, etc.)
python optimized_food_filter.py --source images --no-context

# Disable color analysis  
python optimized_food_filter.py --source images --no-color

# Process smaller batches for large datasets
python optimized_food_filter.py --source images --batch-size 50
```

## System Requirements

- Python 3.7 or newer
- At least 4GB RAM (8GB recommended for large collections)
- GPU support optional but recommended for faster processing

The first run downloads a small AI model (~6MB) automatically. Processing speed depends on your hardware, but expect roughly 1-2 seconds per image on modern systems.
