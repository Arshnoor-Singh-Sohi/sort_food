# Food Image Filter - Quick Start Guide

A Python tool that automatically sorts food images from non-food images using AI detection.

## Prerequisites

Install all required packages with this single command:

```bash
pip install ultralytics torch torchvision pillow tqdm matplotlib opencv-python numpy
```

### What each package does:
- **ultralytics**: Provides the YOLO object detection model
- **torch & torchvision**: Deep learning framework for running AI models
- **pillow**: Image processing and loading
- **tqdm**: Progress bars to track processing
- **matplotlib**: Creates sample image grids for verification
- **opencv-python**: Advanced image analysis for color detection
- **numpy**: Numerical operations for image arrays

## Quick Start

1. Place all your images in a folder called `images`
2. Run the basic command:
   ```bash
   python food_filter.py --source images
   ```
3. Check the three output folders:
   - `food_images/` - Images with food detected
   - `non_food_images/` - Images without food
   - `uncertain_images/` - Images needing manual review

## Confidence Values Guide

The confidence threshold determines how certain the AI needs to be before classifying an image as containing food.

### Recommended Settings by Use Case

**Default (Balanced)**
```bash
python food_filter.py --source images --confidence 0.35
```
Good for most cases - balances accuracy with coverage.

**Catch More Food (Inclusive)**
```bash
python food_filter.py --source images --confidence 0.25
```
Use when many food images are being missed. You'll get more false positives but fewer missed foods.

**High Accuracy (Conservative)**
```bash
python food_filter.py --source images --confidence 0.50
```
Use when accuracy matters more than catching every food image. Fewer false positives but might miss some food.

**Maximum Coverage (Very Inclusive)**
```bash
python food_filter.py --source images --confidence 0.15
```
Catches almost all food images but requires more manual review of uncertain cases.

### Understanding Confidence Values

- **0.15 - 0.25**: Very inclusive, catches most food but more false positives
- **0.25 - 0.35**: Balanced approach, good for diverse food images
- **0.35 - 0.50**: Conservative, high accuracy but might miss unusual foods
- **0.50+**: Very strict, only obvious food images

## Advanced Options

**Disable context detection** (plates, tables, etc.):
```bash
python food_filter.py --source images --no-context
```

**Disable color analysis**:
```bash
python food_filter.py --source images --no-color
```

**Analyze why images are uncertain**:
```bash
python food_filter.py --source images --analyze
```

**Process in smaller batches** (for large datasets):
```bash
python food_filter.py --source images --batch-size 50
```

## First Time Setup Notes

1. The first run will download the YOLO model (~6MB) automatically
2. Processing speed depends on GPU availability (check "Using device: cuda" in output)
3. For 26GB of images, expect several hours of processing time

## Troubleshooting

**"No images found"**: Check that images are directly in the source folder (not subfolders)

**Too many in uncertain folder**: Lower confidence to 0.25 or 0.20

**Too many false positives**: Raise confidence to 0.40 or 0.45

**Out of memory errors**: Reduce batch size with `--batch-size 25`

## Quick Decision Guide

Not sure what confidence to use? Start here:

- Restaurant/professional food photos → Use 0.35
- Home cooking/phone photos → Use 0.25  
- Mixed collection with non-food → Use 0.40
- Asian cuisine/soups/curries → Use 0.25
- Packaged foods → Use 0.20

After first run, check the statistics and adjust accordingly!
