import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import json
import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import gc

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_filter_log.txt'),
        logging.StreamHandler()
    ]
)

class FoodImageFilter:
    """
    A class to filter images based on food content detection.
    Uses YOLO for accurate food detection.
    """
    
    def __init__(self, source_folder: str, confidence_threshold: float = 0.7):
        """
        Initialize the food image filter.
        
        Args:
            source_folder: Path to folder containing images
            confidence_threshold: Minimum confidence score to consider an image as containing food
        """
        self.source_folder = Path(source_folder)
        self.confidence_threshold = confidence_threshold
        
        # Create folders for organization
        self.food_folder = self.source_folder / "food_images"
        self.non_food_folder = self.source_folder / "non_food_images"
        self.uncertain_folder = self.source_folder / "uncertain_images"
        
        # Create directories if they don't exist
        self.food_folder.mkdir(exist_ok=True)
        self.non_food_folder.mkdir(exist_ok=True)
        self.uncertain_folder.mkdir(exist_ok=True)
        
        # Initialize the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_yolo_model()
        
        # Food-related categories - expanded list
        self.food_categories = {
            'pizza', 'hamburger', 'hot dog', 'hotdog', 'apple', 'banana', 'orange',
            'broccoli', 'carrot', 'sandwich', 'cake', 'ice cream', 'donut', 'doughnut',
            'pasta', 'salad', 'bread', 'cheese', 'meat', 'chicken', 'fish',
            'egg', 'soup', 'rice', 'sushi', 'steak', 'french fries', 'fries',
            'chocolate', 'coffee', 'wine', 'beer', 'juice', 'milk', 'tea',
            'vegetable', 'fruit', 'dessert', 'snack', 'meal', 'dish',
            'bowl', 'cup', 'wine glass', 'dining table', 'kitchen',
            'breakfast', 'lunch', 'dinner', 'food', 'beverage', 'drink'
        }
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'food_images': 0,
            'non_food_images': 0,
            'uncertain_images': 0,
            'errors': 0
        }
        
        # Track processed files to avoid duplicates
        self.processed_files = set()
    
    def _load_yolo_model(self):
        """
        Load YOLO model for food detection.
        """
        try:
            from ultralytics import YOLO
            # Load YOLOv8 model
            model = YOLO('yolov8n.pt')
            logging.info("Successfully loaded YOLO model")
            return model
        except ImportError:
            logging.error("Ultralytics not installed. Please install it with: pip install ultralytics")
            raise
    
    def _is_food_image(self, image_path: Path) -> Tuple[bool, float, str]:
        """
        Determine if an image contains food using YOLO.
        
        Returns:
            Tuple of (is_food, confidence, detected_class)
        """
        try:
            # Check if file exists
            if not image_path.exists():
                logging.error(f"File does not exist: {image_path}")
                return False, 0.0, "file_not_found"
            
            # Run YOLO inference
            results = self.model(str(image_path), verbose=False)
            
            # Check detected objects
            food_detected = False
            max_confidence = 0.0
            detected_items = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get class name and confidence
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Check if detected object is food-related
                        if any(food_word in class_name.lower() for food_word in self.food_categories):
                            food_detected = True
                            if confidence > max_confidence:
                                max_confidence = confidence
                            detected_items.append(f"{class_name}({confidence:.2f})")
            
            detected_string = ", ".join(detected_items) if detected_items else "none"
            return food_detected, max_confidence, detected_string
            
        except Exception as e:
            logging.error(f"Error in YOLO detection for {image_path}: {str(e)}")
            return False, 0.0, "error"
    
    def process_images(self):
        """
        Process all images in the source folder.
        """
        logging.info(f"Starting to process images in {self.source_folder}")
        
        # Get all image files in the source folder (not in subdirectories)
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        all_images = []
        
        for item in self.source_folder.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                all_images.append(item)
        
        logging.info(f"Found {len(all_images)} images to process")
        
        if not all_images:
            logging.warning("No images found in the source folder")
            return
        
        # Process images with progress bar
        for image_path in tqdm(all_images, desc="Processing images"):
            # Skip if already processed
            if image_path.name in self.processed_files:
                logging.debug(f"Skipping already processed file: {image_path.name}")
                continue
            
            try:
                # Detect if food is present
                is_food, confidence, detected_items = self._is_food_image(image_path)
                
                # Determine destination based on detection results
                if is_food and confidence >= self.confidence_threshold:
                    destination_folder = self.food_folder
                    self.stats['food_images'] += 1
                    logging.info(f"Food detected: {image_path.name} - {detected_items}")
                elif confidence >= self.confidence_threshold * 0.5:
                    # Uncertain - might need manual review
                    destination_folder = self.uncertain_folder
                    self.stats['uncertain_images'] += 1
                    logging.info(f"Uncertain: {image_path.name} - confidence: {confidence:.2f}")
                else:
                    destination_folder = self.non_food_folder
                    self.stats['non_food_images'] += 1
                    logging.info(f"Non-food: {image_path.name}")
                
                # Move the file
                destination_path = destination_folder / image_path.name
                
                # Handle case where file with same name already exists
                if destination_path.exists():
                    base_name = image_path.stem
                    extension = image_path.suffix
                    counter = 1
                    while destination_path.exists():
                        new_name = f"{base_name}_{counter}{extension}"
                        destination_path = destination_folder / new_name
                        counter += 1
                
                shutil.move(str(image_path), str(destination_path))
                self.stats['total_processed'] += 1
                self.processed_files.add(image_path.name)
                
            except Exception as e:
                logging.error(f"Error processing {image_path}: {str(e)}")
                self.stats['errors'] += 1
                continue
        
        # Save statistics
        self._save_statistics()
        
        # Clear GPU memory
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def _save_statistics(self):
        """Save processing statistics to a JSON file."""
        stats_file = self.source_folder / "processing_stats.json"
        self.stats['timestamp'] = datetime.now().isoformat()
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logging.info(f"Statistics saved to {stats_file}")
        logging.info(f"Summary: {self.stats}")
    
    def verify_results(self, sample_size: int = 10):
        """
        Display a sample of results for manual verification.
        """
        try:
            import matplotlib.pyplot as plt
            from random import sample
            
            categories = [
                (self.food_folder, "Food Images"),
                (self.non_food_folder, "Non-Food Images"),
                (self.uncertain_folder, "Uncertain Images")
            ]
            
            for folder, title in categories:
                images = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
                if not images:
                    logging.info(f"No images found in {folder}")
                    continue
                
                # Sample images
                sampled = sample(images, min(sample_size, len(images)))
                
                # Create a figure
                fig, axes = plt.subplots(2, 5, figsize=(15, 6))
                fig.suptitle(title)
                axes = axes.flatten()
                
                for idx, img_path in enumerate(sampled[:10]):
                    try:
                        img = Image.open(img_path)
                        axes[idx].imshow(img)
                        axes[idx].set_title(img_path.name[:20])
                        axes[idx].axis('off')
                    except Exception as e:
                        logging.error(f"Error displaying {img_path}: {str(e)}")
                
                # Hide unused subplots
                for idx in range(len(sampled), 10):
                    axes[idx].axis('off')
                
                plt.tight_layout()
                sample_filename = f"{title.lower().replace(' ', '_')}_sample.png"
                plt.savefig(self.source_folder / sample_filename)
                plt.close()
                logging.info(f"Sample images saved to {sample_filename}")
                
        except ImportError:
            logging.warning("Matplotlib not installed. Skipping visual verification.")
        except Exception as e:
            logging.error(f"Error in verify_results: {str(e)}")


def main():
    """
    Main function to run the food image filtering process.
    """
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Filter food images from a folder')
    parser.add_argument('--source', type=str, default='images', 
                        help='Path to folder containing images (default: images)')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Confidence threshold for food detection (default: 0.7)')
    parser.add_argument('--verify', action='store_true',
                        help='Create sample images for verification')
    
    args = parser.parse_args()
    
    # Validate source folder
    source_path = Path(args.source)
    if not source_path.exists():
        logging.error(f"Source folder does not exist: {source_path}")
        return
    
    # Create filter instance
    try:
        filter_instance = FoodImageFilter(args.source, args.confidence)
    except Exception as e:
        logging.error(f"Failed to initialize filter: {str(e)}")
        return
    
    # Process the images
    filter_instance.process_images()
    
    # Verify results if requested
    if args.verify:
        filter_instance.verify_results()
    
    # Print final statistics
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total processed: {filter_instance.stats['total_processed']}")
    print(f"Food images: {filter_instance.stats['food_images']}")
    print(f"Non-food images: {filter_instance.stats['non_food_images']}")
    print(f"Uncertain images: {filter_instance.stats['uncertain_images']}")
    print(f"Errors: {filter_instance.stats['errors']}")
    print("="*50)
    
    # Provide guidance on next steps
    if filter_instance.stats['uncertain_images'] > 0:
        print(f"\nNote: {filter_instance.stats['uncertain_images']} images were marked as uncertain.")
        print("Please review the 'uncertain_images' folder manually.")
    
    if filter_instance.stats['errors'] > 0:
        print(f"\nWarning: {filter_instance.stats['errors']} errors occurred during processing.")
        print("Check the log file for details.")


if __name__ == "__main__":
    main()