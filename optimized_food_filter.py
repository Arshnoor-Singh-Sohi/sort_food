import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import logging
from datetime import datetime
import json
from pathlib import Path
from typing import List, Tuple, Dict, Set
import gc
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('food_filter_log.txt'),
        logging.StreamHandler()
    ]
)

class OptimizedFoodImageFilter:
    """
    Enhanced food image filter with multiple detection strategies.
    """
    
    def __init__(self, source_folder: str, 
                 confidence_threshold: float = 0.35,  # Lowered from 0.7
                 use_context_clues: bool = True,
                 use_color_analysis: bool = True):
        """
        Initialize the optimized food image filter.
        
        Args:
            source_folder: Path to folder containing images
            confidence_threshold: Minimum confidence score (lowered for better recall)
            use_context_clues: Whether to use contextual object detection
            use_color_analysis: Whether to use color-based food detection
        """
        self.source_folder = Path(source_folder)
        self.confidence_threshold = confidence_threshold
        self.use_context_clues = use_context_clues
        self.use_color_analysis = use_color_analysis
        
        # Create folders
        self.food_folder = self.source_folder / "food_images"
        self.non_food_folder = self.source_folder / "non_food_images"
        self.uncertain_folder = self.source_folder / "uncertain_images"
        
        for folder in [self.food_folder, self.non_food_folder, self.uncertain_folder]:
            folder.mkdir(exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load YOLO model
        self.model = self._load_yolo_model()
        
        # EXPANDED food-related categories
        # Direct food items
        self.direct_food_items = {
            'pizza', 'hamburger', 'hot dog', 'hotdog', 'apple', 'banana', 'orange',
            'broccoli', 'carrot', 'sandwich', 'cake', 'ice cream', 'donut', 'doughnut',
            'bagel', 'croissant', 'waffle', 'pancake', 'cookie', 'pretzel',
            'burrito', 'taco', 'pasta', 'salad', 'bread', 'cheese', 'meat', 
            'chicken', 'beef', 'pork', 'fish', 'seafood', 'shrimp', 'lobster',
            'egg', 'soup', 'rice', 'sushi', 'steak', 'french fries', 'fries',
            'chocolate', 'candy', 'dessert', 'pie', 'muffin', 'cupcake'
        }
        
        # Food-related items (context clues)
        self.food_context_items = {
            'bottle', 'bowl', 'cup', 'fork', 'knife', 'spoon', 'plate',
            'wine glass', 'dining table', 'kitchen', 'restaurant',
            'refrigerator', 'oven', 'microwave', 'sink', 'potted plant',
            'vase', 'wine bottle', 'beer bottle', 'can', 'jar'
        }
        
        # Combine all food-related categories
        self.all_food_categories = self.direct_food_items | self.food_context_items
        
        # Color ranges for food detection (in HSV)
        # These are typical color ranges for various foods
        self.food_color_ranges = [
            # Reds (tomatoes, strawberries, meat)
            ((0, 50, 50), (10, 255, 255)),
            ((170, 50, 50), (180, 255, 255)),
            # Oranges (oranges, carrots, salmon)
            ((10, 50, 50), (25, 255, 255)),
            # Yellows (bananas, corn, cheese)
            ((25, 50, 50), (35, 255, 255)),
            # Greens (vegetables)
            ((35, 50, 50), (85, 255, 255)),
            # Browns (bread, chocolate, cooked meat)
            ((10, 30, 30), (25, 150, 200)),
        ]
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'food_images': 0,
            'non_food_images': 0,
            'uncertain_images': 0,
            'errors': 0,
            'detection_methods': {
                'yolo_direct': 0,
                'context_clues': 0,
                'color_analysis': 0,
                'combined': 0
            }
        }
        
        self.processed_files = set()
    
    def _load_yolo_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            model = YOLO('yolov8n.pt')
            logging.info("Successfully loaded YOLO model")
            return model
        except ImportError:
            logging.error("Ultralytics not installed. Please install it with: pip install ultralytics")
            raise
    
    def _analyze_color_histogram(self, image_path: Path) -> float:
        """
        Analyze color histogram to detect food-like colors.
        Returns a score between 0 and 1 indicating food likelihood.
        """
        try:
            # Load image and convert to HSV
            image = Image.open(image_path).convert('RGB')
            # Resize for faster processing
            image.thumbnail((300, 300))
            img_array = np.array(image)
            
            # Convert to HSV
            import cv2
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Calculate percentage of pixels in food color ranges
            total_pixels = hsv.shape[0] * hsv.shape[1]
            food_pixels = 0
            
            for lower, upper in self.food_color_ranges:
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv, lower, upper)
                food_pixels += np.sum(mask > 0)
            
            # Return percentage of food-colored pixels
            food_color_score = food_pixels / total_pixels
            return min(food_color_score * 2, 1.0)  # Scale up and cap at 1.0
            
        except Exception as e:
            logging.debug(f"Color analysis failed for {image_path}: {str(e)}")
            return 0.0
    
    def _detect_food_with_context(self, image_path: Path) -> Tuple[bool, float, str, Dict]:
        """
        Enhanced food detection using multiple strategies.
        Returns: (is_food, confidence, detected_items, detection_method)
        """
        detection_scores = {
            'yolo_direct': 0.0,
            'context_clues': 0.0,
            'color_analysis': 0.0
        }
        detected_items = []
        
        try:
            # 1. YOLO Detection
            results = self.model(str(image_path), verbose=False)
            
            direct_food_found = False
            context_items_found = []
            max_direct_confidence = 0.0
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        confidence = float(box.conf[0])
                        
                        # Check for direct food items
                        if any(food in class_name.lower() for food in self.direct_food_items):
                            direct_food_found = True
                            max_direct_confidence = max(max_direct_confidence, confidence)
                            detected_items.append(f"{class_name}({confidence:.2f})")
                        
                        # Check for context items
                        elif any(context in class_name.lower() for context in self.food_context_items):
                            context_items_found.append((class_name, confidence))
            
            # Calculate YOLO direct score
            detection_scores['yolo_direct'] = max_direct_confidence
            
            # 2. Context-based scoring
            if self.use_context_clues and context_items_found:
                # Higher score if multiple food-related items detected
                context_score = 0.0
                for item, conf in context_items_found:
                    detected_items.append(f"{item}(context)")
                    # Dining table, bowl, plate are stronger indicators
                    if any(strong in item.lower() for strong in ['dining table', 'bowl', 'plate', 'cup']):
                        context_score += conf * 0.7
                    else:
                        context_score += conf * 0.3
                
                detection_scores['context_clues'] = min(context_score, 0.9)  # Cap at 0.9
            
            # 3. Color-based analysis
            if self.use_color_analysis:
                color_score = self._analyze_color_histogram(image_path)
                detection_scores['color_analysis'] = color_score * 0.5  # Weight color analysis lower
            
            # Combine scores with weighted average
            # Direct food detection is weighted highest
            weights = {
                'yolo_direct': 1.0,
                'context_clues': 0.6,
                'color_analysis': 0.3
            }
            
            total_weight = sum(weights.values())
            combined_score = sum(score * weights[method] for method, score in detection_scores.items())
            combined_score /= total_weight
            
            # Determine if it's food based on combined scoring
            is_food = (
                direct_food_found or  # Always classify as food if direct food found
                (detection_scores['context_clues'] > 0.5 and detection_scores['color_analysis'] > 0.3) or
                combined_score > self.confidence_threshold
            )
            
            # Track which method led to detection
            if is_food:
                if direct_food_found:
                    self.stats['detection_methods']['yolo_direct'] += 1
                elif detection_scores['context_clues'] > detection_scores['color_analysis']:
                    self.stats['detection_methods']['context_clues'] += 1
                else:
                    self.stats['detection_methods']['color_analysis'] += 1
            
            detected_string = ", ".join(detected_items) if detected_items else "none"
            return is_food, combined_score, detected_string, detection_scores
            
        except Exception as e:
            logging.error(f"Error in food detection for {image_path}: {str(e)}")
            return False, 0.0, "error", detection_scores
    
    def process_images(self, batch_size: int = 100):
        """Process all images with progress tracking."""
        logging.info(f"Starting to process images in {self.source_folder}")
        logging.info(f"Settings: confidence_threshold={self.confidence_threshold}, "
                    f"use_context={self.use_context_clues}, use_color={self.use_color_analysis}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        all_images = []
        
        for item in self.source_folder.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                all_images.append(item)
        
        logging.info(f"Found {len(all_images)} images to process")
        
        if not all_images:
            logging.warning("No images found in the source folder")
            return
        
        # Process images in batches
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1} of {len(all_images)//batch_size + 1}")
            
            for image_path in tqdm(batch, desc=f"Batch {i//batch_size + 1}"):
                if image_path.name in self.processed_files:
                    continue
                
                try:
                    # Detect food with enhanced methods
                    is_food, confidence, detected_items, scores = self._detect_food_with_context(image_path)
                    
                    # Log detailed detection info
                    logging.debug(f"{image_path.name} - Scores: YOLO={scores['yolo_direct']:.2f}, "
                                f"Context={scores['context_clues']:.2f}, Color={scores['color_analysis']:.2f}")
                    
                    # Determine destination
                    if is_food and confidence >= self.confidence_threshold:
                        destination_folder = self.food_folder
                        self.stats['food_images'] += 1
                        logging.info(f"Food detected: {image_path.name} - {detected_items} (conf: {confidence:.2f})")
                    elif confidence >= self.confidence_threshold * 0.7:  # More lenient for uncertain
                        destination_folder = self.uncertain_folder
                        self.stats['uncertain_images'] += 1
                        logging.info(f"Uncertain: {image_path.name} - confidence: {confidence:.2f}")
                    else:
                        destination_folder = self.non_food_folder
                        self.stats['non_food_images'] += 1
                        logging.info(f"Non-food: {image_path.name}")
                    
                    # Move file
                    destination_path = destination_folder / image_path.name
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
            
            # Clear memory after each batch
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Save statistics
        self._save_statistics()
    
    def _save_statistics(self):
        """Save detailed statistics."""
        stats_file = self.source_folder / "processing_stats.json"
        self.stats['timestamp'] = datetime.now().isoformat()
        
        # Calculate detection method percentages
        total_food = self.stats['food_images']
        if total_food > 0:
            self.stats['detection_method_percentages'] = {
                method: (count / total_food * 100) 
                for method, count in self.stats['detection_methods'].items()
            }
        
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logging.info(f"Statistics saved to {stats_file}")
        logging.info(f"Summary: {self.stats}")
    
    def analyze_misclassified_samples(self, sample_size: int = 20):
        """
        Analyze why certain images might be misclassified.
        This helps in understanding and improving the detection.
        """
        import matplotlib.pyplot as plt
        
        # Check uncertain folder for potential misclassifications
        uncertain_images = list(self.uncertain_folder.glob("*.jpg")) + \
                          list(self.uncertain_folder.glob("*.png"))
        
        if not uncertain_images:
            logging.info("No uncertain images to analyze")
            return
        
        logging.info(f"\nAnalyzing {min(sample_size, len(uncertain_images))} uncertain images...")
        
        # Sample and analyze
        from random import sample
        sampled = sample(uncertain_images, min(sample_size, len(uncertain_images)))
        
        analysis_results = []
        for img_path in sampled:
            is_food, confidence, detected, scores = self._detect_food_with_context(img_path)
            analysis_results.append({
                'image': img_path.name,
                'scores': scores,
                'detected_items': detected
            })
        
        # Save analysis
        with open(self.source_folder / "uncertain_analysis.json", 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logging.info("Analysis saved to uncertain_analysis.json")


def main():
    """Enhanced main function with optimization options."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimized food image filter')
    parser.add_argument('--source', type=str, default='images', 
                        help='Path to folder containing images')
    parser.add_argument('--confidence', type=float, default=0.35,
                        help='Confidence threshold (default: 0.35, lower = more inclusive)')
    parser.add_argument('--no-context', action='store_true',
                        help='Disable context-based detection')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable color-based detection')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze uncertain images to improve detection')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of images to process at once')
    
    args = parser.parse_args()
    
    # Validate source
    source_path = Path(args.source)
    if not source_path.exists():
        logging.error(f"Source folder does not exist: {source_path}")
        return
    
    # Create optimized filter
    try:
        filter_instance = OptimizedFoodImageFilter(
            args.source,
            confidence_threshold=args.confidence,
            use_context_clues=not args.no_context,
            use_color_analysis=not args.no_color
        )
    except Exception as e:
        logging.error(f"Failed to initialize filter: {str(e)}")
        return
    
    # Process images
    filter_instance.process_images(batch_size=args.batch_size)
    
    # Analyze if requested
    if args.analyze:
        filter_instance.analyze_misclassified_samples()
    
    # Print results
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"Total processed: {filter_instance.stats['total_processed']}")
    print(f"Food images: {filter_instance.stats['food_images']}")
    print(f"Non-food images: {filter_instance.stats['non_food_images']}")
    print(f"Uncertain images: {filter_instance.stats['uncertain_images']}")
    print(f"Errors: {filter_instance.stats['errors']}")
    print("\nDetection method breakdown:")
    for method, count in filter_instance.stats['detection_methods'].items():
        print(f"  {method}: {count}")
    print("="*60)
    
    # Recommendations
    if filter_instance.stats['uncertain_images'] > filter_instance.stats['food_images'] * 0.3:
        print("\nRecommendation: Many images are uncertain. Consider:")
        print("1. Lower the confidence threshold further (try --confidence 0.25)")
        print("2. Run with --analyze to understand why images are uncertain")
        print("3. Review the uncertain folder to identify patterns")
    
    if filter_instance.stats['non_food_images'] > filter_instance.stats['total_processed'] * 0.7:
        print("\nWarning: Many images classified as non-food. This might indicate:")
        print("1. The images contain food types not recognized by the model")
        print("2. The images have unusual lighting or angles")
        print("3. Consider using a specialized food detection model")


if __name__ == "__main__":
    main()