import numpy as np
import os
import csv
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import argparse

class HeightRegressionMetrics(object):
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics to initial state."""
        self.mse = 0.0
        self.rmse = 0.0
        self.abs = 0.0
        self.rmse_building = 0.0
        # rmse_matched removed - not applicable for IM2ELEVATION
        self.total_high_rise_rmse = 0.0
        self.total_mid_rise_rmse = 0.0
        self.total_low_rise_rmse = 0.0
        self.delta1_sum = 0.0
        self.delta2_sum = 0.0
        self.delta3_sum = 0.0
        self.total_samples = 0
        self.img_sample = 0
        self.count_mid_rise = 0
        self.count_high_rise = 0
        self.count_low_rise = 0
       

    def add_batch(self, gt_image, pre_image, gt_mask, pred_mask, eps=1e-5):
        assert gt_image.shape == pre_image.shape, "Shape mismatch: gt_image shape {}, pre_image shape {}".format(
            gt_image.shape, pre_image.shape)
        
        delta_gt_image = gt_image.copy()
        delta_pre_image = pre_image.copy()
        pre_image[pre_image <= 0] = eps
        gt_image[gt_image <= 0] = eps
        valid_mask = ((gt_image > 0) | (pre_image > 0))
        #print("Valid Mask", valid_mask.shape)
        building_mask = np.expand_dims((gt_mask == 1), axis=0)  # Buildings are 1 in your dataset
        #print("Building Mask", building_mask.shape)
        # Note: matched_building_mask not needed since IM2ELEVATION doesn't predict building masks
        if valid_mask.sum() > 0:
            
            mse_i = np.nanmean((gt_image[valid_mask] - pre_image[valid_mask]) ** 2)
            rmse_i = np.sqrt(mse_i)
            abs_i = np.nanmean(np.abs(gt_image[valid_mask] - pre_image[valid_mask]))

        if building_mask.sum() > 0:
            rmse_b = (np.nanmean((pre_image[building_mask] - gt_image[building_mask]) ** 2)) ** 0.5
        else:
            rmse_b = 0.0

        # Note: rmse_matched is not applicable for IM2ELEVATION since it doesn't predict building masks
        # Only ground truth building mask is available
        rmse_m = 0.0  # Placeholder - not used in final metrics


        low_rise_building_mask = (gt_image >= 1) & (gt_image < 15)
        mid_rise_building_mask = (gt_image >= 15) & (gt_image < 40)
        high_rise_building_mask = gt_image >= 40
        
        low_rise = gt_image[low_rise_building_mask]
        mid_rise = gt_image[mid_rise_building_mask]
        high_rise = gt_image[high_rise_building_mask]

        low_rise_pred = pre_image[low_rise_building_mask]
        mid_rise_pred = pre_image[mid_rise_building_mask]
        high_rise_pred = pre_image[high_rise_building_mask]

        if high_rise.size > 0 and high_rise_pred.size > 0:
            high_rise_mse = np.nanmean((high_rise - high_rise_pred) ** 2)
            high_rise_rmse = np.sqrt(high_rise_mse)
            self.total_high_rise_rmse += high_rise_rmse
            self.count_high_rise += 1  
        else:
            high_rise_rmse = None

        if mid_rise.size > 0 and mid_rise_pred.size > 0:
            mid_rise_mse = np.nanmean((mid_rise - mid_rise_pred) ** 2)
            mid_rise_rmse = np.sqrt(mid_rise_mse)
            self.total_mid_rise_rmse += mid_rise_rmse
            self.count_mid_rise += 1  
        else:
            mid_rise_rmse = None
        
        if low_rise.size > 0 and low_rise_pred.size > 0:
            low_rise_mse = np.nanmean((low_rise - low_rise_pred) ** 2)
            low_rise_rmse = np.sqrt(low_rise_mse)
            self.total_low_rise_rmse += low_rise_rmse
            self.count_low_rise += 1
        else:
            low_rise_rmse = None

        
        self.mse += mse_i
        self.rmse += rmse_i
        self.rmse_building += rmse_b
        # rmse_matched removed - not applicable for IM2ELEVATION
        self.abs += abs_i

        # DELTA METRICS
        delta_pre_image[delta_pre_image <= 0] = eps
        # delta_pre_image[delta_pre_image < 0] = 999
        delta_gt_image[delta_gt_image <= 0] = eps
        maxRatio = np.maximum(delta_pre_image / delta_gt_image, delta_gt_image / delta_pre_image)
        self.delta1_sum += (maxRatio < 1.25).mean()
        self.delta2_sum += (maxRatio < 1.25 ** 2).mean()
        self.delta3_sum += (maxRatio < 1.25 ** 3).mean()
        
        self.img_sample += 1

    def calculate_metrics(self):
        #mse = np.nanmean(self.mse_list)
       
        #mae = np.nanmean(self.abs_list)
        #rmse = np.nanmean(self.rmse_list)
        #rmse_building = np.nanmean(self.rmse_building_list)
        
        #delta1 = np.nanmean(self.delta1)#self.delta1 / self.total_samples
        #delta2 = np.nanmean(self.delta2)#self.delta2 / self.total_samples
        #delta3 = np.nanmean(self.delta3)#self.delta3 / self.total_samples
        mse = self.mse / self.img_sample
        rmse = self.rmse / self.img_sample
        rmse_building = self.rmse_building / self.img_sample
        # rmse_matched removed - not applicable for IM2ELEVATION
        high_rise_rmse = self.total_high_rise_rmse / self.count_high_rise if self.count_high_rise > 0 else 0
        mid_rise_rmse = self.total_mid_rise_rmse / self.count_mid_rise if self.count_mid_rise > 0 else 0
        low_rise_rmse = self.total_low_rise_rmse / self.count_low_rise if self.count_low_rise > 0 else 0
        mae = self.abs / self.img_sample
        delta1 = self.delta1_sum / self.img_sample
        delta2 = self.delta2_sum / self.img_sample
        delta3 = self.delta3_sum / self.img_sample
        
        return mse, rmse, rmse_building, high_rise_rmse, mid_rise_rmse, low_rise_rmse, mae, delta1, delta2, delta3

    def evaluate_from_saved_predictions(self, predictions_dir, csv_file, dataset_name,
                                      enable_clipping=False, clipping_threshold=30.0,
                                      enable_target_filtering=True, target_threshold=1.0):
        """
        Evaluate predictions saved as .npy files against ground truth DSM and SEM files.
        
        Args:
            predictions_dir: Directory containing prediction .npy files
            csv_file: CSV file containing paths to ground truth files  
            dataset_name: Name of the dataset for determining SEM path structure
        
        Returns:
            Dictionary containing all computed metrics
        """
        
        # Reset metrics for fresh evaluation
        self.reset()
        
        # Read CSV file to get ground truth paths
        gt_files = []
        with open(csv_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    rgb_path = parts[0]
                    dsm_path = parts[1]
                    
                    # Construct SEM path based on DSM path
                    sem_path = dsm_path.replace('/dsm/', '/sem/')
                    
                    # Intelligently determine SEM file extension by checking what exists
                    sem_base = sem_path
                    if sem_path.endswith('.tiff'):
                        sem_base = sem_path.replace('.tiff', '')
                    elif sem_path.endswith('.tif'):
                        sem_base = sem_path.replace('.tif', '')
                    
                    # Try different extensions to find the actual SEM file
                    possible_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
                    sem_path = None
                    for ext in possible_extensions:
                        test_path = sem_base + ext
                        if os.path.exists(test_path):
                            sem_path = test_path
                            break
                    
                    # If no SEM file found, skip this sample
                    if sem_path is None:
                        continue
                    
                    gt_files.append({
                        'rgb_path': rgb_path,
                        'dsm_path': dsm_path, 
                        'sem_path': sem_path
                    })
        
        print(f"Found {len(gt_files)} ground truth files to evaluate")
        
        evaluated_count = 0
        missing_predictions = []
        
        for gt_file in gt_files:
            # Get image name for prediction file (remove extension to match prediction file naming)
            image_name = os.path.basename(gt_file['rgb_path'])
            image_name = os.path.splitext(image_name)[0]
            pred_file = os.path.join(predictions_dir, f"{image_name}_pred.npy")
            
            if not os.path.exists(pred_file):
                missing_predictions.append(image_name)
                continue
                
            try:
                # Load prediction
                pred_dsm = np.load(pred_file)
                
                # Load ground truth DSM
                if not os.path.exists(gt_file['dsm_path']):
                    print(f"Warning: Ground truth DSM not found: {gt_file['dsm_path']}")
                    continue
                    
                gt_dsm = self._load_tiff_image(gt_file['dsm_path'])
                
                # Load semantic segmentation (building mask)
                if not os.path.exists(gt_file['sem_path']):
                    print(f"Warning: SEM file not found: {gt_file['sem_path']}")
                    continue
                    
                gt_sem = self._load_tiff_image(gt_file['sem_path'])
                
                # Ensure predictions and ground truth have same dimensions
                # Follow the same preprocessing as training/testing: resize ground truth to 440x440
                if gt_dsm.shape != (440, 440):
                    print(f"Resizing ground truth DSM from {gt_dsm.shape} to (440, 440) for {image_name}")
                    gt_dsm = cv2.resize(gt_dsm, (440, 440), interpolation=cv2.INTER_LINEAR)
                
                if gt_sem.shape != (440, 440):
                    print(f"Resizing ground truth SEM from {gt_sem.shape} to (440, 440) for {image_name}")
                    gt_sem = cv2.resize(gt_sem, (440, 440), interpolation=cv2.INTER_NEAREST)
                
                # Predictions should already be 440x440, but verify
                if pred_dsm.shape != (440, 440):
                    print(f"Warning: Prediction has unexpected size {pred_dsm.shape}, expected (440, 440) for {image_name}")
                    pred_dsm = cv2.resize(pred_dsm, (440, 440), interpolation=cv2.INTER_LINEAR)
                
                # Apply the same preprocessing as in test phase (util.evaluateError)
                # Note: predictions are already scaled by 100, so no additional scaling needed
                
                # Apply masking: set predictions to 0 where ground truth <= threshold (if enabled)
                pred_dsm_processed = pred_dsm.copy()
                if enable_target_filtering:
                    idx_zero = np.where(gt_dsm <= target_threshold)
                    pred_dsm_processed[idx_zero] = 0
                
                # Apply clipping: set predictions to 0 where >= threshold (if enabled)
                if enable_clipping:
                    pred_dsm_processed[np.where(pred_dsm_processed >= clipping_threshold)] = 0
                
                # Add batch dimension for compatibility with original metrics
                pred_dsm_batch = np.expand_dims(pred_dsm_processed, axis=0)
                gt_dsm_batch = np.expand_dims(gt_dsm, axis=0)
                
                # Add batch to metrics (pred_mask is not used in IM2ELEVATION)
                self.add_batch(gt_dsm_batch, pred_dsm_batch, gt_sem, None)
                evaluated_count += 1
                
                if evaluated_count % 10 == 0:
                    print(f"Evaluated {evaluated_count} samples...")
                    
            except Exception as e:
                print(f"Error evaluating {image_name}: {str(e)}")
                continue
        
        if missing_predictions:
            print(f"Warning: {len(missing_predictions)} prediction files were missing")
            if len(missing_predictions) <= 10:
                print("Missing predictions:", missing_predictions)
        
        print(f"Successfully evaluated {evaluated_count} samples")
        
        if evaluated_count == 0:
            print("No samples were successfully evaluated!")
            return None
        
        # Calculate final metrics using original method
        results = self.calculate_metrics()
        mse, rmse, rmse_building, high_rise_rmse, mid_rise_rmse, low_rise_rmse, mae, delta1, delta2, delta3 = results
        
        metrics_dict = {
            'evaluated_samples': evaluated_count,
            'mse': mse,
            'rmse': rmse,
            'rmse_building': rmse_building,
            'high_rise_rmse': high_rise_rmse,
            'mid_rise_rmse': mid_rise_rmse, 
            'low_rise_rmse': low_rise_rmse,
            'mae': mae,
            'delta1': delta1,
            'delta2': delta2,
            'delta3': delta3
        }
        
        return metrics_dict

    def _load_tiff_image(self, path):
        """Load a TIFF image and return as numpy array."""
        try:
            # Try using PIL first
            with Image.open(path) as img:
                return np.array(img, dtype=np.float32)
        except Exception:
            # Fallback to cv2
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                return img.astype(np.float32)
            else:
                raise ValueError(f"Could not load image: {path}")

    def print_results(self, metrics_dict, dataset_name):
        """Print evaluation results in a formatted way."""
        
        print("=" * 60)
        print(f"IM2ELEVATION EVALUATION RESULTS FOR {dataset_name.upper()}")
        print("=" * 60)
        print(f"Evaluated Samples: {metrics_dict['evaluated_samples']}")
        print(f"MSE: {metrics_dict['mse']:.4f}")
        print(f"RMSE: {metrics_dict['rmse']:.4f}")
        print(f"MAE: {metrics_dict['mae']:.4f}")
        print(f"RMSE Building: {metrics_dict['rmse_building']:.4f}")
        print(f"High-rise RMSE: {metrics_dict['high_rise_rmse']:.4f}")
        print(f"Mid-rise RMSE: {metrics_dict['mid_rise_rmse']:.4f}")
        print(f"Low-rise RMSE: {metrics_dict['low_rise_rmse']:.4f}")
        print(f"δ₁ < 1.25: {metrics_dict['delta1']*100:.2f}%")
        print(f"δ₂ < 1.25²: {metrics_dict['delta2']*100:.2f}%")
        print(f"δ₃ < 1.25³: {metrics_dict['delta3']*100:.2f}%")
        print("=" * 60)

    def save_results(self, metrics_dict, dataset_name, output_dir):
        """Save evaluation results to a text file."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"evaluation_results_{dataset_name}_{timestamp}.txt"
        results_path = os.path.join(output_dir, results_filename)
        
        with open(results_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"IM2ELEVATION EVALUATION RESULTS FOR {dataset_name.upper()}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Evaluated Samples: {metrics_dict['evaluated_samples']}\n")
            f.write(f"MSE: {metrics_dict['mse']:.4f}\n")
            f.write(f"RMSE: {metrics_dict['rmse']:.4f}\n")
            f.write(f"MAE: {metrics_dict['mae']:.4f}\n")
            f.write(f"RMSE Building: {metrics_dict['rmse_building']:.4f}\n")
            f.write(f"High-rise RMSE: {metrics_dict['high_rise_rmse']:.4f}\n")
            f.write(f"Mid-rise RMSE: {metrics_dict['mid_rise_rmse']:.4f}\n")
            f.write(f"Low-rise RMSE: {metrics_dict['low_rise_rmse']:.4f}\n")
            f.write(f"δ₁ < 1.25: {metrics_dict['delta1']*100:.2f}%\n")
            f.write(f"δ₂ < 1.25²: {metrics_dict['delta2']*100:.2f}%\n")
            f.write(f"δ₃ < 1.25³: {metrics_dict['delta3']*100:.2f}%\n")
            f.write("=" * 60 + "\n")
        
        print(f"Results saved to: {results_path}")
        return results_path


# Legacy function for backward compatibility
def load_tiff_image(path):
    """Load a TIFF image and return as numpy array."""
    metrics = HeightRegressionMetrics()
    return metrics._load_tiff_image(path)


def evaluate_predictions_from_files(predictions_dir, csv_file, dataset_name):
    """
    Legacy function for backward compatibility.
    Use HeightRegressionMetrics.evaluate_from_saved_predictions() instead.
    """
    metrics = HeightRegressionMetrics()
    return metrics.evaluate_from_saved_predictions(predictions_dir, csv_file, dataset_name)


def print_evaluation_results(metrics_dict, dataset_name):
    """Legacy function for backward compatibility."""
    print("=" * 60)
    print(f"IM2ELEVATION EVALUATION RESULTS FOR {dataset_name.upper()}")
    print("=" * 60)
    print(f"Evaluated Samples: {metrics_dict['evaluated_samples']}")
    print(f"MSE: {metrics_dict['mse']:.4f}")
    print(f"RMSE: {metrics_dict['rmse']:.4f}")
    print(f"MAE: {metrics_dict['mae']:.4f}")
    print(f"RMSE Building: {metrics_dict['rmse_building']:.4f}")
    print(f"High-rise RMSE: {metrics_dict['high_rise_rmse']:.4f}")
    print(f"Mid-rise RMSE: {metrics_dict['mid_rise_rmse']:.4f}")
    print(f"Low-rise RMSE: {metrics_dict['low_rise_rmse']:.4f}")
    print(f"δ₁ < 1.25: {metrics_dict['delta1']*100:.2f}%")
    print(f"δ₂ < 1.25²: {metrics_dict['delta2']*100:.2f}%")
    print(f"δ₃ < 1.25³: {metrics_dict['delta3']*100:.2f}%")
    print("=" * 60)


def save_evaluation_results(metrics_dict, dataset_name, output_dir):
    """Legacy function for backward compatibility."""
    metrics = HeightRegressionMetrics()
    return metrics.save_results(metrics_dict, dataset_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IM2ELEVATION predictions')
    parser.add_argument('--predictions-dir', required=True, 
                       help='Directory containing prediction .npy files')
    parser.add_argument('--csv-file', required=True,
                       help='CSV file with ground truth paths') 
    parser.add_argument('--dataset-name', required=True,
                       help='Name of the dataset')
    parser.add_argument('--output-dir', 
                       help='Directory to save results (default: parent of predictions-dir)')
    
    # Optional clipping parameters
    parser.add_argument('--enable-clipping', action='store_true', default=False,
                       help='Enable clipping of predictions >= threshold (default: disabled)')
    parser.add_argument('--clipping-threshold', type=float, default=30.0,
                       help='Threshold for clipping predictions (default: 30.0)')
    parser.add_argument('--disable-target-filtering', action='store_true', default=False,
                       help='Disable filtering targets <= threshold (default: enabled)')
    parser.add_argument('--target-threshold', type=float, default=1.0,
                       help='Threshold for target filtering (default: 1.0)')
    
    args = parser.parse_args()
    
    # Default to parent directory of predictions-dir instead of predictions-dir itself
    output_dir = args.output_dir or os.path.dirname(args.predictions_dir)
    
    print(f"Evaluating predictions from: {args.predictions_dir}")
    print(f"Using ground truth from: {args.csv_file}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Clipping enabled: {args.enable_clipping}")
    if args.enable_clipping:
        print(f"Clipping threshold: {args.clipping_threshold}")
    print(f"Target filtering enabled: {not args.disable_target_filtering}")
    if not args.disable_target_filtering:
        print(f"Target threshold: {args.target_threshold}")
    
    # Initialize metrics and run evaluation
    metrics_calculator = HeightRegressionMetrics()
    metrics = metrics_calculator.evaluate_from_saved_predictions(
        args.predictions_dir, 
        args.csv_file, 
        args.dataset_name,
        enable_clipping=args.enable_clipping,
        clipping_threshold=args.clipping_threshold,
        enable_target_filtering=not args.disable_target_filtering,
        target_threshold=args.target_threshold
    )
    
    if metrics:
        metrics_calculator.print_results(metrics, args.dataset_name)
        metrics_calculator.save_results(metrics, args.dataset_name, output_dir)
