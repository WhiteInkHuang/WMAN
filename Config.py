import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Advanced Brain Infarction Detection/Segmentation')
    parser.add_argument('--data_dir', type=str, default='./splits', 
                       help='Path to dataset (DCM format)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                       help='Path to save model')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, 
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of Epoch')
    parser.add_argument('--optimizer', type=str, default='Adam', 
                       help='Optimizer to use')
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device to use')
    parser.add_argument('--k_split_value', type=int, default=5, 
                       help='k split value for k_fold mode')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='Number of data loading workers (set to 0 for Windows)')
    parser.add_argument('--model_name', type=str, default='AttentionDetection2D',
                       help='Model name: Detection2DCNN, AttentionDetection2D (for detection); UNet3D, AttentionUNet3D (for segmentation)')
    parser.add_argument('--target_size', type=int, nargs=3, default=[32, 512, 512],
                       help='Target size for volumes (D H W)')
    args = parser.parse_args()
    return args
