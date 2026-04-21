import numpy as np
import argparse
import os

def parse_args():
    description = "Lab639 Fisheye Dataset HAR Config"
    parser = argparse.ArgumentParser(description=description)

    # dataset
    parser.add_argument('--csv_path', type=str, default='data/lab639_fisheye')
    parser.add_argument('--train_csv', type=str, required=True)
    parser.add_argument('--val_csv', type=str, required=True)
    parser.add_argument('--test_csv', type=str, required=True)
    parser.add_argument('--result_path', type=str, default='result/lab639_fisheye')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory containing HDF5 files')
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--output_name', type=str)
    parser.add_argument('--fold_num', type=int, required=True)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--num_views', type=int)
    # [TESTED] a setting to enable `camera-id label`
    # default False
    parser.add_argument('--baseline', action='store_true', 
                    help='Set to True to use physical Camera ID for baseline experiment')

    # training & inference
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='Mode of operation: train or test')
    parser.add_argument('--model_path', type=str, help='Path to the trained model file')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--validation_interval', type=int, default=3)
    parser.add_argument('--fusion_type', type=str, required=True, choices=['max', 'mean', 'sum', 'concat', 'motion_weighted', 'transformer'], help='Fusion type for the model')
    parser.add_argument('--motion_score', type=str, required=True, help='Use motion score for fusion')

    args = parser.parse_args()

    if args.mode == 'train':
        if not args.exp_name:
            parser.error("Experiment name is required for training mode.")
        if not args.num_epochs:
            parser.error("Number of epochs is required for training mode.")
        if not args.learning_rate:
            parser.error("Learning rate is required for training mode.")
        if not args.weight_decay:
            parser.error("Weight decay is required for training mode.")
        if not args.optimizer:
            parser.error("Optimizer is required for training mode.")
        if not args.validation_interval:
            parser.error("Validation interval is required for training mode.")
    elif args.mode == 'test':
        if not args.model_path:
            parser.error("Model path is required for testing mode.")


    return args

class Lab639Config(object):
    def __init__(self, args):

        # dataset
        self.csv_path = args.csv_path
        self.train_csv = args.train_csv
        self.val_csv = args.val_csv
        self.test_csv = args.test_csv
        self.csv_offset = ""
        self.result_path = args.result_path
        self.data_path = args.data_path
        self.fold_num = args.fold_num
        self.num_classes = args.num_classes
        self.num_frames = args.num_frames
        self.num_views = args.num_views

        self.seed = 42

        # training & inference
        self.mode = args.mode
        self.model_path = args.model_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.fusion_type = args.fusion_type
        self.motion_score = True if args.motion_score == 'True' else False

        # [TESTED] a setting to enable `camera-id label`
        self.baseline = args.baseline


        if args.mode == 'train':
            self.exp_name = args.exp_name
            self.output_name = args.output_name
            self.num_epochs = args.num_epochs
            self.learning_rate = args.learning_rate
            self.weight_decay = args.weight_decay
            self.optimizer = args.optimizer
            self.validation_interval = args.validation_interval
        elif args.mode == 'test':
            self.model_path = args.model_path

    def __str__(self):
        return f"Lab639Config: ({self.__dict__})"