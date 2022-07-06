
# Prerequisites
	Linux
	Python 3
	PyTorch 1.0+ (Orginally developed upder v1.0, testing on v1.5 is also fine)
	NVIDIA GPU + CUDA CuDNN
	Other common libraries: matplotlib, cv2, PIL

# Getting Started

Data Preparation: 
	Please refer to [<a href="http://www.cvlibs.net/datasets/kitti/index.php">KITTI</a>] or [<a href="https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html">NYU Depth V2</a>] and process them into h5 files. <a href="https://github.com/fangchangma/sparse-to-dense.pytorch">Here</a> also provides preprocessed data.

# Tutorial:

1. Create a folder and a subfolder 'checkpoint/kitti'
2. Download the pretrained weights: [<a href="https://drive.google.com/file/d/1rFvrqQ1Qf5bT_WSmtZZP5c-FKAhRHKUn/view?usp=sharing">NYU-Depth 500 points training</a>] [<a href="https://drive.google.com/open?id=1RJZMnohlp9OVSkxkSUWm7psnbW2mRunH">KITTI 500 points training</a>] and put the .pth under 'checkpoint/kitti/'
3. Prepare data in the previous "getting started" section
4. Run "python3 evaluate.py --name kitti --checkpoints_dir ./checkpoint/ --test_path [path ot the testing file] "
4. You'll see completed depth maps are saved under 'vis/'

# Train/Evaluation:

For training, please run

	python3 train_depth_complete.py --name kitti --checkpoints_dir [path to save_dir] --train_path [train_data_dir] --test_path [test_data_dir]

If you use the preprocessed data from <a href="https://github.com/fangchangma/sparse-to-dense.pytorch">here</a>. The train/test data path should be ./kitti/train or ./kitti/val/ under your data directory.

If you want to use your data, please make your data into h5 dataset. (See dataloaders/dataloader.py) 

Other specifications: `--continue_train` would load the lastest saved ckpt. Also set --epoch_count to tell what's the next epoch_number. Otherwise, will start from epoch 0. Set hyperparameters by `--lr`, `--batch_size`, `--weight_decay`, or others. Please refer to the options/base_options.py and options/options.py

Note that the default batch size is 4 during the training and use gpu:0. You can set larger batch size (--batch_size=xx) with more gpus (--gpu_ids="0,1,2,3") to attain larger batch size training.

Example command:

	python3 train_depth_complete.py --name kitti --checkpoints_dir ./checkpoints --lr 0.001 --batch_size 4 --train_path './kitti/train/' --test_path './kitti/val/' --continue_train --epoch_count [next_epoch_number]
	
For evalutation, please run

	python3 evaluate.py --name kitti --checkpoints_dir [path to save_dir to load ckpt] --test_path [test_data_dir] [--epoch [epoch number]]

This will load the latest checkpoint to evaluate. Add `--epoch` to specify which epoch checkpoint you want to load.

1.Fix several bugs and take off redundant options.

2.Release Orb sparsifier

3.Pretrain models release: [<a href="https://drive.google.com/file/d/1rFvrqQ1Qf5bT_WSmtZZP5c-FKAhRHKUn/view?usp=sharing">NYU-Depth 500 points training</a>] [<a href="https://drive.google.com/open?id=1RJZMnohlp9OVSkxkSUWm7psnbW2mRunH">KITTI 500 points training</a>]