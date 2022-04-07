# Difference Net for HuMID Problem
## About

Comparison between Difference Net and ST-Siamese Net for the HuMID problem.

## Architecture: DifferenceNet
<p float="left">
  <img src="https://github.com/FKTechLab/DifferenceNet/blob/main/humid/images/DN-HuMID.png" width="300" />
</p>

## Requirements
```sh
python==3.6
tensorflow==2.2.0
keras==2.2.0
```

## Usage

Clone this repository

Install Packages
```sh
pip install -r requirements.txt
```

Data:
Download the data from the following location and save into the dataset folder
```sh
https://github.com/huiminren/ST-SiameseNet/tree/master/dataset
```

optional arguments:
```sh
  -h, --help            show this help message and exit
    parser.add_argument('--mode', type=str, default='abs', 
                        help='difference function to the model (abs/simple/square)')    
    parser.add_argument('--model_size', type=str, default='small', 
                        help='model architecture size (big/medium/small)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='training batch size (default: 8)')
```


Run model
```sh
python main.py
```

