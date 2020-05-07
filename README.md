# Visual Tracking ML-Project implemented on SiamR-CNN

This is Visual-Tracking machine learning project implemented on SiamR-CNN which is based on Faster R-CNN with visualization, written in Python3 and powered by TensorFlow 1.

We borrow some code from TensorPack's Faster R-CNN example: 
https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN

And from Siam R-CNN example: Visual Tracking by Re-Detection:
https://github.com/VisualComputingInstitute/SiamR-CNN

## Installation

### Download necessary libraries
Here we will put all external libraries and this repository into /home/${USERNAME}/vision and use 
pip to install common libraries
```
mkdir /home/${USERNAME}/vision
cd /home/${USERNAME}/vision

git clone https://github.com/VisualComputingInstitute/SiamR-CNN.git
git clone https://github.com/pvoigtlaender/got10k-toolkit.git
git clone https://github.com/tensorpack/tensorpack.git

cd tensorpack
git checkout d24a9230d50b1dea1712a4c2765a11876f1e193c
cd ..

pip3 install cython
pip3 install tensorflow-gpu==1.15
pip3 install wget shapely msgpack msgpack_numpy tabulate xmltodict pycocotools opencv-python tqdm zmq annoy
```
### Add libraries to your PYTHONPATH
```
export PYTHONPATH=${PYTHONPATH}:/home/${USERNAME}/vision/got10k-toolkit/:/home/${USERNAME}/vision/tensorpack/
```

### Make Folder for models and logs and download pre-trained model
```
cd SiamR-CNN/
mkdir train_log
cd train_log
wget --no-check-certificate -r -nH --cut-dirs=2 --no-parent --reject="index.html*" https://omnomnom.vision.rwth-aachen.de/data/siamrcnn/hard_mining3/
cd ..
```
## Running Tracking and Evaluation
First set the path to the dataset on which you want to evaluate in tracking/do_tracking.py, e.g.
```
OTB_2015_ROOT_DIR = '/data/otb2015/'
```

Then run tracking/do_tracking.py and specify the dataset you want to evaluate on using the main function for this dataset using e.g. --main main_otb
 
```
python3 tracking/do_tracking.py --main main_otb
```

The result will then be written to tracking_data/results/
## Visualization
这里需要写可视化的步骤 @瑞
