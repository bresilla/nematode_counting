# SEGNEMA

### Installation

SEGNEMA is a software written in Python 3.10. It requires the following packages to be installed:
- pytorch 
- torchvision 
- pytorch-cuda
- ultralytics
- opencv 
- pillow
- numpy
- scipy
- tqdm
- pyyaml
- matplotlib

To install the required packages, run the following command:
```bash
pip install packagex packagex packagex ...
```

Or you wan use mamaba to install the required packages:
```bash
mamba install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
mamba install -c conda-forge opencv pillow numpy scipy tqdm pyyaml matplotlib
```

### Usage
The repository contains a 'gui.py' file that can be used to run the software. To run the software, execute the following command:
```bash
python gui.py
```

In addition, there is a folder called "models" which contains the segmentation model for segmenting the nematodes, and another model for detection the eggs.
The models are places in "eggs" or "nematodes" folders, respectively. Then all the files from training are placed there. The folder "weights" contains the weights of the models.

In file "nema.py" in line 162 and 163, you can change the path to the models and weights.
```python
    seg_model = YOLO("make/sure/you/put/the/path/here/weights/best.pt")
    det_model = YOLO("make/sure/you/put/the/path/here/weights/best.pt")
```

### License
This software is licensed under the MIT License.