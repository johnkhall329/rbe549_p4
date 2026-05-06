
# Setup

install required packages:

```pip install -r requirements.txt```

# Data Generation

Plane textures downloaded from: https://www.esri.com/en-us/arcgis/products/arcgis-reality/resources/sample-drone-datasets

Once installed in ```Phase2/Data```, use `data_gen.py` to generate trajectories for training. Use ```--help``` to see the adjustable parameters.

# Inference

Run ```python3 Phase2/Code/test.py```. Use ```--help``` to see the adjustable parameters.

# Training

Run ```python3 Phase2/Code/train.py```. Use ```--help``` to see the adjustable parameters.

# Code Sources

3D Transform functions done with code from PyTorch 3D: https://github.com/facebookresearch/pytorch3d

IMU data generation done with code from Oystersim in ImuUtils.py: https://github.com/prgumd/Oystersim/blob/master/code/ImuUtils.py

