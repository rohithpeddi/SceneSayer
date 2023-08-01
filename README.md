# NeSysVideoPrediction
Neuro Symbolic Video Prediction/ Action Anticipation

```
conda create -n sgg python=3.7 pip
conda activate sgg

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```


# Dataset Preparation 

### Estimated time: 10 hours

Follow the instructions from [here](https://github.com/JingweiJ/ActionGenome)

Download Charades videos ```data/ag/videos```

Download all action genome annotations ```data/ag/annotations```

Dump all frames ```data/ag/frames```

#### Change the corresponding data file paths in ```datasets/action_genome/tools/dump_frames.py```


# Setup

### Build draw_rectangles modules

```
cd lib/draw_rectangles
```
Remove any previous builds
```
rm -rf build/
rm -rf *.so
rm -rf *.c
rm -rf *.pyd
```
Build the module
```
python setup.py build_ext --inplace
cd ..
```
Add the path to the current directory to the PYTHONPATH

```
conda develop draw_rectangles/
```

### Build bbox modules

```
cd fpn/box_intersections_cpu
```
Remove any previous builds
```
rm -rf build/
rm -rf *.so
rm -rf *.c
rm -rf *.pyd
```
Build the module
```
python setup.py build_ext --inplace
cd ..
```
Add the path to the current directory to the PYTHONPATH

```
conda develop fpn/box_intersections_cpu/
```

# fasterRCNN model

Remove any previous builds

``` 
cd fastRCNN/lib
rm -rf build/
```

Change the folder paths in 'fasterRCNN/lib/faster_rcnn.egg.info/SOURCES.txt' to the current directory

```
python setup.py build develop
```

If any errors check gcc version ```Works for 9.x.x```

Follow [this](https://www.youtube.com/watch?v=aai42Qp6L28) for changing gcc version
