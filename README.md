# NeSysVideoPrediction
Neuro Symbolic Video Prediction/ Action Anticipation

```
conda create -n sgg python=3.7 pip
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt
```

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
