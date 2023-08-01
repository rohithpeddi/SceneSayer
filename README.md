# NeSysVideoPrediction
Neuro Symbolic Video Prediction/ Action Anticipation

conda create -n sgg python=3.7 pip
conda install -c anaconda _libgcc_mutex
pip install -r requirements.txt



# Setup

cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace

# Download the FastRCNN model


