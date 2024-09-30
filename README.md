<h1 align=center>
  Towards Scene Graph Anticipation
</h1>

<p align=center>  
  Rohith Peddi, Saksham Singh, Saurabh, Parag Singla, Vibhav Gogate
</p>

<div align=center>
  <a src="https://img.shields.io/badge/project-website-green" href="">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://arxiv.org/pdf/2403.04899v1.pdf">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div>

<p align="center">
  (This page is under continuous update)
</p>

----

## UPDATE

**Sep 30th 2024:** Released pre-trained models and updated code.

**Aug 2024:** Our submission is accepted as an **Oral Presentation** in ECCV.

**July 2024:** Towards Scene Graph Anticipation was accepted at ECCV 2024.

**Mar 2024:** Released code for the [paper](https://arxiv.org/pdf/2403.04899v1.pdf) 


----

<h3 align=center>
  TASK PICTURE
</h3>

![TaskPicture](https://github.com/rohithpeddi/SceneSayer/assets/23375299/cd5a7092-7b4f-4711-8835-c6a1ff621162)

----

<h3 align=center>
  TECHNICAL APPROACH
</h3>


![TechnicalApproach](https://github.com/rohithpeddi/SceneSayer/assets/23375299/43bda602-a9ab-4846-9501-51e2ba4474ad)


-------
### ACKNOWLEDGEMENTS

This code is based on the following awesome repositories. 
We thank all the authors for releasing their code. 

1. [STTran](https://github.com/yrcong/STTran)
2. [DSG-DETR](https://github.com/Shengyu-Feng/DSG-DETR)
3. [Tempura](https://github.com/sayaknag/unbiasedSGG)
4. [TorchDiffEq](https://github.com/rtqichen/torchdiffeq)
5. [TorchDyn](https://github.com/DiffEqML/torchdyn)


-------
# SETUP

## Dataset Preparation 

**Estimated time: 10 hours**

Follow the instructions from [here](https://github.com/JingweiJ/ActionGenome)

Download Charades videos ```data/ag/videos```

Download all action genome annotations ```data/ag/annotations```

Dump all frames ```data/ag/frames```

#### Change the corresponding data file paths in ```datasets/action_genome/tools/dump_frames.py```


Download object_bbox_and_relationship_filtersmall.pkl from [here](https://drive.google.com/file/d/19BkAwjCw5ByyGyZjFo174Oc3Ud56fkaT/view)
and place it in the data loader folder

### Install required libraries

```
conda create -n sga python=3.7 pip
conda activate sga
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

If there are any errors, check gcc version ``` Works for 9.x.x```


Follow [this](https://www.youtube.com/watch?v=aai42Qp6L28) for changing gcc version


Download pretrained fasterRCNN model [here](https://utdallas.box.com/s/gj7n57na15cel6y682pdfn7bmnbbwq8g) and place in fasterRCNN/models/


------

## CHECKPOINTS

| Method Name     | Checkpoint method Name |
|-----------------|------------------------|
| STTran+         | sttran_ant  |
| DSGDetr+        | dsgdetr_ant  | 
| STTran++        | sttran_gen_ant | 
| DSGDetr++       | dsgdetr_gen_ant | 
| SceneSayerODE   | ode | 
| SceneSayerSDE   | sde | 


Please find the checkpoints with the following name structure.

```
<CKPT_METHOD_NAME>_<MODE>_future_<#TRAIN_FUTURE_FRAMES>_epoch_<#STORED_EPOCH>.tar
```
Eg:

**dsgdetr_ant**\_**sgdet**\_future\_**3**\_epoch\_**0**

**ode**\_**sgdet**\_future\_**1**\_epoch\_**0**

## Settings

### Action Genome Scenes [AGS] (~sgdet)

Download the required checkpoints from [here](https://utdallas.box.com/s/g94v2zfgxkxfgcs68q31lkf3olg6p7wy)

### Partially Grounded Action Genome Scenes [PGAGS] (~sgcls)

Download the required checkpoints from [here](https://utdallas.box.com/s/mvdwz8fe1ct9q8l1pv6ndi0wi7bkbl8r)

### Grounded Action Genome Scenes [GAGS] (~predcls)

Download the required checkpoints from [here](https://utdallas.box.com/s/9xncf5498o4nvqkjzpjp268gajmhiygo)

------

# Instructions to run

Please see the scripts/training for Python modules.

Please see the scripts/tests for testing Python modules.

------


# Citation

```
@misc{peddi2024scene,
      title={Towards Scene Graph Anticipation}, 
      author={Rohith Peddi and Saksham Singh and Saurabh and Parag Singla and Vibhav Gogate},
      year={2024},
      eprint={2403.04899},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```



