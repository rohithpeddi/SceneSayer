# SceneSayer-Compatible Detector Training

This training path is intended to produce a DINOv3-backed torchvision Faster R-CNN
checkpoint that is easier to wrap for SceneSayer than the earlier resize-based
detector experiments.

## Files

- `ag_dataset_scenesayer.py`
  - loads Action Genome frames at original resolution
  - keeps GT boxes in original image coordinates
- `dinov3_scenesayer_frcnn.py`
  - builds a DINOv3 + simple FPN + torchvision Faster R-CNN model
- `train_scenesayer_detector.py`
  - trains and saves detector checkpoints with metadata needed for later bridging
- `detector_map.py`
  - computes dependency-light detection mAP / AP50 / AP75 on the validation split

## Why this is closer to SceneSayer

- uses original image coordinates instead of square-stretching every frame
- lets Faster R-CNN handle resize with `min_size=600`, `max_size=1000`
- saves object class names and detector metadata in the checkpoint

## Example

```bash
python train_scenesayer_detector.py \
  --data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome \
  --output_dir /home/cse/msr/csy227518/scratch/Projects/Active/scene_sayer_detector/checkpoints \
  --backbone_name facebook/dinov3-vitl16-pretrain-lvd1689m \
  --epochs 12 \
  --batch_size 2 \
  --eval_every 1 \
  --freeze_backbone
```

## Output checkpoint fields

Each checkpoint includes:

- `model_state_dict`
- `optimizer_state_dict`
- `scene_sayer_detector`
- `detector_family`
- `backbone_family`
- `backbone_name`
- `freeze_backbone`
- `num_classes`
- `object_classes`
- `transform_config`

These fields are meant to simplify a future SceneSayer wrapper that loads this
detector and reconstructs the ROI features, class distributions, and detection
outputs expected by the SceneSayer pipeline.

## Metrics printed during training

At each evaluation epoch, the trainer prints:

- `mAP` : mean AP over IoU thresholds 0.50:0.95
- `AP50`
- `AP75`
- `per_threshold_ap`
