# Image Style Transfer - CycleGAN vs. NST

#### Requirements
Before running the code, you should install all neccessary dependencies (you can also use `conda`):
```bash
pip install torch torchvision
pip install tqdm
pip install numpy
pip install matplotlib
pip install gradio
```

#### Model files
CycleGAN model files are available at links:
- Portraits: https://drive.google.com/file/d/1ZANuuR44nl-fDAk1m0gsNSrtcxn02dJn/view?usp=sharing
- Landscapes: https://drive.google.com/file/d/1lDm7DelR42LBdwDBvuP62152LQs8lnZd/view?usp=sharing

Download them and change the path in `inference.py` and `training.py` scripts.

#### Training
For CycleGAN training, run command in folder root:
```bash
python cycleGAN/training.py
```

#### Inference
For CycleGAN inference, run command in folder root (command with args will be added in future):
```bash
python cycleGAN/inference.py
```
For NST, run command in folder root:
```bash
python nst/inference.py --content_image_path path/to/your/content/image --style_image_path path/to/your/style/image --output_path path/to/your/saved/generated/image
```

#### UI demo
Basic UI demo was built using Gradio. To lunch it, run command in folder root:
```bash 
python main.py 
```

#### Dataset
Dataset is not currently available, but you can test model and get the results with your own images just by running `nst/inference.py` or `cycleGAN/inference.py`.

#### Documentation
Research poster, paper and presentation are available in upper files.

#### Updates coming soon...
