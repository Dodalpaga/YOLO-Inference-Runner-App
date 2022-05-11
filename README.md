# Table of Contents

- [Table of Contents](#table-of-contents)
- [Project layout](#project-layout)
- [Create (and activate) a Virtual Environment](#create-and-activate-a-virtual-environment)
- [Requirements](#requirements)
- [Launch the server](#launch-the-server)
- [How it works ...](#how-it-works-)
  - [Inference Setion](#inference-setion)
  - [Webcam Section](#webcam-section)
  - [Documentation](#documentation)

# Project layout

```
📦YOLO-Object-Detection-Template
┣ 📦components
┃ ┣ 📂Test
┃ ┃ ┣ 📂Images_predites
┃ ┃ ┃ ┗ 📜.gitkeep
┃ ┃ ┗ 📂Prediction
┃ ┃ ┃ ┗ 📜.gitkeep
┃ ┗ 📜weights.pt
┣ 📦static
┃ ┣ 📂css
┃ ┃ ┣ 📜colorful.css
┃ ┃ ┣ 📜index.css
┃ ┃ ┣ 📜interface.css
┃ ┃ ┗ 📜notebook.css
┃ ┗ 📂images
┃ ┣ 📜logo.png
┃ ┗ 📜tensorboard.png
┣ 📦templates
┃ ┣ 📜doc.html
┃ ┣ 📜index.html
┃ ┣ 📜interface.html
┃ ┣ 📜live_streaming.html
┃ ┣ 📜training.html
┃ ┗ 📜training_notebook.html
┗ 📦YOLO
  ┣ 📂mAP
  ┃ ┣ 📂.temp_files
  ┃ ┣ 📂input
  ┃ ┃ ┣ 📂detection-results
  ┃ ┃ ┃ ┗ 📂backup
  ┃ ┃ ┣ 📂ground-truth
  ┃ ┃ ┃ ┗ 📂backup
  ┃ ┃ ┗ 📂images
  ┃ ┣ 📂output
  ┃ ┃ ┗ 📂classes
  ┃ ┣ 📂scripts
  ┃ ┃ ┗ 📂extra
  ┃ ┃ ┃ ┣ 📜README.md
  ┃ ┃ ┃ ┣ 📜class_list.txt
  ┃ ┃ ┃ ┣ 📜convert_dr_darkflow_json.py
  ┃ ┃ ┃ ┣ 📜convert_dr_perso_yolo.py
  ┃ ┃ ┃ ┣ 📜convert_dr_yolo.py
  ┃ ┃ ┃ ┣ 📜convert_gt_xml.py
  ┃ ┃ ┃ ┣ 📜convert_gt_yolo.py
  ┃ ┃ ┃ ┣ 📜convert_keras-yolo3.py
  ┃ ┃ ┃ ┣ 📜find_class.py
  ┃ ┃ ┃ ┣ 📜intersect-gt-and-dr.py
  ┃ ┃ ┃ ┗ 📜result.txt
  ┃ ┣ 📜LICENSE
  ┃ ┣ 📜README.md
  ┃ ┗ 📜main.py
  ┣ 📂yolov5
  ┃ ┣ 📂data
  ┃ ┣ 📂images
  ┃ ┣ 📂models
  ┃ ┣ 📂runs
  ┃ ┃ ┗ 📂train
  ┃ ┃ ┃ ┣ 📂exp1
  ┃ ┃ ┃ ┃ ┣ 📂weights
  ┃ ┃ ┃ ┃ ┃ ┣ 📜best.pt
  ┃ ┃ ┃ ┃ ┃ ┗ 📜last.pt
  ┃ ┃ ┃ ┃ ┣ 📜events.out.tfevents.1651744720.1ca71d3adcdf.367.0
  ┃ ┃ ┃ ┃ ┣ 📜hyp.yaml
  ┃ ┃ ┃ ┃ ┣ 📜labels.jpg
  ┃ ┃ ┃ ┃ ┣ 📜labels_correlogram.jpg
  ┃ ┃ ┃ ┃ ┣ 📜opt.yaml
  ┃ ┃ ┃ ┃ ┣ 📜results.csv
  ┃ ┃ ┃ ┃ ┣ 📜train_batch0.jpg
  ┃ ┃ ┃ ┃ ┣ 📜train_batch1.jpg
  ┃ ┃ ┃ ┃ ┗ 📜train_batch2.jpg
  ┃ ┃ ┃ ┗ 📂exp2
  ┃ ┃ ┃ ┃ ┣ 📂weights
  ┃ ┃ ┃ ┃ ┃ ┣ 📜best.pt
  ┃ ┃ ┃ ┃ ┃ ┗ 📜last.pt
  ┃ ┃ ┃ ┃ ┣ 📜events.out.tfevents.1651492827.80c1f15bf8d7.289.0
  ┃ ┃ ┃ ┃ ┣ 📜hyp.yaml
  ┃ ┃ ┃ ┃ ┣ 📜labels.jpg
  ┃ ┃ ┃ ┃ ┣ 📜labels_correlogram.jpg
  ┃ ┃ ┃ ┃ ┣ 📜opt.yaml
  ┃ ┃ ┃ ┃ ┣ 📜results.csv
  ┃ ┃ ┃ ┃ ┣ 📜train_batch0.jpg
  ┃ ┃ ┃ ┃ ┣ 📜train_batch1.jpg
  ┃ ┃ ┃ ┃ ┗ 📜train_batch2.jpg
  ┃ ┣ 📂utils
  ┣ 📜Convert.py
  ┣ 📜Evaluate.py
  ┣ 📜Inference.py
  ┗ 📜Training.ipynb
```

# Create (and activate) a Virtual Environment

In order to create a new venv, type the following in a terminal :
`python3 -m venv /path/to/new/virtual/environment/`

Then, activate it so you can install the dependencies :
`source /path/to/new/virtual/environment/bin/activate`

# Requirements

Once the venv is activated, install the python dependencies
`pip install -r requirements.txt` - Install requirements

# Launch the server

To launch the flask server, type in a terminal :
`python3 app.py`

# How it works ...

## Inference Setion

## Webcam Section

## Documentation
