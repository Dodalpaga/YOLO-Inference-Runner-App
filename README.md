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
┃ ┃ ┗ 📂Images_predites
┃ ┃   ┗ 📜.gitkeep
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
┣ 🐍app.py
┣ 📜README.md
┗ 📜requirements.txt
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
