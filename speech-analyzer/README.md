# Speech analyzer

Extracts formants and speech speed from video

## Tech
python 3.8.6
with libraries:
* matplotlib 3.3.3
* scipy 1.6.0
* moviepy 1.0.3
* cv2 4.5.1

## Installation
```sh
pip3 install matplotlib
pip3 install scipy
pip3 install moviepy
pip3 install opencv-contrib-python
```

## Usage
Speech speed:
```sh
python3 speechSpeed.py video.mp4
```

To save formant data to formants.npy:
```sh
python3 formants.py video.mp4
```
And to plot formant frequency plot:
```sh
python3 load.py
```
