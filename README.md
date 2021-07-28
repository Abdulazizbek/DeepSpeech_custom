# DeepSpeech (minimized) for custom project
Code is used from official implementation of [DeepSpeech](https://github.com/mozilla/DeepSpeech)

## Version: 0.9.3

## 1. Recommended to create a new virtual environment and activate:
`conda create --name ds_env python --no-default-packages`

## 2. Download desired version
(Used 0.9.3 semantic versioning of DeepSpeech)

`git clone --branch v0.9.3 https://github.com/mozilla/DeepSpeech`

## 3. Install required packages
`pip3 install --upgrade pip wheel setuptools`

`pip3 install numpy progressbar2 six pyxdg attrdict absl-py semver opuslib==2.0.0 optuna sox bs4 pandas requests numba llvmlite librosa soundfile ds_ctcdecoder==0.9.3 tensorflow-gpu==1.15.4 deepspeech-gpu==0.9.3 deepspeech-tflite==0.9.3 webrtcvad`

## 4. Edit Flags and start training
`training/util/flags.py`

## 5. Documentation  
Detailed information for installation, usage and training models are available on [deepspeech.reedthedocs.io](https://deepspeech.readthedocs.io/en/r0.9/?badge=latest)
