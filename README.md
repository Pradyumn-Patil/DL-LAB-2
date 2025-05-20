# Imitation and Reinforcement Learning exercise for Deep Learning Lab 2025. 
Tested with python 3.7 and 3.8

Recommended virtual environments: `conda` or `virtualenv`.

Activate your virtual environment and install dependencies. Please don't skip the installation of pip, setuptools and wheel with the version we specified.
```[bash]
pip install --upgrade pip==21 setuptools==65.5.0 wheel==0.38.4 # Needed to install gym==0.22.0
pip install swig # Needs to be installed before requirements
pip install -r requirements.txt
```

Please format your code with `black .` before submission.

## Imitation Learning
Data Collection
```[bash]
python imitation_learning/drive_manually.py
```

Training
```[bash]
python imitation_learning/training.py
```

Testing
```[bash]
python imitation_learning/test.py
```
