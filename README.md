## Download dataset
Credential is required to get access to the dataset. More details @ https://physionet.org/content/mimic-cxr/2.0.0/
Run a command to download dataset using the terminal: `wget -r -N -c -np --user jialong2 --ask-password https://physionet.org/files/mimic-cxr/2.0.0/`

## Processing data
`python preprocess.py`: csv files contain path to images and reports.  Merge multiple csv files and filter by some of our restrictions. Split data to train and test.

## Training
`python main.py`: run a training on DCGAN with default hyperparameter settings. Run with argument `--help` for more details.

## Evaluating
`python evaluate.py`: calculate FID score between fake images and real images.
