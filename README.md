# AMP Scanner Version 2

Antimicrobial Peptide Scanner Version 2. Open source GLPv3 release of code from 2018 paper "Deep learning improves
antimicrobial peptide recognition" published in the journal Bioinformatics: https://doi.org/10.1093/bioinformatics/bty179.

<em>NOTE: While best efforts have been made to ensure the integrity of this script, we take no
responsibility for damages that may result from its use!</em>

---
Quick Links:
 - [Prerequisites](#prerequisites)
 - [Installation](#installation)
 - [Making Predictions with Pre-Trained Models](#making-predictions-with-pre-trained-models)
 - [Training Your Own Model](#training-your-own-model)
 - [Running with Docker](#running-with-docker)

---

## Prerequisites

### For the Original Paper Model
 - Python==3.6
 - Tensorflow==1.2.1
 - Keras==2.0.6
 - numpy==1.16.0
 - h5py==2.6
 - biopython>=1.69
 - scikit-learn>=0.20.0

### For the 2019 or Newer Server Models
 - Python==3.6
 - Tensorflow==1.12.0
 - Keras==2.2.4
 - numpy=1.16.0
 - h5py==2.8.0
 - biopython>=1.69
 - scikit-learn>=0.20.1

### Protein Sequence Requirements:

Protein sequences <em>must</em> be provided in FASTA format (see: https://en.wikipedia.org/wiki/FASTA_format) and all be
&ge; 10 amino acids in length. It is also highly recommended to only use sequences &le; 200 amino acids in length. 
All sequences must consist of the following amino acid characters: `XACDEFGHIKLMNPQRSTVWY`. See the Paper and the
[AMP Scanner website](https://www.dveltri.com/ascan/v2/about.html) for details. If you need to consider longer sequences, change the
`max_length` argument passed to the scripts but know the models were not trained on sequences that long (it may be best
to train a new model for your data). You can also add an `X` character to the front/end of shorter sequences to force
the code to work, but these predictions may not be accurate and this is not recommended.

---

## Installation

These scripts require Python v3.6 and user the older Tensorflow v1.x - most system now have newer versions installed so
we highly recommend using either a conda or virtual environment to install the packages needed to run everything.
Note, that the package versions are slightly different depending on if you are using the pertained model from the
original paper or newer (2019+) pre-trained models.

### Using Conda/Miniconda
To use conda/miniconda (see: https://docs.anaconda.com/distro-or-miniconda/) you can use the included "environment.yml"
files to easily install the environments.

For the original paper model, install and activate with:

```bash
conda env create -f environment_original_paper_model.yml
conda activate ascan2_orig
```

For the 2019, or newer, models install and activate with:

```bash
conda env create -f environment_2019_and_newer_model.yml
conda activate ascan2_tf1
```

When finished, deactivate an environment with:

```bash
conda deactivate
```

### Using PIP and Virtual Environments

To use Python 3.6 with PIP (see: https://pypi.org/project/pip/) and Virtual Environments
(see: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) you can use the
appropriate "requirements.txt" file to install the needed libraries in a similar fashion:

For the original paper model follow these steps:

#### Create Virtual Environment Using Python v3.6.x
```bash
python3.6 -m venv ascan2_orig # point to python3.6 location for your system
```

#### Activate Virtual Environment
```bash
source ascan2_orig/bin/activate # On Linux/MacOS
ascan2_orig\Scripts\activate # On Windows Command Prompt
.\ascan2_orig\Scripts\Activate.ps1 # On Windows PowerShell
```

#### Install Dependencies and Verify Installation
```bash
pip install --upgrade pip
pip install -r requirements_original_paper_model.txt
pip freeze
```

#### Deactivate After Finished Running AMP Scanner Scripts
```bash
deactivate
```

For the 2019 or newer models replace `ascan2_orig` with `ascan2_tf1` and use the
`requirements_2019_and_newer_model.txt` requirements file.

You may need admin (sudo) access on your system to install conda some packages manually. However, you may also be able
to install them locally, See: https://stackoverflow.com/questions/7465445/how-to-install-python-modules-without-root-access

<em>**Unfortunately, we can't provide support for getting packages installed**</em>.
However, the TensorFlow and Keras communities are very active in helping users with problems.
Please see: https://www.tensorflow.org/install/ for TensorFlow and https://keras.io/#installation for Keras-related issues.

### Testing Installation Worked Correctly 

If you wish to test to see if the installation worked correctly you can run the following from the <em>parent directory</em>
(do not run from within the `tests` folder):

```bash
python -m unittest discover -s tests
```

If things work correctly, you should see an `OK` at the end of the scripts. If you installed the `ascan2_orig` environment
you will see a warning that the `ascan2_tf1` (2019+) environment will not work (or vice-versa). This is expected and can
be ignored.

---

## Making Predictions with Pre-Trained Models

The `amp_scanner_v2_predict_tf1.py` script is used to run predictions on your own query FASTA file using a pre-trained
AMP Scanner v2 model file (saved in .h5 format). It saves two results files, an "`AMP_Candidates.fasta`" FASTA file with
the sequences predicted to be AMPs, and an "`AMP_Predictions.csv`" CSV file with prediction results for each sequence in
your input FASTA file. Below is a basic example and further details on available command-line
flags.

### Basic Usage with Provided Original Dataset AMP Test File and Original Trained Model:
```bash
python amp_scanner_v2_predict_tf1.py \
  -fasta original-dataset/AMP.te.fa \
  -model trained-models/OriginalPaper_081917_FULL_MODEL.h5 \
  -candidates My_AMP_Candidates.fasta \ 
  -preds My_AMP_Predictions.csv
```
This will save `My_AMP_Candidates.fasta` and `My_AMP_Prediction.csv` in the current directory. 
<em>Note, if you installed the 2019+ environment (see above) switch to a newer model file like
`trained-models/020419_FULL_MODEL.h5`.</em>

### Command-Line Flags

#### Required Flags
| Flag                                     | Argument | Description                                                                                                                              | Example          |
|------------------------------------------|----------|------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| `-f` / `-fasta` / `-q` / `--query_fasta` | `<path>` | Path to the **input FASTA file** containing protein sequences. Refer to the README for acceptable formats and amino acid characters.     | `-f input.fasta` |
| `-m` / `-model` / `--model_file`         | `<path>` | Path to the **TensorFlow model file** (in `.h5` HDF5 format). Ensure compatibility with the TensorFlow version used in your environment. | `-m model.h5`    |

#### Optional Flags
| Flag                                                  | Argument    | Description                                                                                                                                                                                   | Default                               | Example                  |
|-------------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------|--------------------------|
| `-c` / `-candidates` / `--candidate_amp_fasta_output` | `<path>`    | Path and filename for the **output FASTA file** containing predicted AMP candidates. If not provided, the filename is generated using the input sequence basename.                            | `<input_prefix>_AMP_Candidates.fasta` | `-c my_candidates.fasta` |
| `-p` / `-preds` / `--predictions_csv_output`          | `<path>`    | Path and filename for the **output CSV file** containing predictions and probabilities for each sequence. If not provided, the filename is generated using the input sequence basename.       | `<input_prefix>_AMP_Predictions.csv`  | `-p my_predictions.csv`  |
| `-l` / `--max_length`                                 | `<int>`     | Maximum sequence length allowed. The `model_file` **must** have been trained on this same size! Sequences longer than this value are flagged, and prediction reliability may be questionable. | `200`                                 | `-l 250`                 |
| `-t` / `-thrsh` / `--threshold_cutoff`                | `<int>`     | Probability threshold for AMP classification. Predictions with a probability **greater than this value** are classified as AMPs, while lower probabilities are non-AMPs.                      | `0.5`                                 | `-t 0.7`                 |
| `-ns` / `-nosave` / `--skip_output_files`             | No argument | **Flag to skip saving output files.** Predictions are printed directly to the standard output (STDOUT).                                                                                       | `False`                               | `-ns`                    |
| `-nt` / `-notime` / `--skip_timing`                   | No argument | **Flag to skip reporting start and end times.** Suppresses printing of timing information to the standard output.                                                                             | `False`                               | `-nt`                    |

#### **Help**
Use the `-h` or `--help` flag to view the usage instructions and flag descriptions.
```bash
python amp_scanner_v2_predict_tf1.py -h
```

---

## Example Commands

#### Basic Prediction with Default Output Files
```bash
python amp_scanner_v2_predict_tf1.py -f input.fasta -m model.h5
```
Predicts AMPs in the `input.fasta` file using the `model.h5` TensorFlow model. Outputs:
- A FASTA file with AMP candidates (`input_AMP_Candidates.fasta`)
- A CSV summary of all predictions (`input_AMP_Predictions.csv`).

#### Customizing Output Filenames
```bash
python amp_scanner_v2_predict_tf1.py -f input.fasta -m model.h5 -c my_candidates.fasta -p my_predictions.csv
```

#### Skipping File Output - Prints Output to STDOUT
```bash
python amp_scanner_v2_predict_tf1.py -f input.fasta -m model.h5 -ns
```

#### Custom Prediction Threshold Cutoff and Sequence Length
```bash
python amp_scanner_v2_predict_tf1.py -f input.fasta -m model.h5 -t 0.7 -l 250
```

---

## Training Your Own Model

The `amp_scanner_v2_train_tf1.py` script is used to train and evaluate an AMP prediction model using your own AMP
and Decoy FASTA files. These should be split into separate training, (optional) validation, and testing FASTA files
for AMPs, and Decoys, respectively. Below is a basic example and further details on available command-line flags.

### Basic Usage with Provided Original Dataset Files:
```bash
python amp_scanner_v2_train_tf1.py \
  --amp_train_fasta original-dataset/AMP.tr.fa \
  --amp_validate_fasta originial-dataset/AMP.eval.fa \
  --amp_test_fasta original-dataset/AMP.te.fa \
  --decoy_train_fasta original-dataset/DECOY.tr.fa \
  --decoy_validate_fasta original-dataset/DECOY.eval.fa \
  --decoy_test_fasta original-dataset/DECOY.te.fa \
  --output_model_name my_ascan2_model.h5
```

### Command-Line Flags

##### **Required Flags**
| Flag                          | Argument | Description                            | Example                  |
|-------------------------------|----------|----------------------------------------|--------------------------|
| `-atr`/ `--amp_train_fasta`   | `<path>` | Path to **AMP training FASTA** file.   | `-atr train_amp.fasta`   |
| `-ate`/ `--amp_test_fasta`    | `<path>` | Path to **AMP test FASTA** file.       | `-ate test_amp.fasta`    |
| `-dtr`/ `--decoy_train_fasta` | `<path>` | Path to **Decoy training FASTA** file. | `-dtr train_decoy.fasta` |
| `-dte`/ `--decoy_test_fasta`  | `<path>` | Path to **Decoy test FASTA** file.     | `-dte test_decoy.fasta`  |

##### **Optional Flags**
| Flag                                    | Argument    | Description                                                                                    | Default           | Example                       |
|-----------------------------------------|-------------|------------------------------------------------------------------------------------------------|-------------------|-------------------------------|
| `-ava`/ `--amp_validate_fasta`          | `<path>`    | Path to optional AMP **validation FASTA** file.                                                | `None`            | `-ava validation_amp.fasta`   |
| `-dva`/ `--decoy_validate_fasta`        | `<path>`    | Path to optional Decoy **validation FASTA** file.                                              | `None`            | `-dva validation_decoy.fasta` |
| `-o`/ `-out`/ `--output_model_name`     | `<path>`    | Output file name for the trained model (HDF5 `.h5`).                                           | `ascan2_model.h5` | `-o model_output.h5`          |
| `-n`/ `-nosave`/ `--skip_output_model`  | No argument | Skip saving the output model. Used for evaluation only.                                        | `False`           | `-n`                          |
| `-m`/ `-merge`/ `--merge_train_and_val` | No argument | Merge the training and validation datasets for combined training.                              | `False`           | `-m`                          |
| `-s`/ `-seed`/ `--shuffle_seed`         | `<int>`     | Seed for shuffling training and validation data for reproducibility.                           | `123`             | `-s 42`                       |
| `-r`/ `-reprod`/ `--make_reproducible`  | No argument | Ensure determinism by forcing the use of a single-threaded CPU. **Warning: Slow performance!** | `False`           | `-r`                          |

#### **Help**
Use the `-h` or `--help` flag to view the usage instructions and flag descriptions.

---

### Example Commands

#### Training with Required Parameters - No Validation Set
```bash
python amp_scanner_v2_train_tf1.py \
  -atr train_amp.fasta \
  -ate test_amp.fasta \
  -dtr train_decoy.fasta \
  -dte test_decoy.fasta
```

#### Training with Validation and Saved Output Model
```bash
python amp_scanner_v2_train_tf1.py \
  -atr train_amp.fasta \
  -ate test_amp.fasta \
  -ava validation_amp.fasta \
  -dtr train_decoy.fasta \
  -dte test_decoy.fasta \
  -dva validation_decoy.fasta \
  -o output_model.h5
```

#### Evaluation-Only Mode (No Output Model Saved - Just Prints Results)
```bash
python amp_scanner_v2_train_tf1.py \
  -atr train_amp.fasta \
  -ate test_amp.fasta \
  -dtr train_decoy.fasta \
  -dte test_decoy.fasta \
  -n
```

#### Reproducible Training with Merged Dataset
```bash
python amp_scanner_v2_train_tf1.py \
  -atr train_amp.fasta \
  -ate test_amp.fasta \
  -ava validate_amp.fasta \
  -dtr train_decoy.fasta \
  -dte test_decoy.fasta \
  -dva validate_decoy.fasta \
  -m \
  -r
```

---
### A Note on Training Reproducibility

Because these scripts were developed using the older Tensorflow vr 1.x backend - training a model on a multicore (or GPU)
machine (even using the same random seed) on the original datasets will result in <em>slightly</em> different results from
that of the paper (but should still be in the ballpark of the standard deviations of the 10-fold cross-validation experiments
in the paper). A Google search of this issue will bring back a lot of discussions on the topic. I believe the multicore
reproducibility problem has finally been addressed with Tensorflow 2.x, but you will need to update the code and packages
to make things compatible. For more information on this please see: 
https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development.

If you are training a new model and reproducibility is critical for you, we can force TF v1.x to use a single CPU thread
and set random seeds by calling the script with the `-r` flag described above. Note, especially for larger datasets, this
will run <em>very slow!</em>

---

## Running with Docker

If you prefer using Docker, prebuilt images or Dockerfiles are available to build locally.
You must have Docker installed on your system to use this, see: https://www.docker.com/.
In the container, AMP Scanner V2 scripts, original datasets, and pretrained models are stored under `/app` which is set
as the default working directory. We recommend mapping local input/output files to `/data`.

### Pull Images from Docker Hub

To pull an image for use with the original paper Tensorflow v1 model run:
```bash
docker pull dveltri/ascan2:orig
```

For an image for use with the 2019 or newer Tensorflow v1 server models run:
```bash
docker pull dveltri/ascan2:tf1
```

### Building Images Locally

Separate Docker files are provided in the project `docker` folder to locally build an image for the original paper and/or
2019+ model environments outlined above. From within the `docker` folder, run this to build the original paper environment:
```bash
docker build -f Dockerfile.orig_paper_tf1 -t ascan2:orig .
```

Or, for the 2019+ newer model environment:
```bash
docker build -f Dockerfile.newer_models_tf1 -t ascan2:tf1 .
```

### Examples of Running Scripts with a Docker Image 

Examples for how you can run the AMP Scanner Version 2 scripts are shown below. Call the script arguments as outlined
above. These examples map the present working directory `$(pwd)` (change to your desired path) on the local system to
the directory `/data` in the container. 

#### Predict Using Included FASTA and Model Files - Save Results in Current Working Directory
```bash
docker run -v $(pwd):/data ascan2:orig python amp_scanner_v2_predict_tf1.py \
  -fasta original-dataset/AMP.te.fa \
  -model trained-models/OriginalPaper_081917_FULL_MODEL.h5 \
  -candidates /data/AMP_Candidates.fasta \
  -preds /data/AMP_Predictions.csv
```

#### Make Predictions on File `my_query.fasta` Located in Present Working Directory - Save Results in Current Working Directory
```bash
docker run -v $(pwd):/data ascan2:tf1 python amp_scanner_v2_predict_tf1.py \
  -fasta /data/my_query.fasta \
  -model trained-models/020419_FULL_MODEL.h5 \
  -candidates /data/My_Query_AMP_Candidates.fasta \
  -preds /data/My_Query_AMP_Predictions.csv
```

#### Train Using Provided Example Files - Save Model in Current Working Directory
```bash
docker run -v $(pwd):/data ascan2:tf1 python amp_scanner_v2_train_tf1.py \
  --amp_train_fasta original-dataset/AMP.tr.fa \
  --amp_validate_fasta originial-dataset/AMP.eval.fa \
  --amp_test_fasta original-dataset/AMP.te.fa \
  --decoy_train_fasta original-dataset/DECOY.tr.fa \
  --decoy_validate_fasta original-dataset/DECOY.eval.fa \
  --decoy_test_fasta original-dataset/DECOY.te.fa \
  --output_model_name /data/my_ascan2_model.h5
```

