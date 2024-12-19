# CS433_Project_2

# Running a job on EPFL cluster


# Data Generation Pipeline

This guide provides an overview of the data generation pipeline and instructions for using it effectively. All relevant code is located in the `src/data_generation_pipeline` folder.

---

## Table of Contents
1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Running the Pipeline](#running-the-pipeline)
4. [Configuration Files](#configuration-files)
5. [Alternative Method](#alternative-method)

---

## Overview
The data generation pipeline is designed to generate data efficiently and save it to a specified folder. The main components of the pipeline include:

- **Main Script**: `data_generation.py` (entry point for the pipeline)
- **Utility Scripts**: `image_setup.py`, `image_generation.py`, `image_processing.py` (helper scripts for generating data)
- **Configuration Files**: `config_setup.json`, `config_process.json`, `config_characters.json` (modular configurations for customization)
- **Jupyter Notebook**: `Data_Generation.ipynb` (alternative data generation method for interactive use)

---

## Getting Started

1. **Prerequisites**:
   - Ensure you have Python installed (version >= 3.x).
   - Install the required dependencies using:
     ```bash
     pip install -r requirements.txt
     ```
   - For the CNN-BiLSTM model, you will need to install the following dependencies:
     ```bash
     pip install -r requirements_cnn-bilstm.txt
     ```
   - For the ViT model, you will need to install the following dependencies:
     ```conda install environment_vit.yml
     ```


2. **Setup**:
   - Navigate to the `src/data_generation_pipeline` folder:
     ```bash
     cd src/data_generation_pipeline
     ```
   - Review and modify the configuration files as needed (see [Configuration Files](#configuration-files)).

---

## Running the Pipeline

To generate data using the main script:

1. Run the `data_generation.py` script:
   ```bash
   python data_generation.py
   ```

2. The data will be generated and saved to the output folder specified in the configuration files.

---

## Configuration Files

The pipeline uses the following configuration files for customization:

- **`config_setup.json`**: Defines initial setup parameters for the data generation process.
- **`config_process.json`**: Specifies processing settings, such as transformations or filters.
- **`config_characters.json`**: Contains character-specific configurations for data generation.

To modify the configurations:
1. Open the respective JSON file in a text editor or IDE.
2. Update the values according to your requirements.
3. Save the changes before running the pipeline.

---

## Alternative Method

If you prefer an interactive approach, use the provided Jupyter notebook:

1. Open `Data_Generation.ipynb` in Jupyter Notebook or Jupyter Lab.
   ```bash
   jupyter notebook Data_Generation.ipynb
   ```

2. Follow the step-by-step instructions in the notebook to generate data.

3. The notebook provides a quick and flexible way to experiment with the pipeline.

---

# CNN-BiLSTM Training and Inference

## Training the model
You can configure some of the model parameters in the `src/models/cnn-bilstm/configs.py` file, including the `dataset_path` which should point to the location of the generated dataset.

To run the training script, make sure you are in the project root directory and run the following command:
```bash
python src/models/cnn-bilstm/train_torch.py 
```

The model will be saved in the `models/cnn-bilstm/` directory, in a folder named with the current timestamp.
To use the model for inference you should put that model folder name as the `MODEL_NAME` variable in the `src/models/cnn-bilstm/inferenceModel.py` script.

## Inference with the model
To run the inference script, make sure you are in the project root directory and run the following command:
```bash
python src/models/cnn-bilstm/inferenceModel.py 
```

The training and inference scripts store results using wandb, if you do not have an account set up with wandb, you can comment out the wandb related code in the scripts.

---

---

# ViT Training and Inference

## Training/Test/Validation the model

1. Install environment_vit.yml using conda
2. Use jupyter notebook: \VIT\main_pretrained.ipynb

---


```
├── environment_vit.yml
├── models
│   └── cnn-bilstm
│       └── best
│           ├── best_model.onnx
│           ├── best_model.pt
│           ├── configs.yaml
│           ├── error_analysis
│           │   ├── error_analysis.csv
│           │   └── preds_and_labels_padded.csv
│           ├── logs
│           │   ├── test
│           │   │   └── events.out.tfevents.1733556571.i01.796664.1
│           │   └── train
│           │       └── events.out.tfevents.1733556571.i01.796664.0
│           ├── model.pt
│           ├── train.csv
│           └── val.csv
├── README.md
├── requirements_cnn-bilstm.txt
├── requirements.txt
├── src
│   ├── data_generation_pipeline
│   │   ├── all_moves_proba.txt
│   │   ├── config_characters.json
│   │   ├── config_process.json
│   │   ├── config_setup.json
│   │   ├── Data_Generation.ipynb
│   │   ├── data_generation.py
│   │   ├── fonts
│   │   │   ├── aAccountantSignature.ttf
│   │   │   ├── Yellowtail-Regular.ttf
│   │   │   └── ylee Mortal Heart, Immortal Memory v.1.11 (TTF).ttf
│   │   ├── image_generation.py
│   │   ├── image_processing.py
│   │   ├── image_setup.py
│   │   ├── Pipeline_Test.ipynb
│   │   └── __pycache__
│   │       ├── image_generation.cpython-310.pyc
│   │       ├── image_generation.cpython-312.pyc
│   │       ├── image_generation.cpython-39.pyc
│   │       ├── image_processing.cpython-310.pyc
│   │       ├── image_processing.cpython-39.pyc
│   │       ├── image_setup.cpython-310.pyc
│   │       ├── image_setup.cpython-312.pyc
│   │       └── image_setup.cpython-39.pyc
│   ├── models
│   │   └── cnn-bilstm
│   │       ├── configs.py
│   │       ├── ctc_beam_search_decoder.py
│   │       ├── error_analysis.py
│   │       ├── inferenceModel.py
│   │       ├── model_architecture.txt
│   │       ├── model.py
│   │       ├── train_job.run
│   │       └── train_torch.py
│   └── utils
│       └── tests
└── VIT
    ├── architecture
    │   ├── main_scratch.ipynb
    │   ├── __pycache__
    │   │   └── transformer.cpython-310.pyc
    │   └── transformer.py
    ├── logs
    │   └── output_2024-12-19_0804.csv
    ├── main_pretrained.ipynb
    ├── saved
    ├── utils
    │   ├── config.py
    │   ├── datasets.py
    │   ├── model.py
    │   ├── plotting.py
    │   ├── __pycache__
    │   │   ├── config.cpython-310.pyc
    │   │   ├── datasets.cpython-310.pyc
    │   │   ├── model.cpython-310.pyc
    │   │   ├── plotting.cpython-310.pyc
    │   │   └── tokeniser.cpython-310.pyc
    │   └── tokeniser.py
    └── visualisation.ipynb

```
