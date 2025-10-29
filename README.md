# 🧬 MolHuiTu — Molecular HyperGraph V8.1
### Intelligent Drug–Target Interaction (DTI) Prediction Platform
# MolHuiTu

MolHuiTu (Molecular Intelligence Graph) is a web-based platform for drug-target interaction (DTI) prediction. It leverages advanced deep learning models (including pre-trained protein language models like ProtBERT) to analyze both small molecule and protein target data, providing predictions and interactive visualizations. MolHuiTu is designed to help researchers quickly evaluate potential drug-target interactions with an intuitive interface and high-performance backend (optimized for NVIDIA GPUs).

**MolHuiTu Overview:**  
<img width="3548" height="1652" alt="MolHuiTu Overview" src="https://github.com/user-attachments/assets/0bf60f5b-a63f-4708-9910-d043bc655497" />  
*_(Figure: Overall architecture of MolHuiTu, illustrating the flow from input data to prediction results.)_*

## Features

- **Drug-Target Interaction Prediction:** Predict potential interactions between drug molecules (ligands) and protein targets. Supports single query predictions and batch processing of multiple queries.
- **Interactive Web Interface:** User-friendly front-end for submitting predictions and viewing results. The interface includes input forms for molecules and protein sequences, and dashboards to track job status.
- **Batch Job Management:** Easily submit multiple DTI prediction tasks in batch. The system queues and processes tasks asynchronously, allowing monitoring of each task's status in real time.
- **Detailed Reports:** For each prediction, MolHuiTu generates a comprehensive report including predicted interaction scores and visualization of molecular structures. Results can be viewed in the browser or downloaded for further analysis.
- **High Performance with GPU Acceleration:** MolHuiTu's backend is optimized to utilize GPU acceleration (tested on NVIDIA RTX 4090) for faster model inference. CPU and GPU usage can be monitored during runtime to ensure efficient resource utilization.

**MolHuiTu Web Interface – Homepage:**  
<img width="3172" height="1582" alt="image" src="https://github.com/user-attachments/assets/185000bc-4b54-4178-81e6-f7050db1f3cf" />  
*_(Screenshot: The MolHuiTu front-end main page, providing navigation to single prediction and batch submission sections.)_*

## Architecture Overview

MolHuiTu consists of a backend machine learning inference engine and a front-end web interface:

- **Backend:** The core prediction engine is built with Python. It uses a pre-trained protein language model (ProtBERT) to encode protein sequences and chemical informatics techniques to encode molecular structures. A deep learning model then predicts the interaction likelihood between each drug-target pair. The backend exposes a RESTful API (e.g., endpoints for submitting prediction tasks and checking their status).
- **Front-End:** A lightweight web interface (HTML/JavaScript, utilizing libraries like 3Dmol.js for molecular visualization) communicates with the backend via API calls. Users can input data and view results through this interface. The front-end displays 3D molecular structures and provides real-time updates on batch job progress.

## Prerequisites

Before installing MolHuiTu, ensure you have the following:

- **Operating System:** Ubuntu 24.04 LTS (or a similar Linux distribution). The guide assumes a fresh Ubuntu 24.04 server environment.
- **GPU:** An NVIDIA GPU is recommended for acceleration (MolHuiTu has been tested with an NVIDIA GeForce RTX 4090). Ensure your NVIDIA drivers are properly installed. CUDA toolkit is optional if using the Conda environment (which can provide its own CUDA libraries).
- **Python & Conda:** Python 3.x (the project is tested with Python 3.10+). Install [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) to manage the project environment.
- **Memory:** Sufficient RAM for loading models (e.g., ProtBERT) and VRAM on the GPU for inference (the RTX 4090 with 24GB VRAM is used in our example).
- **Disk Space:** Adequate space for storing any downloaded models and output files.

## Installation

Follow these steps to set up MolHuiTu on a new Ubuntu 24.04 server:

1. **Update System and Install Git:**  
   It’s good practice to update your system first and install Git if not already available:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y git
   ```
2. **Clone the MolHuiTu Repository:**  
   Choose a directory to install the application and clone the GitHub repository:
   ```bash
   git clone https://github.com/yourusername/molhuitu.git
   cd molhuitu
   ```
3. **Setup Conda Environment:**  
   Create a Conda environment for MolHuiTu to manage dependencies. An environment YAML (e.g. `environment.yml` or `transferconda.yml`) is provided with all required packages:
   ```bash
   # If the environment file is named transferconda.yml
   conda env create -f transferconda.yml -n molhuitu
   ```
   This will create a new environment named `molhuitu` with all necessary dependencies (including deep learning frameworks like PyTorch, and any other libraries).
4. **Activate the Environment:**  
   Once the environment is created, activate it:
   ```bash
   conda activate molhuitu
   ```
5. **Install Additional Dependencies (if any):**  
   If there are any missing dependencies or if you prefer using pip, install them from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *_(Note: The `requirements.txt` may be empty or minimal if all dependencies are covered in the Conda environment file. This step can be skipped if the environment is fully set up.)_*
6. **Download/Prepare Models:**  
   MolHuiTu may require pre-trained model files (such as the ProtBERT model for protein embedding). If these are not bundled in the repository, you should download them:
   - Ensure the `protbert_model/` directory is populated with the necessary model weights. If not, the first run of the application might download the ProtBERT model from Hugging Face or another source automatically. Make sure the server has internet access for this step.
7. **Configuration (Optional):**  
   MolHuiTu uses configuration files (possibly via Hydra). Check the `hydra/` or `config/` directory for configuration options. Default settings should work out-of-the-box, but advanced users can adjust parameters like model thresholds, batch sizes, etc., by editing config files or using environment variables.

## Usage

With the environment set up, you can now run the MolHuiTu application and perform predictions.

### Starting the Backend Server

Start the MolHuiTu backend server which handles the predictions. Depending on how the project is structured, this could be done via a provided launch script or a direct Python command. For example:
```bash
# Example: if there's a script or module to run the web server
python src/app.py
```
Or, if using a framework like FastAPI with Uvicorn:
```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```
After running the above, you should see the server starting up, loading models into memory, and listening on a port (e.g. 8000). Ensure that this port is accessible if you are using a remote server.

### Accessing the Web Interface

Once the backend is running, open a web browser and navigate to the MolHuiTu interface. If you’re running locally, use:  
```
http://localhost:8000
```  
(Adjust the port if needed, and use the server’s IP or domain if accessing remotely.)

You should see the MolHuiTu home page with options for different prediction modes.

**MolHuiTu Drug-Target Prediction Interface:**  
<img width="2814" height="1492" alt="image" src="https://github.com/user-attachments/assets/b400ecd4-d50b-4c79-9718-c95243c61ac3" />  
*_(Screenshot: The interface for Drug-Target Interaction prediction, where users can input a molecule (SMILES or structure file) and a protein sequence or identifier.)_*

#### Single Prediction

For a single DTI prediction:

1. Navigate to the **Single Prediction** section of the interface.
2. Input the required data:  
   - **Drug Molecule:** Provide the molecule, typically as a SMILES string, a MOL file, or an identifier of the compound. The interface may also allow drawing the structure or uploading a file.  
   - **Target Protein:** Provide the protein information, either as an amino acid sequence (FASTA format) or a known identifier (such as a UniProt ID).
3. Click the **Submit** button to start the prediction.

After submission, the task will be sent to the backend for processing. You will see a status indicator for the job.

**Single Prediction Submission Example:**  
<img width="2646" height="1404" alt="image" src="https://github.com/user-attachments/assets/45176290-8fe9-4349-95f0-428bec62b5da" />  
*_(Screenshot: A single prediction entry form filled with a sample drug and target, ready to be submitted.)_*

While the prediction is running, MolHuiTu will indicate that the task is in progress.

**Single Prediction In Progress:**  
<img width="2248" height="868" alt="image" src="https://github.com/user-attachments/assets/f356af59-b01f-4e61-84ba-44877d8b384f" />  
*_(Screenshot: The interface showing a single prediction task in progress. Users are advised to wait as the model computes the results.)_*

Once the prediction is complete, a result report will be available for viewing and download.

#### Batch Prediction

MolHuiTu also supports batch processing, allowing you to run multiple predictions in one go:

1. Go to the **Batch Prediction** section.
2. Prepare an input file (CSV format) with each row representing a drug-target pair. For example, the repository provides a `batch_template.csv` as a template:  
   *Each row might contain columns such as Drug_SMILES (or compound ID) and Target_Sequence (or target ID).*
3. Upload the CSV file through the interface (or as instructed on the page).
4. Submit the batch job. The interface will queue all tasks and start processing them one by one on the backend.

After submitting, you'll see a list of tasks with their statuses (e.g. queued, running, completed).

**Batch Task Submission Example:**  
<img width="1416" height="394" alt="image" src="https://github.com/user-attachments/assets/1a96e41f-dd1a-4231-8f16-be7045243fd4" />  
*_(Screenshot: A batch submission form where a CSV file has been selected for upload.)_*

While the batch is running, you can monitor the progress of each task in real-time. Each job will update its status from "pending" to "running" to "completed" (or "failed" if an issue occurs).

**Batch Tasks Status Dashboard:**  
<img width="2920" height="752" alt="image" src="https://github.com/user-attachments/assets/371e45a6-ef83-43d2-a9b2-6675680ccb30" />  
*_(Screenshot: Batch task list showing multiple tasks and their current status. Completed tasks have results available and links to view reports.)_*

When all tasks are finished, you can review the results for each pair. You may also download a consolidated results file (e.g. a CSV similar to `batch_template.pred.csv` with added prediction outcomes for all input pairs).

### Viewing Results and Reports

For each completed prediction (single or batch), MolHuiTu provides a detailed report. A report typically includes:

- The input details (drug and target, with identifiers or sequence).
- The predicted interaction score or probability (indicating how likely the drug is to interact with the target).
- Visualizations of the molecular structure of the drug (and possibly the target, if structural data is available or relevant).
- Additional information such as confidence metrics, similarity to known compounds, target annotations, etc.

**Example DTI Prediction Report:**  
<img width="2870" height="1250" alt="image" src="https://github.com/user-attachments/assets/73de69af-97b0-49d7-a709-ba364b5899c9" />  
*_(Screenshot: Part of a DTI prediction report, listing the input details and the predicted interaction score among other details.)_*

Large reports may contain multiple sections, possibly including tables of results or interactive components.

**Continuation of DTI Report:**  
<img width="1064" height="568" alt="image" src="https://github.com/user-attachments/assets/914e7efe-a150-4e7d-8411-cf0d78e0cb7e" />  
*_(Screenshot: Another section of the report, possibly showing additional metrics or a summary of results.)_*

Reports can be viewed in the web interface and are also saved to the server (e.g., in the `outputs/` directory) for future reference or downloading.

### Monitoring Performance

MolHuiTu is designed to utilize system resources efficiently. You can monitor CPU and GPU usage during execution to ensure that the application is making use of hardware acceleration:

- **CPU Monitoring:** Use `htop` or `top` in the terminal to observe CPU cores utilization. The backend will use CPU for data preprocessing and coordinating tasks.
- **GPU Monitoring:** Use `nvidia-smi` to watch GPU memory and compute utilization. When a prediction is running, you should see GPU memory usage and compute activity, indicating that the model is running on the GPU.

Below are example outputs showing CPU and GPU usage during MolHuiTu operation:

<img width="2026" height="600" alt="image" src="https://github.com/user-attachments/assets/184e0fc4-4d8a-498f-a039-9d8e0f3e7b99" />  
<img width="594" height="288" alt="image" src="https://github.com/user-attachments/assets/af0e8d3c-aad1-43c9-951c-e161d0fac141" />  
*_(Screenshots: Terminal output of `htop` (top image) showing CPU usage across cores, and `nvidia-smi` (bottom image) showing the GPU (RTX 4090) memory and utilization during a batch inference.)_*

Monitoring these resources can help in understanding performance. For instance, you can verify that the GPU is fully utilized during heavy computations. If the GPU is underutilized, you might consider increasing batch sizes or running multiple tasks in parallel (if supported) to better leverage the hardware.

## Contributing

Contributions to MolHuiTu are welcome. If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request on GitHub. When contributing, follow the project’s coding style and include relevant tests or examples to demonstrate your changes.

## License

This project is released under the **[License Name]**. See the [LICENSE](./LICENSE) file for details.

---

By following this guide, you should be able to deploy and run MolHuiTu on a fresh Ubuntu 24.04 server (with GPU support). We hope this tool accelerates your research in drug discovery and bioinformatics by providing quick and accurate predictions of drug-target interactions. **Happy researching!**
