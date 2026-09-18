# cxr-engine
This repository contains model building engine for binary chest x-ray classification (normal vs pneumonia) using lightweight CNN model that combines depthwise separable convolution (building blocks of MobileNet) with residual connection (building blocks of ResNet).

---
## 1. Universal Setup
Clone the repository, create the virtual environment, and install dependencies:
```bash
# Clone repository
git clone https://github.com/Habiburr0hman/cxr-engine.git
cd cxr-engine

# Create and activate virtual environment
python -m venv pyenv

# Windows:
pyenv\Scripts\activate

# Linux/macOS:
source pyenv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---
## 2. Data Preparation
Image binaries are tracked via DVC and excluded from Git. Prepare the dataset using either of the following methods:

### Option A: Sync via DVC
If pulling from the public storage remote:
```bash
dvc pull
```

### Option B: Build from Raw Public Source
If building offline from the original Kermany et al. (2018) archive:
1. Download the raw images from [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3).
2. Extract the archive
3. Open extracted folder `rscbjbr9sj-3` and navigate into `rscbjbr9sj-3/ZhangLabData/CellData/chest_xray`
4. Move all images inside nested folders `chest_xray/{train,test}/{normal,pneumonia}`to single directory `data/raw/` inside this repo.
5. Generate the metadata catalog and preprocess the images:
```bash
python -m src.create_catalog
python -m src.preprocess
```

---
## 3. Local Mode Usage
Runs entirely on your local machine using SQLite. Requires zero accounts or credentials.

### Step 1: Set Configuration
Ensure `configs/config.yaml` points to local tracking:
```yaml
project:
  tracking_mode: "local"
```

### Step 2: Run Training
```bash
python -m src.train
```

### Step 3: View Results Locally
Launch the local MLflow dashboard:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open `http://localhost:5000` in any web browser.


## 4. Cloud Mode Usage
Logs parameters, metrics, model binaries, and visual artifacts to a shared DagsHub experiment board.

### Step 1: Configure Credentials
Create a `.env` file at the root of the project with your personal DagsHub credentials:
```bash
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_personal_access_token
```

### Step 2: Set Configuration
In `configs/config.yaml`, switch tracking to DagsHub:
```yaml
project:
  tracking_mode: "dagshub"
```

### Step 3: Run Training
```bash
python -m src.train
```

### Step 4: View Results Remotely
Open your browser and navigate to your DagsHub experiment repository:
`https://dagshub.com/<username>/<repo_name>.mlflow`


## 5. Local Artifact Structure
Regardless of tracking mode, all training runs produce self-contained artifacts isolated by run ID under `artifacts/runs/{run_id}/`:
* `checkpoints/`: Model binaries.
* `manifests/`: Catalog audit manifest and summary
* `visualization/`: Training history and evaluation metrics.