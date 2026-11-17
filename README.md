# Emotion Classification Studio

## Project structure

```text
Emotion-Classification/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── LSTM/
│   ├── GRU/
│   ├── BiLSTM_Attention/
│   └── DistilBERT/
│
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── predictor.py
│
└── results/
```

Local Gradio application for the GoEmotions-based Emotion Classification project. It supports four exported models:
LSTM, GRU, BiLSTM + Attention, and DistilBERT.

The main objective is to classify user-written text into six emotion categories:

- Joy
- Sadness
- Anger
- Fear
- Surprise
- Disgust

The application loads the trained models exported by the Kaggle notebook. It does not train or fine-tune models locally.

## What the app provides

1. **Emotion prediction**: enter free text and receive the dominant predicted emotion.
2. **Confidence scores**: view the model's confidence score across all six emotion classes.
3. **Model selection**: choose any available exported model from the dropdown.
4. **Attention visualization**: view token-level attention weights when using BiLSTM + Attention.

The app lists only model folders that contain the required export files. It does not invent results for models that are missing.

## 1. Project workflow: training is separate from the application

**Model training and the local application are separate stages.** The local application does not train, fine-tune, or modify any model. It loads the trained model files exported by the Kaggle notebook and uses them for emotion prediction.

| Part | Where it runs | Responsibility |
|---|---|---|
| Model training | Kaggle Notebook | Load and prepare GoEmotions, map the source labels to six target emotions, train LSTM, GRU, BiLSTM + Attention, and DistilBERT, evaluate the models, and export model files and results. |
| Local application | Windows / VS Code | Load the exported models, classify user-entered text, display confidence scores, and visualize BiLSTM attention weights. |

### Dataset

The project uses the [GoEmotions dataset by Google](https://huggingface.co/datasets/google-research-datasets/go_emotions).

The notebook maps the source labels to the following six target classes:

`joy`, `sadness`, `anger`, `fear`, `surprise`, and `disgust`.

The notebook keeps examples that map to exactly one of the six target classes.

### Download the trained models from Kaggle

Use the links below to open the notebook and access its published output:

- **[Open Kaggle Notebook – Emotion Classification](https://www.kaggle.com/code/badrhishamzahran/emotion-classification)**
- **[Open Kaggle Notebook Output – trained models and result files](https://www.kaggle.com/code/badrhishamzahran/emotion-classification/output)**

After the notebook has completed successfully, download the generated output archive:

`Emotion-Classification-Clean.zip`

The notebook creates the final deliverables under:

`/kaggle/working/emotion_project/`

The ZIP archive is created at:

`/kaggle/working/Emotion-Classification-Clean.zip`

The archive contains the exported `models/` and `results/` folders, together with `README_export.json`. Temporary checkpoints and Hugging Face cache files are stored separately and are not included in the final archive.

#### Copy the exported files into the local project

1. Open the completed Kaggle notebook.
2. Open the notebook's output/files area and download `Emotion-Classification-Clean.zip`.
3. Extract the archive on your computer.
4. Copy the extracted `models/` directory into the root of this local project.
5. Copy the extracted `results/` directory into the root of this local project.

The resulting structure should look like this:

```text
Emotion-Classification/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── LSTM/
│   │   ├── model.pth
│   │   ├── config.json
│   │   ├── vocab.json
│   │   └── label_mapping.json
│   │
│   ├── GRU/
│   │   ├── model.pth
│   │   ├── config.json
│   │   ├── vocab.json
│   │   └── label_mapping.json
│   │
│   ├── BiLSTM_Attention/
│   │   ├── model.pth
│   │   ├── config.json
│   │   ├── vocab.json
│   │   └── label_mapping.json
│   │
│   └── DistilBERT/
│       ├── config.json
│       ├── model.safetensors (or pytorch_model.bin)
│       ├── tokenizer files
│       └── label_mapping.json
│
└── results/
    ├── model_comparison.csv
    ├── classification_report.csv
    ├── model_comparison.png
    ├── confusion_matrices.png
    └── training_history.png
```

Keep the model folder names exactly as shown. The application looks for these names when it builds the model-selection dropdown.

**Important:** The model weights are not included in the source-code repository by default. You must copy the actual exports from the Kaggle output before using the application. You only need to run model training in Kaggle when creating or updating the checkpoints.

## 2. Install on Windows

Install Python 3.10 or 3.11 (64-bit), open this folder in VS Code, then run the following commands in the VS Code terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this once in that terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\.venv\Scripts\activate
```

For a CUDA-enabled PyTorch setup, use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose the command matching your CUDA version. CPU mode also works, but prediction may be slower.

## 3. Launch

From the project root directory, run:

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal. The default address is usually:

`http://127.0.0.1:7860`

## 4. Application interface

- **Model Selection**: choose LSTM, GRU, BiLSTM + Attention, or DistilBERT, depending on which model exports are present.
- **Text Input**: enter a sentence or short paragraph for emotion classification.
- **Predicted Emotion**: displays the dominant emotion predicted by the selected model.
- **Confidence Scores**: displays the confidence score for each of the six emotion classes.
- **Attention Visualization**: displays token-level attention weights for BiLSTM + Attention. This visualization is not provided for LSTM, GRU, or DistilBERT in the current application.

## Important methodology notes

- The recurrent models use the vocabulary exported by the notebook. The application uses the same lowercase regular-expression tokenization pattern and vocabulary lookup for inference.
- The recurrent model exports contain `model.pth`, `config.json`, `vocab.json`, and `label_mapping.json`.
- DistilBERT is loaded from its complete Hugging Face export directory using `AutoModelForSequenceClassification` and `AutoTokenizer`.
- The BiLSTM + Attention model produces one attention weight per valid input token. The application visualizes these weights for the submitted text.
- LSTM and GRU do not contain an attention layer in the current notebook architecture. The application does not claim to display attention weights for these models.
- DistilBERT uses transformer self-attention internally, but the current application does not extract or visualize its internal attention maps.
- The recurrent embedding layer is initialized as a trainable embedding layer. The current notebook does not initialize it with pre-trained GloVe or FastText vectors.
- Confidence scores are model probabilities produced by applying softmax to the output logits. They are not a substitute for a labeled evaluation metric.
- Accuracy, Precision, Recall, Macro F1, and Weighted F1 are calculated during notebook evaluation using the held-out test set. They are saved in `results/model_comparison.csv` and `results/classification_report.csv`.
- The local free-text interface does not know the true label of a user's text. Therefore, it cannot calculate real Precision, Recall, or F1 for an individual prediction.
- Prediction speed depends on the local machine, device, model, and software versions.

## 5. Upload the project to GitHub

You can publish the application source code on GitHub so that others can view or clone the project. The instructions below use Git from the VS Code terminal.

### Before uploading

1. Create or sign in to your account at [GitHub](https://github.com/).
2. Check the project's `.gitignore` file. Do not upload:
   - `.venv/`, `__pycache__/`, `.env`, or local machine-specific files.
   - Kaggle API tokens, passwords, access tokens, or private keys.
   - Large model checkpoints unless you intentionally want to publish them and have checked their size and licensing.
3. GitHub blocks ordinary Git files larger than 100 MB. Model weights can be large. A practical approach is to keep model checkpoints in the Kaggle notebook output and document how to download them, as described above. If you need to version large weights, consider [Git LFS](https://git-lfs.com/) and review its storage and bandwidth limits.
4. Make sure you have permission to publish the dataset, model files, and any other included assets. Do not upload the original GoEmotions dataset unless its terms permit your intended redistribution.

### Step 1 — Create a new GitHub repository

1. Open [GitHub](https://github.com/) and click **New repository**.
2. Enter a repository name, for example `Emotion-Classification-Studio`.
3. Add an optional description, such as `Gradio application for six-class emotion classification using LSTM, GRU, BiLSTM Attention, and DistilBERT`.
4. Choose **Public** if you want anyone to see the code, or **Private** if you want to restrict access.
5. For the simplest first push, leave **Add a README**, **.gitignore**, and **license** unchecked because this project already contains its own README and files.
6. Click **Create repository**. Keep the repository page open; GitHub will show its remote URL.

### Step 2 — Open the project in VS Code

Open the root `Emotion-Classification` folder in VS Code. Open **Terminal → New Terminal** and confirm that the terminal is in the folder containing `app.py` and `requirements.txt`.

### Step 3 — Initialize Git and make the first commit

Run these commands one at a time in the VS Code terminal:

```powershell
git init
git add .
git status
git commit -m "Initial commit: Emotion Classification Studio"
```

Review the output of `git status` before committing. If you see `.venv`, secrets, Kaggle credentials, or model files you do not intend to publish, stop and fix `.gitignore` before proceeding.

If Git asks you to identify yourself, configure the name and email you want associated with your commits:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

Then run the `git commit` command again.

### Step 4 — Connect the local project to GitHub and push

Replace `YOUR-USERNAME` with your GitHub username. Use the HTTPS URL shown on your newly created repository page:

```powershell
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/Emotion-Classification-Studio.git
git push -u origin main
```

GitHub may ask you to authenticate in your browser or through Git Credential Manager. Follow the sign-in prompt. Do not put your password or personal access token directly into the remote URL or README.

After the push succeeds, refresh the GitHub repository page. Your project files should appear there.

### Updating GitHub after making changes

Whenever you modify the project, run:

```powershell
git add .
git status
git commit -m "Describe your changes"
git push
```

Use a short commit message that explains the update, for example:

```powershell
git commit -m "Improve emotion prediction interface"
```

### Optional — Upload through the GitHub website

For a small project or a few individual files, you can also:

1. Open your GitHub repository.
2. Select **Add file → Upload files**.
3. Drag the files or folders into the upload area.
4. Enter a commit message and select **Commit changes**.

The website upload method is less convenient for repeated updates and may not support very large files. Use Git from VS Code for the normal development workflow.

### Recommended repository contents

Publish the application source, documentation, requirements, and small example files. Keep private credentials and large model artifacts out of the repository unless you intentionally plan to distribute them.

```text
Emotion-Classification-Studio/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── model.py
│   └── predictor.py
├── models/
│   └── (downloaded model exports; usually excluded from Git)
└── results/
    └── (exported evaluation reports and plots)
```

If the app requires model weights to run, explain in the README how to obtain them from the Kaggle notebook output. You can optionally create a GitHub Release and attach permitted model artifacts, provided each file meets GitHub's current release upload limits.

Useful references:
- [GitHub Docs — Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [GitHub Docs — About large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files)
- [Git LFS](https://git-lfs.com/)

## Troubleshooting

- **No models appear in the dropdown**: copy the exported model folders into `models/` and verify that each folder contains the required files.
- **Recurrent model fails to load**: check that `model.pth`, `config.json`, `vocab.json`, and `label_mapping.json` are all present in the model folder.
- **DistilBERT fails to load**: ensure the complete `DistilBERT` folder was copied, including `config.json`, tokenizer files, model weights (`model.safetensors` or `pytorch_model.bin`), and `label_mapping.json`.
- **PyTorch checkpoint error**: install a compatible PyTorch version and keep the model architecture settings consistent with the exported `config.json`.
- **Gradio does not start**: confirm that the virtual environment is activated and that `pip install -r requirements.txt` completed successfully.
- **`git` is not recognized**: install [Git for Windows](https://git-scm.com/download/win), restart VS Code, and reopen the terminal.
- **`git push` is rejected**: confirm that the remote URL is correct and that you have permission to push. If the GitHub repository was created with an initial README or other files, follow GitHub's instructions to reconcile the remote history before pushing.
