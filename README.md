# Visual Question Answering (VQA)

This project explores three distinct approaches to Visual Question Answering (VQA), a task that combines computer vision and natural language processing to answer questions about images. The implemented methods are:
1. **CNN + LSTM**: A convolutional neural network (CNN) for image processing paired with a long short-term memory (LSTM) network for question understanding.
2. **ViT + RoBERTa**: A Vision Transformer (ViT) for image encoding combined with RoBERTa for robust text processing.
3. **Vision Language Model (VLM)**: A unified model designed to handle both vision and language inputs.

## Project Structure

### Main Folders
- **`data/`**: Contains the processed dataset, including train, validation, and test splits.
  - **Original Dataset**: Available at [Google Drive](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view?usp=sharing).
- **`image/`**: Stores prediction results and loss plots generated during training and evaluation.
- **`model/`**: Contains implementations of the three VQA approaches.
  - **`read_data.py`**: Script for loading and preprocessing the dataset.
  - **`CNN_LSTM/`**: Directory for the CNN + LSTM model.
    - `VQA_dataset.py`: Dataset handling for VQA tasks.
    - `data_loader.py`: Data loading utilities.
    - `modelPipeline.py`: Training and inference pipeline.
    - `CNN_LSTM_model.py`: Base CNN + LSTM model implementation.
  - **`ViT_Roberta/`**: Directory for the ViT + RoBERTa model.
    - `VQA_dataset.py`: Dataset handling for VQA tasks.
    - `data_loader.py`: Data loading utilities.
    - `modelPipeline.py`: Training and inference pipeline.
    - `ViT_Roberta_model.py`: Base ViT + RoBERTa model implementation.
  - **`Vision_Language_Model/`**: Directory for the Vision Language Model.
    - `VLM_model.py`: Implementation of the VLM approach.
  - **`evaluation.py`**: Script for evaluating model performance.

### Key Files
- **`requirements.txt`**: Lists all dependencies required to run the project.
- **`constants.py`**: Configuration file defining directory paths and other constants.
- **`LICENSE`**: Licensing information for the project and dataset.

<div style="text-align: center;">
    <img src="https://github.com/DANGKHOIk22/Project-Visual-Question-Answering/blob/main/image/VQA_FLOW.png?raw=true" width="700"/>
</div>
