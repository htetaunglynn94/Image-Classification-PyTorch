# Image Classification using PyTorch

A deep learning image classification application built with **PyTorch** and **Gradio**. Upload an image to the web application to classify it as one of the following categories:

-   Dog
-   Cat
-   Person

The trained CNN model is loaded by the application to generate a prediction and return a labeled image.



## Live Demo

Try the deployed application on Hugging Face Spaces:

**[Launch Web App](https://huggingface.co/spaces/htetaunglynn/Image-Classification-PyTorch)**

## Project Links

  | Resource | Link |
  |----------|------|
  | GitHub Repository| [LINK](https://github.com/htetaunglynn94/Image-Classification-PyTorch) |
  | Hugging Face Space | [LINK](https://huggingface.co/spaces/htetaunglynn/Image-Classification-PyTorch) |
  | Training Dataset | [LINK](https://drive.google.com/file/d/1FZO4ojM03tP1lTaiVHShzDX5oMDTi3oZ/view?usp=drive_link)

  
## Video Demonstration                 
See my [demonstration](https://drive.google.com/file/d/1T5E5ozpX5RjpXIXj9mPYDB_QaRmV38Q6/view?usp=sharing) as video.


## Features

-   Upload an image through a Gradio web interface.
-   Classify the uploaded image into Dog, Cat, or Person.
-   Display the predicted class.
-   Return a labeled image as the prediction output.
-   Run inference using a trained PyTorch CNN model.

## Model Training

The CNN model was trained using **Google Colab with an NVIDIA T4 GPU**.

- **Framework**: PyTorch
- **Model type**: Convolutional Neural Network (CNN)
- **Training environment**: Google Colab
- **GPU accelerator**: NVIDIA T4
- **Output classes**: Dog, Cat, Person

The training dataset is available through the Google Drive link in the Project Links section.

## Repository Structure

``` text
Image-Classification-PyTorch/
├── core/
│   └── predict.py
├── model/
│   └── cnn_model.pth
├── notebook/
│   └── training notebook(s)
├── app.py
├── requirements.txt
└── README.md
```

> The exact contents of the `notebook/` directory may vary as the
> project is updated.

## Tech Stack

- **Python** --- application development
- **PyTorch** --- deep learning model and inference
- **Torchvision** --- computer vision utilities
- **OpenCV** --- image processing
- **Pillow (PIL)** --- image loading and handling
- **Gradio** --- interactive web application
- **Google Colab** --- model training environment
- **Hugging Face Spaces** --- application deployment

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/htetaunglynn94/Image-Classification-PyTorch.git
cd Image-Classification-PyTorch
```

### 2. Create and activate a virtual environment

``` bash
python -m venv .venv
```

On Linux or macOS:

``` bash
source .venv/bin/activate
```

On Windows:

``` powershell
.venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Run the application

``` bash
python app.py
```

The application will print a local Gradio URL in the terminal. Open that
URL in your browser.

> Note: The application currently uses the Hugging Face Spaces `spaces`
> package and its GPU decorator. If you run it outside Hugging Face
> Spaces, you may need to remove the `import spaces` line and the
> `@spaces.GPU` decorator from `app.py` for local inference.

## How to Use

1.  Open the [live web application](https://huggingface.co/spaces/htetaunglynn/Image-Classification-PyTorch).
2.  Upload an image containing a dog, cat, or person.
3.  Wait for the model to process the image.
4.  View the predicted class and the labeled image.


## Author

**Htet Aung Lynn**

-   [LinkedIn](https://www.linkedin.com/in/htetaglynn)
-   [Portfolio](https://htetaunglynn94.github.io/hal-portfolio/)

------------------------------------------------------------------------

If you find this project useful, feel free to explore the repository and
try the live application.
