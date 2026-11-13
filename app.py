import gradio as gr
import spaces
import os
from core.predict import ImageClassifier
from PIL import Image

cwd = os.getcwd()
model_path = os.path.join(cwd, "model", "cnn_model.pth")
class_names = {0: 'Dog', 1: 'Cat', 2: 'Person'}
classifier = ImageClassifier(model_path=model_path, class_names=class_names)

@spaces.GPU
def classify_image(image):
    os.makedirs("./images", exist_ok=True)
    image_path = "./images/uploaded_image.jpg"
    image.save(image_path)
    label, output_path = classifier.predict(image_path) # return 2 variables
    return label, Image.open(output_path)

demo = gr.Interface(fn=classify_image, 
                    inputs=gr.Image(type='pil'), 
                    outputs=[gr.Textbox(label="Prediction"),   # output1: label
                             gr.Image(label="Labeled image")], # output2: image
                    title="Image Classification Gradio App",
                    description="Upload an image to classiy it as Dog, Cat or Person")

if __name__ == "__main__":
    demo.launch()