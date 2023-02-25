import gradio as gr
from fastai.vision.all import *
import skimage

learner = load_learner("suchith_or_shivani.pth")

labels = learner.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, prob = learner.predict(img)
    return {labels[i]: prob[i].item() for i in range(len(labels))}

title = "Bewakoof Detector"
description = "Presenting the Bewakoof Detector - a tool designed to assess an individual's cognitive abilities. \
    Please utilize this tool to determine if you possess the characteristics of a bewakoof."
examples = ['suchith.jpg', 'shivani.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=2),
             title=title,description=description,article=article,examples=examples,
             interpretation=interpretation,enable_queue=enable_queue).launch()
