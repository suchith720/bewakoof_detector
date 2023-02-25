import gradio as gr
from fastai.vision.all import *
import skimage

learner = load_learner("nadal_or_shark.pth")

labels = learner.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, prob = learner.predict(img)
    return {labels[i]: prob[i].item() for i in range(len(labels))}

title = "Nadal vs Shark Classifier"
description = "We present a classifier that is designed to accurately distinguish between the tennis superstar \
Rafael Nadal and the aquatic animal known as the shark."
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' \
target='_blank'>Blog post</a></p>"
examples = ['nadal.jpg']
interpretation='default'
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=2),
             title=title,description=description,article=article,examples=examples,
             interpretation=interpretation,enable_queue=enable_queue).launch()
