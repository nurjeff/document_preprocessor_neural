from flask import Flask, json, request, Response
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2ForSequenceClassification, LayoutXLMTokenizer, LayoutXLMProcessor, logging
from PIL import Image
import torch
import os
import time


logging.set_verbosity_error()
api = Flask(__name__)

modelPath = "F:/ptmodel/ptmodel.pt"
engine = "cuda"
token_size = 512
print("Loading model:", modelPath, "on engine:", engine, "..")
model = LayoutLMv2ForSequenceClassification.from_pretrained(modelPath).to(engine)
feature_extractor = LayoutLMv2FeatureExtractor()
print("Loading tokenizer and processor..")
tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
processor = LayoutXLMProcessor(feature_extractor, tokenizer)
dataset_path = "./train_data"
labels = [label for label in os.listdir(dataset_path)]
id2label = {v: k for v, k in enumerate(labels)}

@api.route('/image', methods=['POST'])
def get_companies():
  file = request.files['image']
  img = Image.open(file.stream)
  img = img.convert("RGB")
  with torch.no_grad():
    print("Tokenizing..")
    tokenizeTimeStart = time.time()
    encoded_inputs = processor(img, return_tensors="pt", max_length=token_size, padding="max_length", truncation=True).to(engine)
    tokenizeTimeFin = time.time()
    print("Inferencing..")
    inferencingTimeStart = time.time()
    outputs = model(**encoded_inputs)
    inferencingTimeFin = time.time()
    p = torch.nn.functional.softmax(outputs.logits, dim=1)
    maxval = p[0].cpu().detach().numpy().max()
    logits = outputs.logits
    id = logits.argmax(-1).item()
    result = json.dumps({"class_id": id, "class": id2label[id], "confidence": str(maxval * 100), "tokenizer_time": (tokenizeTimeFin-tokenizeTimeStart), "inference_time": (inferencingTimeFin-inferencingTimeStart)})
    r = Response(response=result, status=200, mimetype="application/json")
    r.headers["Content-Type"] = "application/json; charset=utf-8"
    print("Responding:", result)
    return r
    



if __name__ == '__main__':
    from waitress import serve
    print("Running webservice..")
    serve(api, host="0.0.0.0", port=5000)








