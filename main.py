from PIL import Image
import pytesseract
import numpy as np
from datasets import Features, Sequence, ClassLabel, Value, Array2D, Array3D, Dataset
import pandas as pd
import os
from transformers import LayoutLMv2FeatureExtractor, LayoutLMv2ForSequenceClassification, LayoutXLMTokenizer, LayoutXLMProcessor, logging
import torch, gc

logging.set_verbosity_error()
reference_path = "./test3.jpg"

image = Image.open(reference_path)
print("\nLoaded reference image: " + reference_path + "\n")
image = image.convert("RGB")

ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
ocr_df = ocr_df.dropna().reset_index(drop=True)
float_cols = ocr_df.select_dtypes('float').columns
ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
words = ' '.join([word for word in ocr_df.text if str(word) != 'nan'])

feature_extractor = LayoutLMv2FeatureExtractor()
#tokenizer = LayoutLMv2Tokenizer.from_pretrained("microsoft/layoutlmv2-base-uncased")
tokenizer = LayoutXLMTokenizer.from_pretrained("microsoft/layoutxlm-base")
processor = LayoutXLMProcessor(feature_extractor, tokenizer)

token_size = 512
encoded_inputs = processor(image, return_tensors="pt", max_length=token_size, padding="max_length", truncation=True)

dataset_path = "./train_data"
labels = [label for label in os.listdir(dataset_path)]
id2label = {v: k for v, k in enumerate(labels)}
label2id = {k: v for v, k in enumerate(labels)}

images = []
labels = []

#
print("\nTraining Categories:")
print("-----------------------\n")
for label_folder, _, file_names in os.walk(dataset_path):
  if label_folder != dataset_path:
    label = label_folder[len(dataset_path)+1:]
    for _, _, image_names in os.walk(label_folder):
      relative_image_names = []
      for image_file in image_names:
        relative_image_names.append(dataset_path + "/" + label + "/" + image_file)
      images.extend(relative_image_names)
      print("[" +str(len(image_names)) + "] " + label)
      labels.extend([label] * len (relative_image_names)) 
print("\n")


data = pd.DataFrame.from_dict({'image_path': images, 'label': labels})

dataset = Dataset.from_pandas(data)

labels = list( dict.fromkeys(labels) )

features = Features({
    'image': Array3D(dtype="int64", shape=(3, 224, 224)),
    'input_ids': Sequence(feature=Value(dtype='int64')),
    'attention_mask': Sequence(Value(dtype='int64')),
    'token_type_ids': Sequence(Value(dtype='int64')),
    'bbox': Array2D(dtype="int64", shape=(512, 4)),
    'labels': ClassLabel(num_classes=len(labels), names=labels),
})


def preprocess_data(examples):
  images = [Image.open(path).convert("RGB") for path in examples['image_path']]
  encoded_inputs = processor(padding="max_length", truncation=True, images=images, max_length=token_size, return_token_type_ids=True)
  encoded_inputs["labels"] = [label2id[label] for label in examples["label"]]

  return encoded_inputs

batch_size = 1
print("Converting training data to tensors..\n")
encoded_dataset = dataset.map(preprocess_data, remove_columns=dataset.column_names, features=features, 
                              batched=True, batch_size=batch_size)


#für CUDA relevant, getestet auf RTX 2080 Super 8GB, bei weniger VRAM eventuell auf cpu umstellen
processing_units = "cuda" # "cpu"

encoded_dataset.set_format(type="torch", device=processing_units)
cuda_split_size = 30
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:' + str(cuda_split_size)
gc.collect()
torch.cuda.empty_cache()

# Nur für GPUs mit Tensor Cores (ab RTX 2XXX), bei 'nur' CUDA eventuell deaktivieren
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dataloader = torch.utils.data.DataLoader(encoded_dataset, batch_size=batch_size)
batch = next(iter(dataloader))

device = torch.device(processing_units)
model = LayoutLMv2ForSequenceClassification.from_pretrained("microsoft/layoutxlm-base", 
                                                            num_labels=len(labels)).to(processing_units)

# lernrate funktioniert multilingual ganz gut, bei nur englisch eventuell vergrößern
learn_rate = 5e-5
optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

global_step = 0
num_train_epochs = 10
t_total = len(dataloader) * num_train_epochs

print("\n\nSetup:\n-------\nLearning rate: "+str(learn_rate)+"\nBatch Size: "+str(batch_size)+"\nCuda Split Size: "+str(cuda_split_size)+"MB\n")
print("\nStarting neural net training with " + processing_units + " loaded tensors..")
print("-----------------------")
model.train()
for epoch in range(num_train_epochs):
  print("\nEpoch:", epoch, "/", num_train_epochs)
  running_loss = 0.0
  correct = 0
  for batch in dataloader:
      # forward pass
      outputs = model(**batch)
      loss = outputs.loss

      running_loss += loss.item()
      predictions = outputs.logits.argmax(-1)
      correct += (predictions == batch['labels']).float().sum()

      # backward pass
      loss.backward()

      # update
      optimizer.step()
      optimizer.zero_grad()
      global_step += 1
  
  print("Loss:", running_loss / batch["input_ids"].shape[0])
  accuracy = 100 * correct / len(data)
  print("Training accuracy:", accuracy.item())

encoded_inputs = processor(image, return_tensors="pt", max_length=token_size, padding="max_length", truncation=True)

for k,v in encoded_inputs.items():
  encoded_inputs[k] = v.to(model.device)

# forward pass
outputs = model(**encoded_inputs)

p = torch.nn.functional.softmax(outputs.logits, dim=1)
maxval = p[0].cpu().detach().numpy().max()
print("Confidence:" + str(maxval * 100) + "%")
#model.save_pretrained("./model")
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", id2label[predicted_class_idx])