# NeuralNet Document Classifier

Trains a model capable of classifying any document using some Huggingface transformers and LayoutXLM multi-language weights.

Requires:
- PyTorch
- Huggingface bindings
- Tesseract installation as well as PyTesseract bindings for testing purposes

plus some pip dependencies.

To train the model, create a **/train_data** subdirectory on main.py dir level containing folders representing the classifier labels, e.G:

```
/invoices
 - invoice_train1.jpg
 - invoice_train2.png
 - invoice_train3.tif
/vehicle_registrations
 - vehicle_registration_train1.jpg
 - ...

and so forth
```

and run main.py.
This will put out a model capable of being loaded and inferenced from by loader.py, which spins up a simple webserver serving the classified response in a form of:
```json
{
    "class": "vehicle_registration",
    "class_id": 15,
    "confidence": "99.7821569442749",
    "inference_time": 0.055999755859375,
    "tokenizer_time": 1.834416389465332
}
```

~
