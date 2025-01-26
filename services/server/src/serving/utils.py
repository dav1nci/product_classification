import torch
import numpy as np

def predict_on_batch(tokenizer, model, batch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_ids = tokenizer(batch,
              max_length=128,
              padding='max_length',
              truncation=True,
              return_tensors="pt").to(device)
    preds = model(**input_ids)
    return torch.argmax(preds.logits, dim=1)


def process_in_batches(tokenizer, model, elements, batch_size):
    """
    Iterates through a list in batches of size batch_size.

    :param elements: List of elements to process.
    :param batch_size: Size of each batch.
    """
    predictions_combined = list()
    for i in range(0, len(elements), batch_size):
        batch = elements[i:i + batch_size]
        # Process the current batch
        # print(f"Processing batch: {batch}")
        # Add your processing logic here
        batch_prediction = predict_on_batch(tokenizer, model, batch)
        predictions_combined.extend(batch_prediction.cpu().numpy().astype(np.int32).tolist())

    return predictions_combined

def get_class_titles(labelmap, class_values):
    value_to_title = {value: title for title, value in labelmap.items()}
    class_titles = [value_to_title.get(value, "Unknown") for value in class_values]

    return class_titles