import json
import torch
from datasets import Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification

class Ner():

    def __init__(self, model_path: str, device: torch.device, train_data_path: str) -> None:
        self.model_path = model_path
        self.train_data_path = train_data_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.label_map = None
    
    def get_train_data(self) -> Dataset:
        with open(self.train_data_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        train_data = Dataset.from_dict({
            "tokens": [data["tokens"] for data in data],
            "tags": [data["tags"] for data in data]
        })
        return train_data

    def tokenize_and_align_labels(self, data: Dataset):
        tokenized_inputs = self.tokenizer(data["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
        labels = []
        for i, label in enumerate(data["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(self.label_map[label[word_idx]])
                    except IndexError:
                        print(f"Error in example {i}: word_idx {word_idx} is out of range for label {label}")
                        label_ids.append(self.label_map["O"])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def train(self) -> None:
        # get train data
        data = self.get_train_data()
        # set train labels
        label_list = ["O", "B-種類", "I-種類", "B-演員", "I-演員"]
        self.label_map = {label: i for i, label in enumerate(label_list)}
        # load BERT fine-tune resources
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
        self.model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=len(label_list)).to(self.model)
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        # preprocess data
        tokenized_data = data.map(self.tokenize_and_align_labels, batched=True)
        training_args = TrainingArguments(
            output_dir="./results",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=1,
            evaluation_strategy="steps",
            eval_steps=1,
            save_steps=1,
            load_best_model_at_end=True,
        )
        # build trainer instance
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_data,
            eval_dataset=tokenized_data,
            data_collator=data_collator,
        )
        # start to train
        trainer.train()
        # store model and tokenizer
        self.model.save_pretrained("./ner_model")
        self.tokenizer.save_pretrained("./ner_model")

    def create_predict_env(self) -> None:
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
        self.model = BertForTokenClassification.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
     
    def predict(self, text: str) -> str:
        label_list = ["O", "B-種類", "I-種類", "B-演員", "I-演員"]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        predictions = outputs.argmax(dim=2)
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predicted_labels = [label_list[pred] for pred in predictions[0].tolist() if pred != -100]
        entities = []
        current_entity = {"type": "", "text": ""}

        for token, label in zip(tokens, predicted_labels):
            if label.startswith("B-"):
                if current_entity["text"]:
                    entities.append(current_entity)
                current_entity = {"type": label[2:], "text": token}
            elif label.startswith("I-") and current_entity["text"]:
                current_entity["text"] += token.replace("##", "")
            elif label == "O" and current_entity["text"]:
                entities.append(current_entity)
                current_entity = {"type": "", "text": ""}
        
        if current_entity["text"]:
            entities.append(current_entity)
        
        aligned_entities = []
        for entity in entities:
            start = text.lower().find(entity["text"].lower())
            if start != -1:
                end = start + len(entity["text"])
                aligned_entities.append({
                    "type": entity["type"],
                    "text": text[start:end],
                    "start": start,
                    "end": end
                })

        words = []
        for entity in aligned_entities:
            words.append(entity["text"])
        return words
