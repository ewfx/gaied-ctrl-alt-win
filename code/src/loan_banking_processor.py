from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from accelerate import Accelerator, DataLoaderConfiguration
import torch
from typing import Dict, Tuple

class LoanBankingEmailProcessor:
    def __init__(self):
        self.zero_shot_classifier = None
        self.tokenizer = None
        self.model = None
        self.categories = {
            "Loan Application": ["New Application", "Documents Required", "Verification", "Approval Status"],
            "Loan Servicing": ["Payment Inquiry", "Statement Request", "Late Payment", "Prepayment"],
            "Customer Support": ["Account Inquiry", "Complaint", "Technical Issue", "General Query"],
            "Collections": ["Overdue Notice", "Payment Arrangement", "Default Warning", "Recovery"],
            "Marketing": ["Loan Offers", "Rate Updates", "Promotions", "Newsletters"]
        }
        self.label_map = {f"{cat}_{sub}": idx for idx, (cat, subs) in enumerate(self.categories.items()) for sub in subs}
        print(f"Label map size: {len(self.label_map)}, labels: {self.label_map}")
        self.feedback_data = {'texts': [], 'labels': []}
        self.email_history = []  # For duplicate detection

    def _load_zero_shot(self):
        if not self.zero_shot_classifier:
            self.zero_shot_classifier = pipeline("zero-shot-classification", model="prajjwal1/bert-tiny", device=-1)

    def _load_distilbert(self):
        if not self.tokenizer or not self.model:
            self.tokenizer = DistilBertTokenizer.from_pretrained('prajjwal1/bert-tiny')
            self.model = DistilBertForSequenceClassification.from_pretrained(
                'prajjwal1/bert-tiny', 
                num_labels=len(self.label_map)
            )
            print(f"Model initialized with {len(self.label_map)} labels")

    def categorize_email(self, email: Dict) -> Tuple[str, str, float]:
        text = f"{email['subject']} {email['content']}"
        
        if self.model and self.tokenizer:
            inputs = self.tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                predicted_label_id = torch.argmax(logits, dim=1).item()
                print(f"Predicted label ID: {predicted_label_id}, logits shape: {logits.shape}, logits: {logits}")
                
                if predicted_label_id in self.label_map.values():
                    for label, idx in self.label_map.items():
                        if idx == predicted_label_id:
                            category, subcategory = label.split('_', 1)
                            confidence = torch.softmax(logits, dim=1)[0][predicted_label_id].item()
                            return category, subcategory, confidence
                print(f"Warning: Predicted label ID {predicted_label_id} not in label_map, falling back to zero-shot")
        
        # Fallback to zero-shot
        self._load_zero_shot()
        candidate_labels = list(self.categories.keys())
        result = self.zero_shot_classifier(text, candidate_labels)
        category = result['labels'][0]
        subcategories = self.categories[category]
        sub_result = self.zero_shot_classifier(text, subcategories)
        subcategory = sub_result['labels'][0]
        confidence = (result['scores'][0] + sub_result['scores'][0]) / 2
        return category, subcategory, confidence

    def detect_duplicate(self, email: Dict) -> bool:
        """Simple duplicate check based on subject and content."""
        current_text = f"{email['subject']} {email['content']}"
        if current_text in self.email_history:
            return True
        self.email_history.append(current_text)
        return False

    def update_with_feedback(self, text: str, correct_category: str, correct_subcategory: str):
        try:
            label = f"{correct_category}_{correct_subcategory}"
            if label not in self.label_map:
                raise ValueError(f"Invalid category/subcategory: {label}")
            label_id = self.label_map[label]
            self.feedback_data['texts'].append(text)
            self.feedback_data['labels'].append(label_id)
            if len(self.feedback_data['texts']) >= 1:
                self._load_distilbert()
                self._fine_tune_model()
        except Exception as e:
            raise Exception(f"Feedback processing failed: {str(e)}")

    def _fine_tune_model(self):
        try:
            print("Starting fine-tuning...")
            if not self.model:
                self.model = DistilBertForSequenceClassification.from_pretrained(
                    'prajjwal1/bert-tiny', 
                    num_labels=len(self.label_map)
                )
                print("Model loaded successfully")
            encodings = self.tokenizer(self.feedback_data['texts'], truncation=True, padding=True, max_length=512, return_tensors='pt')
            dataset = [
                {
                    "input_ids": encodings['input_ids'][i],
                    "attention_mask": encodings['attention_mask'][i],
                    "labels": torch.tensor(self.feedback_data['labels'][i], dtype=torch.long)
                }
                for i in range(len(self.feedback_data['texts']))
            ]
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=3,
                per_device_train_batch_size=1,
                logging_dir='./logs',
                logging_steps=1,
                save_strategy="no",
            )
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset
            )
            trainer.train()
            print("Training completed")
        except Exception as e:
            raise Exception(f"Fine-tuning failed: {str(e)}")