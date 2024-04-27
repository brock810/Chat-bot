import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from augment import apply_data_augmentation
import pandas as pd
from transformers import GPTNeoForCausalLM, GPT2Tokenizer, DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import os
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

intent_model_name = "distilbert-base-uncased"
intent_model = DistilBertForSequenceClassification.from_pretrained(intent_model_name).to(device)
intent_tokenizer = DistilBertTokenizer.from_pretrained(intent_model_name)


original_data = pd.read_csv("C:/Users/Brock/bot/conversation.csv")
augmented_data = pd.read_csv("C:/Users/Brock/bot/augmented_conversation.csv")


combined_data = pd.concat([original_data, augmented_data], ignore_index=True)


augmented_data = apply_data_augmentation(combined_data, columns=['message'])


train_data, temp_data = train_test_split(augmented_data, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

test_csv_path = "C:/Users/Brock/bot/test.csv"
test_data.to_csv(test_csv_path, index=False)  

val_csv_path = "C:/Users/Brock/bot/val.csv"
val_data.to_csv(val_csv_path, index=False)


curious_indices = augmented_data.index[augmented_data['sentiment'] == 'Curious to dive deeper'].tolist()


num_rows_to_change = len(curious_indices) // 2
indices_to_change = np.random.choice(curious_indices, num_rows_to_change, replace=False)


augmented_data.loc[indices_to_change, 'sentiment'] = 'Happy'


train_text = [
    {"input_ids": str(train_data["message"].values[i])}
    for i in range(len(train_data))
]

val_text = [
    {"input_ids": str(val_data["message"].values[i])}
    for i in range(len(val_data))
]

test_text = [
    {"input_ids": str(test_data["message"].values[i])}
    for i in range(len(test_data))
]

pretrained_model_name = "EleutherAI/gpt-neo-125M"
pretrained_model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name).to(device)
print("Model is on device:", pretrained_model.device)
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)


tokenizer.pad_token = tokenizer.eos_token


class CustomDataset(Dataset):
    def __init__(self, text_data, tokenizer, max_length=64, min_length=2):
        self.input_ids = []

        for text in text_data:
            inputs = tokenizer(
                text["input_ids"],
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            
            if inputs['input_ids'].size(1) >= min_length:
                self.input_ids.append(inputs['input_ids'])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx]
        }


train_dataset = CustomDataset(train_text, tokenizer, max_length=32)
val_dataset = CustomDataset(val_text, tokenizer, max_length=32)
test_dataset = CustomDataset(test_text, tokenizer, max_length=32)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

class CustomGPTNeoModel(GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = self.transformer  


        self.dropout_layers = nn.ModuleList([nn.Dropout(0.1) for _ in range(config.num_layers)])

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, past_key_values=past_key_values)


        for i, dropout_layer in enumerate(self.dropout_layers):
            transformer_outputs.hidden_states[i] = dropout_layer(transformer_outputs.hidden_states[i])

        return transformer_outputs


custom_model = CustomGPTNeoModel.from_pretrained(pretrained_model_name).to(device)


def context_aware_attention(query, keys, values):
    attention_weights = F.cosine_similarity(query, keys, dim=-1)
    attention_weights = F.softmax(attention_weights, dim=-1)
    attended_values = torch.matmul(attention_weights.unsqueeze(1), values)
    attended_values = attended_values.squeeze(1)
    return attended_values


def retrieve_memory(user_input, memory):

    user_input_tensor = user_input.clone().detach().to(device)

   
    for mem_input, mem_response in reversed(memory):

        mem_input_flat = torch.flatten(torch.tensor(mem_input)).to(device)

        if torch.equal(mem_input_flat, user_input_tensor):
            return f"Remembering: {mem_response}"


    decoded_input = tokenizer.decode(user_input_tensor[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


    intent_input = intent_tokenizer(decoded_input, return_tensors='pt').to(device)
    intent_output = intent_model(**intent_input)
    intent_label = torch.argmax(intent_output.logits).item()

    if intent_label == 0:

        return "I'm here to help. If anything is unclear, feel free to ask for clarification."
    elif intent_label == 1:

        return "Sure, I can provide more information. What specific details are you looking for?"


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(pretrained_model.parameters(), lr=1e-5)


scaler = GradScaler()

log_dir = "logs"  
writer = SummaryWriter(log_dir=log_dir)

conversation_memory = []

max_batches = 50
max_memory_size = 10000  


best_val_loss = float('inf')
patience = 5  
early_stopping_counter = 0
accumulation_steps = 4  

for epoch in range(15):
    print(f"Starting Epoch {epoch + 1}")
    pretrained_model.train()
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()  

    for batch_idx, batch in enumerate(train_dataloader):
        input_ids = batch['input_ids'].to(device)

        print(f"Batch {batch_idx + 1}, Input Tensor Size: {input_ids.size()}")

        with autocast():
            outputs = pretrained_model(input_ids=input_ids, labels=input_ids.contiguous())
            loss = outputs.loss

        scaler.scale(loss).backward()

        if torch.isnan(loss) or torch.isinf(loss):
            print("NaN or Inf loss encountered. Skipping gradient update.")
        else:

            torch.nn.utils.clip_grad_norm_(pretrained_model.parameters(), max_norm=0.5)  # Experiment with max_norm values


        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

        print("Model Device:", pretrained_model.device)
        print("Input Tensor Device:", input_ids.device)

        writer.add_scalar("Loss/Batch", loss.item(), num_batches)

        total_loss += loss.item()
        num_batches += 1

        decoded_texts = [tokenizer.decode(ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True) for ids in input_ids.tolist()]

        retrieved_info_batch = [retrieve_memory(user_input[0].unsqueeze(0), conversation_memory) for user_input in batch["input_ids"]]

        for i, (decoded_text, retrieved_info) in enumerate(zip(decoded_texts, retrieved_info_batch)):
            generated_response = f"{decoded_text} {retrieved_info}"
            print(f"Batch {batch_idx + 1}, Decoded Text {i + 1}: {generated_response}")

            conversation_memory.append((batch["input_ids"][i].tolist(), generated_response))

        conversation_memory = conversation_memory[-max_memory_size:]

    pretrained_model.save_pretrained("fine-tuned-gpt-neo-125M")
    tokenizer.save_pretrained("fine-tuned-gpt-neo-125M")

    if num_batches > 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()  

        print(f"Epoch {epoch + 1}, Average Loss: {total_loss / num_batches}")

        
        pretrained_model.eval()
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for val_batch_idx, val_batch in enumerate(val_dataloader):
                val_input_ids = val_batch['input_ids'].to(device)
                val_outputs = pretrained_model(input_ids=val_input_ids, labels=val_input_ids.contiguous())
                val_loss = val_outputs.loss
                total_val_loss += val_loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            average_val_loss = total_val_loss / num_val_batches
            print(f"Validation Loss: {average_val_loss}")

            
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                break

    
    pretrained_model.eval()
    total_test_loss = 0.0
    num_test_batches = 0

    with torch.no_grad():
        for test_batch_idx, test_batch in enumerate(test_dataloader):
            test_input_ids = test_batch['input_ids'].to(device)
            test_outputs = pretrained_model(input_ids=test_input_ids, labels=test_input_ids.contiguous())
            test_loss = test_outputs.loss
            total_test_loss += test_loss.item()
            num_test_batches += 1

        if num_test_batches > 0:
            average_test_loss = total_test_loss / num_test_batches
            print(f"Test Loss: {average_test_loss}")


pretrained_model.save_pretrained("fine-tuned-gpt-neo-125M")
tokenizer.save_pretrained("fine-tuned-gpt-neo-125M")


writer.close()
