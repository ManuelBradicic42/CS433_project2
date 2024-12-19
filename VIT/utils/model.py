import os 
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor
from tqdm import tqdm 
import numpy as np
import datetime
import csv
import ast

####################################################
################## VIT MODEL #######################
####################################################

import torch
import torch.nn as nn
from transformers import ViTModel

# class ViTSequenceClassifier(nn.Module):
#     def __init__(self, num_classes=28, max_length=7):
#         super().__init__()
#         self.max_length = max_length

#         # Load a pretrained ViT backbone
#         self.backbone = ViTModel.from_pretrained(
#             "google/vit-base-patch16-224-in21k"
#         )
#         self.hidden_size = self.backbone.config.hidden_size

#         # Classification heads for each sequence position
#         self.classifiers = nn.ModuleList([
#             nn.Linear(self.hidden_size, num_classes) for _ in range(max_length)
#         ])

#     def forward(self, pixel_values):
#         # Extract features from the image, requesting attentions
#         outputs = self.backbone(pixel_values=pixel_values, output_attentions=True)
        
#         # outputs.attentions is a tuple of shape:
#         # (num_layers, batch_size, num_heads, seq_len, seq_len)
#         attentions = outputs.attentions
#         features = outputs.pooler_output  # Shape: (batch_size, hidden_size)
        
#         # Apply classification heads
#         logits = torch.stack([classifier(features) for classifier in self.classifiers], dim=1)
        
#         return logits, attentions

class ViTSequenceClassifier(nn.Module):
    def __init__(self, num_classes=28, max_length=7):
        super().__init__()
        self.max_length = max_length

        # Load a pretrained ViT backbone
        self.backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", )
        self.hidden_size = self.backbone.config.hidden_size

        # Classification heads for each sequence position
        self.classifiers = nn.ModuleList([
            nn.Linear(self.hidden_size, num_classes) for _ in range(max_length)
        ])

    def forward(self, pixel_values):
        # Extract features from the image
        outputs = self.backbone(pixel_values=pixel_values)
        features = outputs.pooler_output  # Shape: (batch_size, hidden_size)

        # Apply classification heads
        logits = torch.stack([classifier(features) for classifier in self.classifiers], dim=1)
        return logits  # Shape: (batch_size, max_length, num_classes)



####################################################
################## TRAIN FUNCTION ##################
####################################################

def train_model(model, train_dl, validation_dl, test_dl, optimizer, criterion, device, loss_hist, epochs=5, batch_size=64, loss_name="CTC"):
    model.to(device)
    model.train()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")

    for epoch in range(epochs):
        total_loss = 0
        counter = 0

        for pixel_values, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(pixel_values)

            loss = 0
            if loss_name == "CTC":
                loss = compute_loss_CTC(logits, labels, criterion)
            elif loss_name == "cross_entropy":
                for i in range(model.max_length):
                    loss += criterion(logits[:, i, :], labels[:, i])

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            counter +=1 
            # loss_hist["train_loss_iterations"].append(total_loss/(counter * batch_size))
                
        test_acc = evaluate_test(model, test_dl, device)
        test_cer = evalute_model_levenstein(model, test_dl, device)
        loss_hist["test_acc"].append(test_acc)
        loss_hist["test_cer"].append(test_cer)
        print(test_acc)
        print(test_cer)
                
        #Logging 
        loss_train = total_loss/(len(train_dl))
        loss_hist["train_loss"].append(loss_train)

        #Evaluating on val
        loss_val = evaluate_model(model, validation_dl, criterion, batch_size, device, loss_name)
        loss_hist["val_loss"].append(loss_val.cpu().item())

        #Save model
        name = f"model_{current_time}__{test_acc}_{epoch}.pth"
        path = "saved/"
        torch.save(model.state_dict(), os.path.join(path, name))

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {loss_train:.8f}, Val Loss: {loss_val:.8f}")
        
        #Save logs
        save_logs(current_time, loss_hist)

### Support functions

def get_criterion(loss_name):
    if loss_name == "CTC":
        return torch.nn.CTCLoss(reduction='none', zero_infinity=True)
    elif loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

def evaluate_model(model, dataloader, criterion, BATCH_SIZE, device, loss_name):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            logits = model(pixel_values)

            loss = 0
            if loss_name == "CTC":
                loss = compute_loss_CTC(logits, labels, criterion)
            elif loss_name == "cross_entropy":
                for i in range(model.max_length):
                    loss += criterion(logits[:, i, :], labels[:, i])
            
            total_loss += loss
    return total_loss/(len(dataloader))

@torch.no_grad()
def evaluate_test(model, test_dataloader, device):
    model.eval()
    sum = 0
    for x,y in test_dataloader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        out = torch.argmax(out, dim=2)
        out = out.cpu().numpy()
        out = [x[x != 0] for x in out]
        out = np.array([np.pad(arr, (0, 7 - len(arr)), constant_values=0) for arr in out])

        y = y.cpu().numpy()

        sum += np.sum(np.all(out == y, axis=1))
        
    return sum

def evaluate_saved_model_on_test(model, path, test_dataloader, size, device):
    model.load_state_dict(torch.load(path, weights_only=True))
    sum = evaluate_test(model, test_dataloader, device)
    print(f"Total correct predictions on test dataset: {sum}, Acc: {np.round(sum/size, 4)}")

    
def compute_loss_CTC(out, y, criterion):
    list_ = [x[:(x != 0).nonzero()[-1].item() + 1] for x in y]
    target_length = torch.IntTensor([len(x) for x in list_])
    target_values = torch.cat(list_)

    preds = out.float()
    preds = preds.permute(1, 0, 2).log_softmax(2)
    preds_size = torch.IntTensor([preds.shape[0]] * preds.shape[1])

    return criterion(preds, target_values, preds_size, target_length).mean()

################################################
### Levenstein distance and evaluation of test##
################################################

def levenshteinRecursive(pred, target, m, n):
      # pred is empty
    if m == 0:
        return n
    # str2 is empty
    if n == 0:
        return m
    if pred[m - 1] == target[n - 1]:
        return levenshteinRecursive(pred, target, m - 1, n - 1)
    return 1 + min(
          # Insert     
        levenshteinRecursive(pred, target, m, n - 1),
        min(
            levenshteinRecursive(pred, target, m - 1, n),
            levenshteinRecursive(pred, target, m - 1, n - 1))
    )

def batched_levenstein_distance(out, labels):
    total_distance = 0
    for i in range(0,len(out)):
        temp_out = out[i][out[i] != 0]
        temp_lab = labels[i][labels[i] != 0]
        total_distance +=levenshteinRecursive(temp_out, temp_lab, len(temp_out), len(temp_lab))
    return total_distance/len(out)

def evalute_model_levenstein(model, dataloader, device):
    model.eval()
    total_loss_ctc = 0
    total_levenstein_d = 0
    with torch.no_grad():
        for pixel_values, labels in dataloader:
            pixel_values, labels = pixel_values.to(device), labels.to(device)
            out = model(pixel_values)
            
            #Levenstein
            out = torch.argmax(out, dim=2)
            out = out.cpu().numpy()
            out = [x[x != 0] for x in out]
            out = np.array([np.pad(arr, (0, 7 - len(arr)), constant_values=0) for arr in out])

            labels = labels.cpu().numpy()
            total_levenstein_d += batched_levenstein_distance(out, labels)
            
    
    return total_levenstein_d/len(dataloader)

### Logs 

def save_logs(current_time, logs):
    csv_file = f'logs/output_{current_time}.csv'

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(logs.keys())
        writer.writerow(logs.values())
    print(f"Dictionary saved to {csv_file}")

def read_logs(path):
    with open(path, mode="r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                keys = row
                dict_from_list = {key: None for key in row}
            if i == 1:
                for j,row_ in enumerate(row):
                    if j <= 2:
                        python_list = ast.literal_eval(row_)
                        numpy_array = np.array(python_list, dtype=float)
                        dict_from_list[keys[j]] = numpy_array
                    else:
                        dict_from_list[keys[j]] = row_
    
    return dict_from_list


