import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import pickle

# Load the training text file
with open('trainingStory.txt', 'r', encoding='utf-8') as file:
    text = file.read()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tokenizing the text into words
words = text.split()

# Count word frequencies
word_counts = Counter(words)
vocab = {word: idx for idx, word in enumerate(word_counts)}
vocab_size = len(vocab)

# Map the words in the text to their respective indices
data = [vocab[word] for word in words]

# Define sequence length for input-output pairs
seq_length = 10
X = []
y = []

# Prepare input-output pairs for training
for i in range(len(data) - seq_length):
    X.append(data[i:i+seq_length])
    y.append(data[i+seq_length])

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * seq_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        hidden = torch.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

# Hyperparameters
embedding_dim = 50
hidden_dim = 100
output_dim = vocab_size

# Initialize the model
model = MLP(vocab_size, embedding_dim, hidden_dim, output_dim)
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training parameters
num_epochs = 20
batch_size = 32

# Early stopping setup
patience = 3
best_loss = float('inf')
epochs_without_improvement = 0

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    correct_predictions = 0
    for i in range(0, len(X), batch_size):
        inputs = X[i:i+batch_size].to(device)
        targets = y[i:i+batch_size].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Count correct predictions
        _, predicted = torch.max(outputs, dim=1)
        correct_predictions += (predicted == targets).sum().item()

    avg_loss = epoch_loss / (len(X) // batch_size)
    accuracy = (correct_predictions / len(X)) * 100

    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping activated. Stopping training at epoch {epoch+1}.\n")
        break

# Save the model and vocabulary
model_data = {
    "model_state_dict": model.state_dict(),
    "vocabulary": vocab,
}

with open("storyModel.pkl", "wb") as file:
    pickle.dump(model_data, file)

print(f"Model saved as 'financeModel.pkl'. Test Accuracy: {accuracy:.2f}%. Training completed.")

# Function to generate text based on the trained model
def generate_text(model, start_text, num_words, vocab, seq_length, device):
    model.eval()
    words = start_text.split()
    if len(words) < seq_length:
        pad = [words[0]] * (seq_length - len(words))
        words = pad + words
    current_seq = words[-seq_length:]
    
    for _ in range(num_words):
        inputs = torch.tensor([[vocab[word] for word in current_seq]], dtype=torch.long).to(device)
        outputs = model(inputs)
        predicted_word_idx = torch.argmax(outputs, dim=1).item()
        predicted_word = list(vocab.keys())[list(vocab.values()).index(predicted_word_idx)]
        words.append(predicted_word)
        current_seq = words[-seq_length:]
    
    return ' '.join(words)

start_text = "the group discovered murals depicting the primordial spirit of the"
generated_text = generate_text(model, start_text, num_words=73, vocab=vocab, seq_length=seq_length, device=device)
print(f"{start_text} : {generated_text}")