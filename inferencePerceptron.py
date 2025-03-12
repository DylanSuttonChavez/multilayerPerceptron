import torch
import torch.nn as nn
import pickle

# Load the saved model and vocabulary
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model_data = pickle.load(file)
    
    # Retrieve model's state_dict and vocabulary
    model_state_dict = model_data["model_state_dict"]
    vocab = model_data["vocabulary"]
    
    return model_state_dict, vocab

# Define the MLP model class (same as used for training)
class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 10, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x).view(x.size(0), -1)
        hidden = torch.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

# Function to generate text from the trained model
def generate_text(model, start_text, num_words, vocab, seq_length, device):
    model.eval()
    words = start_text.split()
    
    # If start_text is shorter than seq_length, pad it
    if len(words) < seq_length:
        pad = [words[0]] * (seq_length - len(words))
        words = pad + words
        
    current_seq = words[-seq_length:]  # Start with the last 'seq_length' words
    
    # Generate words one by one
    for _ in range(num_words):
        inputs = torch.tensor([[vocab[word] for word in current_seq]], dtype=torch.long).to(device)
        outputs = model(inputs)
        predicted_word_idx = torch.argmax(outputs, dim=1).item()
        predicted_word = list(vocab.keys())[list(vocab.values()).index(predicted_word_idx)]
        words.append(predicted_word)
        current_seq = words[-seq_length:]
    
    return ' '.join(words)

# Load the trained model
model_state_dict, vocab = load_model('storyModel.pkl')

# Initialize the model with the same architecture used during training
embedding_dim = 50
hidden_dim = 100
output_dim = len(vocab)
seq_length = 10

# Create the model and load the saved weights
model = MLP(len(vocab), embedding_dim, hidden_dim, output_dim)
model.load_state_dict(model_state_dict)

# Move the model to the device (CUDA if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use the model to generate text
start_text = "the group discovered murals depicting the primordial spirit of the"
generated_text = generate_text(model, start_text, num_words=50, vocab=vocab, seq_length=seq_length, device=device)

print(f"{start_text} : {generated_text}")