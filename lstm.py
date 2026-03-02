import torch 
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import product

text = """
LSTM modeli için yeterli bir ürün olarak, uzun vadeli trend, mevsimsellik ve rastgele
gürültü içeren günlük zaman serisi 
verilerinden oluşan sentetik bir satış tahmin veri seti oluşturulabilir; 
bu veri seti, geçmiş 30 günlük satış değerlerini kullanarak bir sonraki günü tahmin etmeye uygun olacak şekilde sequence 
formatında hazırlanır ve modelin hem uzun dönem bağımlılıkları  
hem de periyodik yapıları öğrenmesini sağlayacak gerçekçi bir yapı sunar.
"""

words = text.replace(".", "").replace(",", "").replace(";", "").lower().split()

word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_idx = {word: idx for idx, word in enumerate(vocab)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

data = [(words[i], words[i + 1]) for i in range(len(words) - 1)]

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(LSTM,self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self,x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x.view(1,1,-1 ))
        output = self.fc(lstm_out.view(1, -1))
        return output                         

def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

embedding_sizes = [10, 20]
hidden_sizes = [32, 64]
learning_rates = [0.01, 0.005]
best_loss = float('inf')
best_params = {}

for embedding_dim, hidden_dim, lr in product(embedding_sizes, hidden_sizes, learning_rates):
    model = LSTM(len(vocab), embedding_dim, hidden_dim)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(50):
        total_loss = 0
        for word, next_word in data:
            model.zero_grad()
            input_seq = prepare_sequence([word], word_to_idx)
            target_seq = prepare_sequence([next_word], word_to_idx)
            output = model(input_seq)
            loss = loss_function(output, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / len(data)
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_params = {
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'learning_rate': lr
        }

print(f"Best Loss: {best_loss:.4f}")
print(f"Best Hyperparameters: {best_params}")

final_model = LSTM(len(vocab), best_params["embedding_dim"], best_params["hidden_dim"])
optimizer = optim.Adam(final_model.parameters(), lr=best_params["learning_rate"])
epochs = 100
def train_final_model(final_model, data, epochs, optimizer, loss_function ):
    final_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for word , next_word in data:
            final_model.zero_grad()
            input_seq = prepare_sequence([word], word_to_idx)
            target_seq = prepare_sequence([next_word], word_to_idx)
            output = final_model(input_seq)
            loss = loss_function(output, target_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data):.4f}")
    


def predict_sequence(start_word, num_words, final_model):
    final_model.eval()
    current_word = start_word
    output_sequence = [current_word]
    with torch.no_grad():
        for _ in range(num_words):
            input_seq = prepare_sequence([current_word], word_to_idx)
            output = final_model(input_seq)
            predicted_idx = torch.argmax(output).item()
            predicted_word = idx_to_word[predicted_idx]
            output_sequence.append(predicted_word)
            current_word = predicted_word
    return output_sequence

train_final_model(final_model, data, epochs, optimizer, loss_function)
start_word = "lstm"
predicted_sequence = predict_sequence(start_word, 5, final_model)
print(f"Predicted sequence starting with '{start_word}': {' '.join(predicted_sequence)}")
