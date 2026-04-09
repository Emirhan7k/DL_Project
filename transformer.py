import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter

positive_sentences = [
"I really enjoyed using this product",
"This service exceeded my expectations",
"The quality is outstanding",
"I am very pleased with the results",
"Everything worked flawlessly",
"This was a wonderful experience",
"I appreciate the attention to detail",
"It performs exactly as advertised",
"I am happy with my decision",
"The design is elegant and modern",
"It is very easy to use",
"I had a great experience overall",
"The customer support was very helpful",
"This product is worth every penny",
"I was impressed by the performance",
"It arrived earlier than expected",
"The packaging was neat and secure",
"I found it extremely useful",
"This made my life much easier",
"I am satisfied with the quality",
"This is one of the best purchases I have made",
"It works smoothly without any issues",
"The build quality feels premium",
"I would definitely recommend this",
"It delivers excellent results",
"I am glad I bought this",
"The features are very practical",
"This product is reliable",
"It looks even better in person",
"I had no problems using it",
"The instructions were clear and simple",
"It fits my needs perfectly",
"I love how efficient it is",
"This is a high-quality item",
"It performs consistently well",
"I am impressed with the durability",
"This is exactly what I needed",
"It works better than I expected",
"I would happily buy this again",
"It offers great value",
"The experience was seamless",
"It is very well made",
"I am completely satisfied",
"The product feels sturdy",
"It exceeded all my expectations",
"I found it very convenient",
"It works like a charm",
"This is a solid product",
"I trust this brand now",
"It is worth recommending to others",
"The interface is clean and intuitive",
"I like how simple it is to operate",
"It saved me a lot of time",
"I enjoy using it daily",
"The performance is impressive",
"It is designed thoughtfully",
"The results are consistent",
"It is better than similar products",
"I feel confident using this",
"It is extremely practical",
"It exceeded my initial expectations",
"The product is well engineered",
"It is a smart purchase",
"I appreciate the fast response time",
"It works exactly as I hoped",
"It has great usability",
"It adds real value",
"It is easy to set up",
"The quality control seems excellent",
"It runs very smoothly",
"I like the overall design",
"It delivers what it promises",
"It is surprisingly effective",
"I had a positive experience",
"It feels premium and durable",
"It is quite efficient",
"I found no issues at all",
"It works perfectly every time",
"It is built with care",
"I like how reliable it is",
"It meets all my expectations",
"I had a smooth experience",
"It is a great addition",
"I appreciate the effort put into this",
"It is very convenient to use",
"The product works beautifully",
"It is nicely packaged",
"I am happy with the outcome",
"It performs exceptionally well",
"It is thoughtfully designed",
"I like the simplicity",
"It has exceeded my needs",
"It is extremely satisfying",
"It is very dependable",
"I had a great time using it",
"It is worth trying",
"It stands out from others",
"It is highly functional",
"It is well worth the investment",
"I like the build quality",
"It works without any trouble",
"It is very polished",
"It is well executed",
"It is a fantastic solution",
"It feels intuitive to use",
"It delivers strong performance",
"It is very impressive overall",
"It is a quality product",
"It does exactly what it should",
"I appreciate the usability",
"It is very effective",
"It has a nice finish",
"It works efficiently",
"It feels solid and reliable",
"It is a dependable option",
"I enjoyed the experience",
"It is built to last",
"It exceeded my hopes",
"It is extremely user friendly",
"It works seamlessly",
"It is easy to understand",
"It has a great concept",
"It performs reliably",
"It looks fantastic",
"It is highly recommended",
"It feels well constructed",
"It is incredibly useful",
"It adds convenience",
"It is a great choice",
"It meets high standards",
"It is quite impressive",
"It is very satisfying",
"It is well thought out",
"It functions perfectly",
"It is extremely reliable",
"It offers great usability",
"It works wonderfully",
"It is very practical",
"It is nicely designed",
"It is a strong product",
"It is highly efficient",
"It is truly enjoyable",
"It is very smooth",
"It is an excellent option",
"It is very capable",
"It is extremely well made",
"It is a pleasure to use"
]
negative_sentences = [
"I am very disappointed with this product",
"This was a terrible experience",
"The quality is unacceptable",
"It does not work as expected",
"I regret buying this",
"This product is a waste of money",
"It stopped working after a short time",
"The design is poor",
"It is very difficult to use",
"I had a bad experience overall",
"The customer service was unhelpful",
"This item is not worth the price",
"I am not satisfied with the results",
"It arrived damaged",
"The performance is very weak",
"It feels cheaply made",
"I expected much better",
"It does not meet my needs",
"I faced many issues using it",
"It broke very quickly",
"This is one of the worst purchases I have made",
"It does not function properly",
"The instructions were confusing",
"It looks worse in person",
"I would not recommend this",
"It is not reliable",
"The features are disappointing",
"It failed to deliver what was promised",
"I am unhappy with this purchase",
"It is poorly designed",
"It does not justify the cost",
"I had trouble using it",
"It is not user friendly",
"This product is very frustrating",
"It performs inconsistently",
"I would never buy this again",
"It is full of problems",
"The experience was disappointing",
"It feels like a low-quality product",
"It is not durable at all",
"I had high expectations but it failed",
"It is not practical",
"It wastes time and effort",
"I encountered many defects",
"It does not work smoothly",
"This was a bad decision",
"It lacks important features",
"I am completely dissatisfied",
"It is not worth buying",
"It is unreliable in daily use",
"It crashes frequently",
"It is extremely buggy",
"It is very unstable",
"It causes more problems than it solves",
"It is poorly built",
"It is hard to understand",
"It is not intuitive",
"It performs badly",
"It is very disappointing overall",
"It does not feel solid",
"It is a weak product",
"It is not well made",
"It is frustrating to use",
"It has many flaws",
"It is not worth the effort",
"It fails often",
"It is inconsistent",
"It is very annoying",
"It is badly designed",
"It does not meet standards",
"It is far below expectations",
"It is not satisfying",
"It is unreliable",
"It has poor performance",
"It does not last long",
"It is not efficient",
"It is a poor choice",
"It is disappointing to use",
"It is not impressive at all",
"It is cheaply constructed",
"It is not dependable",
"It is not stable",
"It is problematic",
"It is difficult to operate",
"It is not practical at all",
"It is full of errors",
"It is not well thought out",
"It performs poorly",
"It is extremely weak",
"It is not useful",
"It is poorly executed",
"It is not a good product",
"It is not enjoyable",
"It is not effective",
"It is far from ideal",
"It is disappointing in many ways",
"It is badly implemented",
"It is not satisfying to use",
"It is very underwhelming",
"It is not worth recommending",
"It is low quality",
"It is not impressive",
"It is flawed",
"It is a bad option",
"It is not consistent",
"It is not smooth",
"It is not reliable at all",
"It is not durable",
"It is a frustrating experience",
"It is not functional",
"It is poorly made",
"It is not helpful",
"It is not convenient",
"It is disappointing overall",
"It is a terrible choice",
"It is not well designed",
"It is not efficient at all",
"It is not practical for daily use",
"It is not stable enough",
"It is a weak solution",
"It is not user friendly at all",
"It is a bad experience",
"It is not acceptable",
"It is not up to standard",
"It is not impressive in any way",
"It is poorly constructed",
"It is not worth the price",
"It is disappointing from start to finish",
"It is not reliable in any situation",
"It is not a good investment",
"It is not well built",
"It is not satisfactory",
"It is not recommended",
"It is not a smart purchase",
"It is very poor overall",
"It is not worth your time",
"It is not designed well",
"It is not efficient in practice",
"It is not enjoyable at all",
"It is a bad product",
"It is not useful at all",
"It is not practical in real use",
"It is very disappointing indeed"
]

def process_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    return sentence

data = positive_sentences + negative_sentences
labels = [1] * len(positive_sentences) + [0] * len(negative_sentences)

data = [process_sentence(sentence) for sentence in data]

all_words = ' '.join(data).split()
word_counts = Counter(all_words)
vocab = {word: idx+1 for idx, (word, count) in enumerate(word_counts.items())}
vocab["<PAD>"] = 0

max_len = 15
def sentence_to_tensor(sentence, vocab=vocab, max_len=max_len):
    tokens = sentence.split()
    indices = [vocab.get(token, 0) for token in tokens]
    indices = indices[:max_len] + [0] * (max_len - len(indices))
    return torch.tensor(indices)

X = torch.stack([sentence_to_tensor(sentence, vocab, max_len) for sentence in data])
y = torch.tensor(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, num_classes):
        super(SimpleTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.transformer = nn.Transformer(d_model=embed_dim, nhead=num_heads, num_encoder_layers=num_layers,
                                  dim_feedforward=hidden_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim * max_len, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x) + self.positional_encoding
        output = self.transformer(embedded, embedded)
        output = output.view(output.size(0), -1)
        output = torch.relu(self.fc(output))
        output = self.out(output)
        output = self.sigmoid(output)
        return output
    
model = SimpleTransformer(vocab_size=len(vocab), embed_dim=32, num_heads=4, hidden_dim=64, num_layers=4, num_classes=1)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 300

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.long()).squeeze()
    loss = criterion(outputs, y_train.float())
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    test_outputs = model(X_test.long()).squeeze()
    predicted = (test_outputs > 0.5).int()
    accuracy = accuracy_score(y_test, predicted)
    print(f'Test Accuracy: {accuracy:.4f}')

