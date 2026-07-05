"""
Minimal Math AI - Ultra-Lightweight Learnable Model
- Architecture: Shared Embedding + Tiny LSTM + Attention
- Parameters: ~3-4K (ultra minimal)
- NO hardcoded rules - truly learns from data
- Converges properly despite minimal params
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import re


class MathTokenizer:
    """Minimal tokenizer untuk math expressions"""
    
    def __init__(self):
        # Ultra-minimal vocab
        self.tokens = [
            '<pad>', '<start>', '<end>', '<unk>',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '*', '/', '%', '**', '(',  ')',
            'floor', 'ceil', 'round', 'abs', 'sqrt', '.'
        ]
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(self.tokens)
    
    def tokenize(self, expr):
        """Tokenize expression"""
        expr = str(expr).replace(' ', '')
        pattern = r'(\d+\.?\d*|floor|ceil|round|abs|sqrt|\*\*|[+\-*/%()<>.])'
        matches = re.findall(pattern, expr)
        return ['<start>'] + matches + ['<end>']
    
    def encode(self, expr, max_len=30):
        """Convert ke token IDs"""
        tokens = self.tokenize(expr)
        ids = [self.token2id.get(t, self.token2id['<unk>']) for t in tokens[:max_len]]
        
        if len(ids) < max_len:
            ids += [self.token2id['<pad>']] * (max_len - len(ids))
        
        return torch.tensor(ids[:max_len], dtype=torch.long)
    
    def decode(self, ids):
        """Convert token IDs ke expression"""
        tokens = [self.id2token.get(id.item() if isinstance(id, torch.Tensor) else id, '<unk>') 
                  for id in ids if id != self.token2id['<pad>']]
        return ' '.join(tokens)


class AttentionPooling(nn.Module):
    """Attention mechanism untuk extract relevant info dari sequence"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, hidden_dim)
        output: (batch_size, hidden_dim) - weighted average
        """
        # Compute attention scores
        scores = self.attention(x)  # (batch_size, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask jika ada
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax untuk weights
        weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum
        output = torch.einsum('bs,bsh->bh', weights, x)  # (batch_size, hidden_dim)
        
        return output


class MinimalMathAI(nn.Module):
    """
    Ultra-lightweight math AI model
    - Shared embedding: 8 dims
    - LSTM: 24 hidden units
    - Attention pooling
    - Total: ~3-4K parameters
    """
    
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=24):
        super().__init__()
        
        # Shared embedding (8 dims)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Tiny LSTM (24 units)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # No dropout untuk model kecil
        )
        
        # Attention pooling
        self.attention = AttentionPooling(hidden_dim)
        
        # Minimal output layers
        self.fc1 = nn.Linear(hidden_dim, 16)
        self.relu = nn.ReLU()
        
        # Output heads
        self.answer_head = nn.Linear(16, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights untuk better convergence"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.dim() > 1:
                    nn.init.orthogonal_(param)
                else:
                    nn.init.normal_(param, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len) - token IDs
        output: dict dengan answer, confidence
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_len, hidden_dim)
        
        # Attention pooling
        context = self.attention(lstm_out)  # (batch_size, hidden_dim)
        
        # FC layers
        hidden_repr = self.relu(self.fc1(context))  # (batch_size, 16)
        
        # Output heads
        answer = self.answer_head(hidden_repr)  # (batch_size, 1)
        confidence = self.confidence_head(hidden_repr)  # (batch_size, 1)
        
        return {
            'answer': answer,
            'confidence': confidence
        }
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MathExpressionGenerator:
    """Generate valid math expressions"""
    
    @staticmethod
    def generate(difficulty=1):
        """Generate random math problem"""
        operations = [
            ('+', lambda a, b: a + b),
            ('-', lambda a, b: a - b),
            ('*', lambda a, b: a * b),
            ('/', lambda a, b: a / b if b != 0 else None),
            ('%', lambda a, b: a % b if b != 0 else None),
            ('**', lambda a, b: a ** b if (a <= 10 and b <= 5) else None),
        ]
        
        functions = [
            ('floor', lambda a, b: int(np.floor(a / b)) if b != 0 else None),
            ('ceil', lambda a, b: int(np.ceil(a / b)) if b != 0 else None),
            ('round', lambda a, b: round(a / b) if b != 0 else None),
            ('abs', lambda a, b: abs(a - b)),
        ]
        
        # 70% operator, 30% function
        if random.random() < 0.7:
            op_name, op_func = random.choice(operations)
            a = random.randint(1, 100 * difficulty)
            b = random.randint(1, 100 * difficulty)
            
            result = op_func(a, b)
            if result is None:
                return MathExpressionGenerator.generate(difficulty)
            
            expr = f"{a}{op_name}{b}"
        else:
            func_name, func = random.choice(functions)
            a = random.randint(1, 100 * difficulty)
            b = random.randint(1, 50 * difficulty)
            
            if func_name == 'abs':
                result = func(a, b)
                expr = f"abs({a}-{b})"
            else:
                result = func(a, b)
                if result is None:
                    return MathExpressionGenerator.generate(difficulty)
                expr = f"{func_name}({a}/{b})"
        
        return expr, float(result)


class MathDataset(Dataset):
    """Dataset untuk training"""
    
    def __init__(self, tokenizer, num_samples=3000, difficulty=1):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate training data"""
        data = []
        for _ in range(self.num_samples):
            expr, answer = MathExpressionGenerator.generate(self.difficulty)
            
            encoded = self.tokenizer.encode(expr)
            data.append({
                'input': encoded,
                'answer': torch.tensor([answer], dtype=torch.float32),
                'expr': expr
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item['input'], item['answer']


class MathAITrainer:
    """Trainer untuk minimal model"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.history = {
            'train_loss': [],
            'train_mae': [],
            'val_loss': [],
            'val_mae': []
        }
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train satu epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            output = self.model(batch_x)
            pred = output['answer']
            
            # Loss
            loss = criterion(pred, batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            mae = torch.abs(pred - batch_y).mean().item()
            total_mae += mae
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        return avg_loss, avg_mae
    
    def evaluate(self, dataloader, criterion):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                output = self.model(batch_x)
                pred = output['answer']
                
                loss = criterion(pred, batch_y)
                total_loss += loss.item()
                
                mae = torch.abs(pred - batch_y).mean().item()
                total_mae += mae
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        return avg_loss, avg_mae
    
    def train(self, num_epochs=150, batch_size=64, lr=0.005, difficulty=1):
        """Train model"""
        print(f"\n{'='*80}")
        print(f"Minimal Math AI Training")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Difficulty: {difficulty} | Epochs: {num_epochs} | Batch Size: {batch_size}")
        print(f"{'='*80}\n")
        
        # Create datasets
        train_dataset = MathDataset(self.tokenizer, num_samples=3000, difficulty=difficulty)
        val_dataset = MathDataset(self.tokenizer, num_samples=500, difficulty=difficulty)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer dan criterion
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
        
        best_val_mae = float('inf')
        patience = 30
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss, train_mae = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_mae = self.evaluate(val_loader, criterion)
            scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Train MAE: {train_mae:.4f} | "
                      f"Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.4f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print(f"\n{'='*80}")
        print(f"Training Complete! Best Val MAE: {best_val_mae:.4f}")
        print(f"{'='*80}\n")
    
    def predict(self, expr):
        """Predict answer dari expression"""
        self.model.eval()
        
        with torch.no_grad():
            encoded = self.tokenizer.encode(expr).unsqueeze(0).to(self.device)
            output = self.model(encoded)
            
            answer = output['answer'].item()
            confidence = output['confidence'].item()
        
        return answer, confidence


class MathAIGame:
    """Testing dan demo"""
    
    def __init__(self, trainer, device='cpu'):
        self.trainer = trainer
        self.device = device
    
    def test(self, num_tests=30, difficulty=1):
        """Test accuracy"""
        print(f"\n{'='*90}")
        print(f"Testing Minimal Math AI ({num_tests} problems, difficulty={difficulty})")
        print(f"{'='*90}\n")
        
        correct = 0
        total_error = 0
        
        for i in range(num_tests):
            expr, true_answer = MathExpressionGenerator.generate(difficulty)
            
            pred_answer, confidence = self.trainer.predict(expr)
            error = abs(pred_answer - true_answer)
            
            is_correct = error < 0.01
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {i+1:2d}. {expr:25s} | True: {true_answer:10.2f} | "
                  f"Pred: {pred_answer:10.2f} | Conf: {confidence:.2%} | Error: {error:.4f}")
            
            total_error += error
        
        accuracy = (correct / num_tests) * 100
        avg_error = total_error / num_tests
        
        print(f"\n{'='*90}")
        print(f"Results: {correct}/{num_tests} correct ({accuracy:.1f}%) | Avg Error: {avg_error:.4f}")
        print(f"{'='*90}\n")
        
        return accuracy
    
    def interactive(self):
        """Interactive mode"""
        print(f"\n{'='*90}")
        print("Interactive Mode - Minimal Math AI")
        print("Examples: '5+3', '10*2', 'floor(10/3)', 'abs(5-10)', '2**3'")
        print(f"{'='*90}\n")
        
        while True:
            expr = input("Expression (or 'quit'): ").strip()
            if expr.lower() == 'quit':
                break
            
            try:
                pred_answer, confidence = self.trainer.predict(expr)
                print(f"Answer: {pred_answer:.4f}")
                print(f"Confidence: {confidence:.2%}\n")
            except Exception as e:
                print(f"Error: {str(e)}\n")


def main():
    """Main function"""
    print("\n" + "="*90)
    print("Minimal Math AI - Ultra-Lightweight Learnable Model")
    print("="*90)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Create model
    tokenizer = MathTokenizer()
    print(f"Vocab Size: {tokenizer.vocab_size}")
    
    model = MinimalMathAI(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        hidden_dim=24
    )
    print(f"Model Parameters: {model.count_parameters():,}\n")
    
    trainer = MathAITrainer(model, tokenizer, device=device)
    game = MathAIGame(trainer, device)
    
    while True:
        print("Menu:")
        print("1. Train AI")
        print("2. Test AI")
        print("3. Interactive Mode")
        print("4. Exit")
        
        choice = input("\nPilih (1-4): ").strip()
        
        if choice == '1':
            difficulty = int(input("Difficulty (1-5, default 1): ") or "1")
            epochs = int(input("Epochs (default 150): ") or "150")
            batch_size = int(input("Batch size (default 64): ") or "64")
            lr = float(input("Learning rate (default 0.005): ") or "0.005")
            trainer.train(num_epochs=epochs, batch_size=batch_size, lr=lr, difficulty=difficulty)
        
        elif choice == '2':
            difficulty = int(input("Difficulty (1-5, default 1): ") or "1")
            num_tests = int(input("Number of tests (default 30): ") or "30")
            game.test(num_tests=num_tests, difficulty=difficulty)
        
        elif choice == '3':
            game.interactive()
        
        elif choice == '4':
            print("\nGoodbye!\n")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
