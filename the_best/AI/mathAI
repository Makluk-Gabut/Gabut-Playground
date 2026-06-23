"""
Minimal Math Neural Network AI
- Operasi: +, -, *, /, **, %, floor, ceil, round, abs, >, <, ==
- Architecture: Ultra-minimal LSTM dengan regression output
- Modes: Self-training, Self-play, Multiplayer (vs human)
- Parameters: ~3K total (extremely minimal)
- Output: Regresi (bisa handle hasil sampai jutaan)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import re


class MathTokenizer:
    """Parse dan tokenize math expressions"""
    
    def __init__(self):
        self.tokens = [
            '<pad>', '<start>', '<end>', '<unk>',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '+', '-', '*', '/', '%', '**', '(', ')',
            'floor', 'ceil', 'round', 'abs',
            '>', '<', '=='
        ]
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(self.tokens)
    
    def tokenize(self, expr):
        """Tokenize expression string"""
        expr = expr.replace(' ', '')
        pattern = r'(\d+|floor|ceil|round|abs|\*\*|==|>=|<=|[+\-*/%()<>])'
        matches = re.findall(pattern, expr)
        tokens = ['<start>'] + matches + ['<end>']
        return tokens
    
    def encode(self, expr, max_len=40):
        """Convert expression ke token IDs"""
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


class MinimalMathNN(nn.Module):
    """
    Ultra-minimal architecture dengan regression output:
    - Embedding: 8 dims
    - LSTM: 16 hidden units
    - Output: 1 (regression untuk hasil angka)
    - Total params: ~3K
    """
    
    def __init__(self, vocab_size, embedding_dim=8, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        x: (batch_size, seq_len)
        output: (batch_size, 1) - regression output untuk hasil
        """
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        last_output = hidden.squeeze(0)
        output = self.fc(last_output)
        
        return output
    
    def count_parameters(self):
        """Hitung total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MathOperations:
    """Semua operasi matematika yang support"""
    
    @staticmethod
    def safe_eval(expr):
        """Evaluate expression dengan aman"""
        try:
            allowed = {
                'abs': abs,
                'floor': lambda x: int(np.floor(x)),
                'ceil': lambda x: int(np.ceil(x)),
                'round': round,
            }
            result = eval(expr, {"__builtins__": {}}, allowed)
            return float(result)
        except:
            return None
    
    @staticmethod
    def generate_problem(difficulty=1):
        """Generate random math problem"""
        operations = ['+', '-', '*', '/', '%', '**', 'floor', 'ceil', 'round', 'abs']
        comparisons = ['>', '<', '==']
        
        if random.random() < 0.2:
            op = random.choice(comparisons)
            a = random.randint(1, 50 * difficulty)
            b = random.randint(1, 50 * difficulty)
            expr = f"{a}{op}{b}"
            
            if op == '>':
                answer = 1.0 if a > b else 0.0
            elif op == '<':
                answer = 1.0 if a < b else 0.0
            else:
                answer = 1.0 if a == b else 0.0
        else:
            a = random.randint(1, 100 * difficulty)
            b = random.randint(1, 100 * difficulty) if random.random() < 0.7 else random.random() * 100
            
            op = random.choice(operations)
            
            if op == '+':
                expr, answer = f"{a}+{b}", float(a + b)
            elif op == '-':
                expr, answer = f"{a}-{b}", float(a - b)
            elif op == '*':
                expr, answer = f"{a}*{b}", float(a * b)
            elif op == '/':
                if b == 0:
                    return MathOperations.generate_problem(difficulty)
                expr, answer = f"{a}/{b}", float(a / b)
            elif op == '%':
                if b == 0:
                    return MathOperations.generate_problem(difficulty)
                expr, answer = f"{a}%{b}", float(a % b)
            elif op == '**':
                if a > 10 or b > 5:
                    return MathOperations.generate_problem(difficulty)
                expr, answer = f"{a}**{b}", float(a ** b)
            elif op == 'floor':
                expr, answer = f"floor({a}/{b})", float(int(np.floor(a / b)) if b != 0 else 0)
            elif op == 'ceil':
                expr, answer = f"ceil({a}/{b})", float(int(np.ceil(a / b)) if b != 0 else 0)
            elif op == 'round':
                expr, answer = f"round({a}/{b})", float(round(a / b) if b != 0 else 0)
            elif op == 'abs':
                expr, answer = f"abs({a}-{b})", float(abs(a - b))
        
        return expr, answer


class MathDataset(Dataset):
    """Dataset untuk training"""
    
    def __init__(self, tokenizer, num_samples=5000, difficulty=1):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.data = self._generate_data()
    
    def _generate_data(self):
        """Generate training data"""
        data = []
        for _ in range(self.num_samples):
            expr, answer = MathOperations.generate_problem(self.difficulty)
            encoded = self.tokenizer.encode(expr)
            data.append((encoded, torch.tensor([answer], dtype=torch.float32)))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MathAITrainer:
    """Training loop untuk math AI"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.tokenizer = MathTokenizer()
        self.history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': []}
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train 1 epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            pred = self.model(batch_x)
            loss = criterion(pred, batch_y)
            
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
                
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                
                total_loss += loss.item()
                mae = torch.abs(pred - batch_y).mean().item()
                total_mae += mae
        
        avg_loss = total_loss / len(dataloader)
        avg_mae = total_mae / len(dataloader)
        return avg_loss, avg_mae
    
    def train(self, num_epochs=100, batch_size=64, lr=0.01, difficulty=1):
        """Train model"""
        print(f"Starting training... (difficulty={difficulty})")
        print(f"Model parameters: {self.model.count_parameters():,}\n")
        
        train_dataset = MathDataset(self.tokenizer, num_samples=10000, difficulty=difficulty)
        val_dataset = MathDataset(self.tokenizer, num_samples=2000, difficulty=difficulty)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-6)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        
        best_val_mae = float('inf')
        patience = 15
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
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} | "
                      f"Val MAE: {val_mae:.2f}")
            
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training selesai! Best Val MAE: {best_val_mae:.2f}\n")
        return best_val_mae
    
    def predict(self, expr):
        """Prediksi jawaban dari expression"""
        self.model.eval()
        with torch.no_grad():
            encoded = self.tokenizer.encode(expr).unsqueeze(0).to(self.device)
            pred = self.model(encoded)
            result = round(pred.item())
        return result


class MathAIGame:
    """Game modes: Self-play dan Multiplayer"""
    
    def __init__(self, model, trainer, device='cpu'):
        self.model = model
        self.trainer = trainer
        self.device = device
    
    def self_play(self, num_rounds=10, difficulty=1):
        """AI bermain melawan dirinya sendiri"""
        print(f"\nSelf-Play Mode ({num_rounds} rounds)")
        print("=" * 70)
        
        ai_score = 0
        total_error = 0
        
        for round_num in range(num_rounds):
            expr, correct_answer = MathOperations.generate_problem(difficulty)
            ai_answer = self.trainer.predict(expr)
            
            error = abs(ai_answer - correct_answer)
            is_correct = error == 0
            if is_correct:
                ai_score += 1
            total_error += error
            
            status = "CORRECT" if is_correct else f"ERROR: {error:.0f}"
            print(f"Round {round_num+1}: {expr:25s} | "
                  f"AI: {ai_answer:8.0f} | Correct: {correct_answer:8.0f} | {status}")
        
        accuracy = ai_score / num_rounds
        avg_error = total_error / num_rounds
        print("=" * 70)
        print(f"AI Score: {ai_score}/{num_rounds} ({accuracy*100:.1f}%) | Avg Error: {avg_error:.2f}\n")
        return accuracy
    
    def multiplayer(self, num_rounds=10, difficulty=1):
        """Manusia vs AI"""
        print(f"\nMultiplayer Mode ({num_rounds} rounds)")
        print("=" * 70)
        
        human_score = 0
        ai_score = 0
        human_errors = 0
        ai_errors = 0
        
        for round_num in range(num_rounds):
            expr, correct_answer = MathOperations.generate_problem(difficulty)
            
            print(f"\nRound {round_num+1}: {expr}")
            
            while True:
                try:
                    human_answer = float(input("Your answer: "))
                    break
                except:
                    print("Invalid input!")
            
            ai_answer = self.trainer.predict(expr)
            
            human_error = abs(human_answer - correct_answer)
            ai_error = abs(ai_answer - correct_answer)
            
            human_correct = human_error == 0
            ai_correct = ai_error == 0
            
            print(f"Correct answer: {correct_answer:.0f}")
            print(f"You:  {human_answer:.0f} {'CORRECT' if human_correct else f'ERROR: {human_error:.0f}'}")
            print(f"AI:   {ai_answer:.0f} {'CORRECT' if ai_correct else f'ERROR: {ai_error:.0f}'}")
            
            if human_correct:
                human_score += 1
            else:
                human_errors += human_error
            
            if ai_correct:
                ai_score += 1
            else:
                ai_errors += ai_error
        
        print("\n" + "=" * 70)
        print(f"Final Score:")
        print(f"   You: {human_score}/{num_rounds} (Avg Error: {human_errors/num_rounds:.2f})")
        print(f"   AI:  {ai_score}/{num_rounds} (Avg Error: {ai_errors/num_rounds:.2f})")
        
        if human_score > ai_score:
            print(f"You win! (+{human_score - ai_score} points)")
        elif ai_score > human_score:
            print(f"AI wins! (+{ai_score - human_score} points)")
        else:
            print(f"Its a tie!")
        
        print()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("Minimal Math Neural Network AI - Regression Version")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    tokenizer = MathTokenizer()
    model = MinimalMathNN(vocab_size=tokenizer.vocab_size)
    print(f"Model Parameters: {model.count_parameters():,}\n")
    
    trainer = MathAITrainer(model, device=device)
    
    while True:
        print("\nMenu:")
        print("1. Train AI (self-training)")
        print("2. Self-Play (AI vs itself)")
        print("3. Multiplayer (You vs AI)")
        print("4. Single Prediction")
        print("5. Exit")
        
        choice = input("\nPilih (1-5): ").strip()
        
        if choice == '1':
            difficulty = int(input("Difficulty (1-5): ") or "1")
            epochs = int(input("Epochs (default 100): ") or "100")
            trainer.train(num_epochs=epochs, difficulty=difficulty)
        
        elif choice == '2':
            difficulty = int(input("Difficulty (1-5): ") or "1")
            rounds = int(input("Rounds (default 10): ") or "10")
            game = MathAIGame(model, trainer, device)
            game.self_play(num_rounds=rounds, difficulty=difficulty)
        
        elif choice == '3':
            difficulty = int(input("Difficulty (1-5): ") or "1")
            rounds = int(input("Rounds (default 10): ") or "10")
            game = MathAIGame(model, trainer, device)
            game.multiplayer(num_rounds=rounds, difficulty=difficulty)
        
        elif choice == '4':
            expr = input("Enter expression (e.g., '5+3', 'floor(10/3)', '100+200*5'): ").strip()
            pred = trainer.predict(expr)
            print(f"AI Answer: {pred}\n")
        
        elif choice == '5':
            print("\nGoodbye!\n")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()
