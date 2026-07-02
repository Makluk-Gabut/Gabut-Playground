"""
Hybrid Math AI - Symbolic + Seq2Seq + Function Basis
- Learn operasi dasar: +, -, *, /, %, **, floor, ceil, round, abs, sqrt
- Learn step-by-step reasoning via Seq2Seq
- Learn operation confidence via Function Basis
- Parameters: ~100K (powerful + learnable)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import re
import math


class MathTokenizer:
    """Tokenize math expressions"""
    
    def __init__(self):
        self.tokens = [
            '<pad>', '<start>', '<end>', '<unk>',
            '<step>', '<result>', '<op>', '<func>',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.',
            '+', '-', '*', '/', '%', '**', '(', ')',
            'floor', 'ceil', 'round', 'abs', 'sqrt',
            '=', '...'
        ]
        self.token2id = {t: i for i, t in enumerate(self.tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(self.tokens)
    
    def tokenize(self, expr):
        """Tokenize expression string"""
        expr = str(expr).replace(' ', '')
        pattern = r'(\d+\.?\d*|floor|ceil|round|abs|sqrt|\*\*|[+\-*/%()<>=])'
        matches = re.findall(pattern, expr)
        return ['<start>'] + matches + ['<end>']
    
    def encode(self, expr, max_len=50):
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


class SymbolicLayer(nn.Module):
    """Learn symbolic operations: +, -, *, /, floor, etc"""
    
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Learn operation embeddings
        self.num_ops = 11  # +, -, *, /, %, **, floor, ceil, round, abs, sqrt
        
        # Each operation has learnable transformation
        self.op_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(self.num_ops)
        ])
        
        # Learn operation selection
        self.op_selector = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_ops),
            nn.Softmax(dim=-1)
        )
        
        self.op_names = ['+', '-', '*', '/', '%', '**', 'floor', 'ceil', 'round', 'abs', 'sqrt']
    
    def forward(self, a, b):
        """
        a, b: (batch_size, 1) operands
        output: (batch_size, 1) - weighted combination of operations
        """
        operands = torch.cat([a, b], dim=-1)  # (batch_size, 2)
        
        # Get weights untuk setiap operation
        op_weights = self.op_selector(operands)  # (batch_size, num_ops)
        
        # Compute setiap operation
        results = []
        for i, transform in enumerate(self.op_transforms):
            result = transform(operands)  # (batch_size, 1)
            results.append(result)
        
        results = torch.cat(results, dim=-1)  # (batch_size, num_ops)
        
        # Weighted sum
        output = torch.sum(results * op_weights, dim=-1, keepdim=True)  # (batch_size, 1)
        
        return output, op_weights


class PositionalEncoding(nn.Module):
    """Add positional information to embeddings"""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoder(nn.Module):
    """Encode math expression structure"""
    
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_ff=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return x


class TransformerDecoder(nn.Module):
    """Decode reasoning steps"""
    
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_ff=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.embedding(tgt)
        tgt = self.pos_encoding(tgt)
        output = self.transformer(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)
        logits = self.output_layer(output)
        return logits


class FunctionBasisLayer(nn.Module):
    """Learn weighted basis functions untuk approximation"""
    
    def __init__(self, input_dim=64, num_basis=16, output_dim=1):
        super().__init__()
        self.num_basis = num_basis
        
        # Basis function centers
        self.centers = nn.Parameter(torch.randn(num_basis, input_dim))
        
        # Basis function widths
        self.widths = nn.Parameter(torch.ones(num_basis))
        
        # Output weights
        self.weights = nn.Linear(num_basis, output_dim)
        
        self._init_centers()
    
    def _init_centers(self):
        """Initialize centers evenly"""
        for i in range(self.num_basis):
            self.centers.data[i] = torch.randn(self.centers.size(1)) * (i + 1) / self.num_basis
    
    def forward(self, x):
        """
        x: (batch_size, input_dim)
        output: (batch_size, 1)
        """
        # Compute RBF
        dists = torch.norm(x.unsqueeze(1) - self.centers.unsqueeze(0), dim=2)  # (batch_size, num_basis)
        basis_vals = torch.exp(-self.widths.abs() * dists ** 2)  # (batch_size, num_basis)
        
        # Weighted combination
        output = self.weights(basis_vals)  # (batch_size, 1)
        
        return output


class HybridMathAI(nn.Module):
    """Complete hybrid model"""
    
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, hidden_dim=128):
        super().__init__()
        
        # Components
        self.encoder = TransformerEncoder(vocab_size, d_model, nhead, num_layers)
        self.decoder = TransformerDecoder(vocab_size, d_model, nhead, num_layers)
        self.symbolic_layer = SymbolicLayer(hidden_dim)
        self.function_basis = FunctionBasisLayer(d_model, num_basis=16, output_dim=1)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model + 1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU()
        )
        
        # Output heads
        self.answer_head = nn.Linear(64, 1)
        self.confidence_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, src, tgt, operand_a=None, operand_b=None):
        """
        src: (batch_size, src_len) - encoded expression
        tgt: (batch_size, tgt_len) - target sequence
        operand_a, b: (batch_size, 1) - extracted operands
        """
        # Encode expression
        memory = self.encoder(src)  # (batch_size, src_len, d_model)
        
        # Get context vector (mean pooling)
        context = memory.mean(dim=1)  # (batch_size, d_model)
        
        # Decode reasoning steps
        decoder_output = self.decoder(tgt, memory)  # (batch_size, tgt_len, vocab_size)
        
        # Learn symbolic operations (if operands provided)
        if operand_a is not None and operand_b is not None:
            symbolic_result, op_weights = self.symbolic_layer(operand_a, operand_b)
        else:
            symbolic_result = torch.zeros(src.size(0), 1, device=src.device)
            op_weights = None
        
        # Learn function basis approximation
        basis_result = self.function_basis(context)  # (batch_size, 1)
        
        # Fuse all information
        fused = torch.cat([context, symbolic_result, basis_result], dim=-1)
        fused = self.fusion(fused)
        
        # Generate answer and confidence
        answer = self.answer_head(fused)
        confidence = self.confidence_head(fused)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'decoder_logits': decoder_output,
            'op_weights': op_weights,
            'context': context
        }
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MathExpressionGenerator:
    """Generate valid math problems dengan multiple operators"""
    
    OPERATORS = {
        '+': lambda a, b: (a + b, f"{a}+{b}"),
        '-': lambda a, b: (a - b, f"{a}-{b}"),
        '*': lambda a, b: (a * b, f"{a}*{b}"),
        '/': lambda a, b: (a / b if b != 0 else None, f"{a}/{b}") if b != 0 else (None, None),
        '%': lambda a, b: (a % b if b != 0 else None, f"{a}%{b}") if b != 0 else (None, None),
        '**': lambda a, b: (a ** b if not (a > 10 or b > 5) else None, f"{a}**{b}") if not (a > 10 or b > 5) else (None, None),
    }
    
    FUNCTIONS = {
        'floor': lambda a, b: (int(np.floor(a / b)) if b != 0 else None, f"floor({a}/{b})"),
        'ceil': lambda a, b: (int(np.ceil(a / b)) if b != 0 else None, f"ceil({a}/{b})"),
        'round': lambda a, b: (round(a / b) if b != 0 else None, f"round({a}/{b})"),
        'abs': lambda a, b: (abs(a - b), f"abs({a}-{b})"),
        'sqrt': lambda a, b: (np.sqrt(a) if a >= 0 else None, f"sqrt({a})"),
    }
    
    @staticmethod
    def generate(difficulty=1, include_functions=True):
        """Generate random problem"""
        if include_functions and random.random() < 0.3:
            # Function-based
            func_name = random.choice(list(MathExpressionGenerator.FUNCTIONS.keys()))
            func = MathExpressionGenerator.FUNCTIONS[func_name]
            
            a = random.randint(1, 100 * difficulty)
            b = random.randint(1, 50 * difficulty) if func_name in ['floor', 'ceil', 'round', 'abs'] else None
            
            if b is not None:
                result, expr = func(a, b)
            else:
                result, expr = func(a, None)
            
            operand_a, operand_b = a, b
        else:
            # Operator-based
            op = random.choice(list(MathExpressionGenerator.OPERATORS.keys()))
            op_func = MathExpressionGenerator.OPERATORS[op]
            
            a = random.randint(1, 100 * difficulty)
            b = random.randint(1, 100 * difficulty)
            
            result, expr = op_func(a, b)
            operand_a, operand_b = a, b
        
        if result is None or expr is None:
            return MathExpressionGenerator.generate(difficulty, include_functions)
        
        return expr, float(result), operand_a, operand_b


class MathDataset(Dataset):
    """Training dataset"""
    
    def __init__(self, tokenizer, num_samples=5000, difficulty=1):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.difficulty = difficulty
        self.data = self._generate_data()
    
    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            expr, answer, op_a, op_b = MathExpressionGenerator.generate(self.difficulty)
            
            src = self.tokenizer.encode(expr)
            
            # Target: <step> answer <result>
            target_text = f"<step> {expr} = {answer} <result>"
            tgt = self.tokenizer.encode(target_text, max_len=100)
            
            data.append({
                'src': src,
                'tgt': tgt,
                'answer': torch.tensor([answer], dtype=torch.float32),
                'operand_a': torch.tensor([op_a], dtype=torch.float32),
                'operand_b': torch.tensor([op_b], dtype=torch.float32),
            })
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return (item['src'], item['tgt'], item['operand_a'], item['operand_b'], item['answer'])


class MathAITrainer:
    """Trainer untuk hybrid model"""
    
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, dataloader, optimizer, criterion_answer, criterion_decoder):
        self.model.train()
        total_loss = 0
        
        for src, tgt, op_a, op_b, answer in dataloader:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            op_a = op_a.to(self.device)
            op_b = op_b.to(self.device)
            answer = answer.to(self.device)
            
            # Decoder input = tgt tanpa token terakhir
            decoder_input = tgt[:, :-1]
            decoder_target = tgt[:, 1:]
            
            # Forward pass
            output = self.model(src, decoder_input, op_a, op_b)
            
            # Calculate losses
            answer_loss = criterion_answer(output['answer'], answer)
            
            decoder_logits = output['decoder_logits']
            decoder_logits = decoder_logits.reshape(-1, self.tokenizer.vocab_size)
            decoder_target = decoder_target.reshape(-1)
            decoder_loss = criterion_decoder(decoder_logits, decoder_target)
            
            loss = answer_loss + 0.5 * decoder_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader, criterion_answer, criterion_decoder):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for src, tgt, op_a, op_b, answer in dataloader:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                op_a = op_a.to(self.device)
                op_b = op_b.to(self.device)
                answer = answer.to(self.device)
                
                decoder_input = tgt[:, :-1]
                decoder_target = tgt[:, 1:]
                
                output = self.model(src, decoder_input, op_a, op_b)
                
                answer_loss = criterion_answer(output['answer'], answer)
                
                decoder_logits = output['decoder_logits']
                decoder_logits = decoder_logits.reshape(-1, self.tokenizer.vocab_size)
                decoder_target = decoder_target.reshape(-1)
                decoder_loss = criterion_decoder(decoder_logits, decoder_target)
                
                loss = answer_loss + 0.5 * decoder_loss
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, num_epochs=100, batch_size=32, lr=0.001, difficulty=1):
        print(f"\n{'='*90}")
        print(f"Starting Hybrid Math AI Training (difficulty={difficulty})")
        print(f"Model Parameters: {self.model.count_parameters():,}")
        print(f"{'='*90}\n")
        
        train_dataset = MathDataset(self.tokenizer, num_samples=5000, difficulty=difficulty)
        val_dataset = MathDataset(self.tokenizer, num_samples=1000, difficulty=difficulty)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion_answer = nn.MSELoss()
        criterion_decoder = nn.CrossEntropyLoss(ignore_index=0)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)
        
        best_val_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion_answer, criterion_decoder)
            val_loss = self.evaluate(val_loader, criterion_answer, criterion_decoder)
            scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"\nTraining selesai! Best Val Loss: {best_val_loss:.6f}\n")
    
    def predict(self, expr):
        """Predict answer dengan full reasoning"""
        self.model.eval()
        
        with torch.no_grad():
            src = self.tokenizer.encode(expr).unsqueeze(0).to(self.device)
            
            # Create dummy operands (will be extracted in real use)
            op_a = torch.tensor([[1.0]], device=self.device)
            op_b = torch.tensor([[1.0]], device=self.device)
            
            # Create target sequence
            tgt = torch.tensor([[self.tokenizer.token2id['<start>']]], dtype=torch.long, device=self.device)
            
            # Forward pass
            output = self.model(src, tgt, op_a, op_b)
            
            answer = output['answer'].item()
            confidence = output['confidence'].item()
            
            return answer, confidence


class MathAIGame:
    """Demo dan testing"""
    
    def __init__(self, trainer, device='cpu'):
        self.trainer = trainer
        self.device = device
    
    def test(self, num_tests=20, difficulty=1):
        """Test model accuracy"""
        print(f"\n{'='*90}")
        print(f"Testing Hybrid Math AI ({num_tests} problems, difficulty={difficulty})")
        print(f"{'='*90}\n")
        
        correct = 0
        total_error = 0
        
        for i in range(num_tests):
            expr, true_answer, op_a, op_b = MathExpressionGenerator.generate(difficulty)
            
            pred_answer, confidence = self.trainer.predict(expr)
            error = abs(pred_answer - true_answer)
            
            is_correct = error < 0.1
            if is_correct:
                correct += 1
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {i+1:2d}. {expr:30s} | True: {true_answer:8.2f} | Pred: {pred_answer:8.2f} | Conf: {confidence:.2%}")
            
            total_error += error
        
        accuracy = (correct / num_tests) * 100
        avg_error = total_error / num_tests
        
        print(f"\n{'='*90}")
        print(f"Results: {correct}/{num_tests} correct ({accuracy:.1f}%) | Avg Error: {avg_error:.4f}")
        print(f"{'='*90}\n")
    
    def interactive(self):
        """Interactive mode"""
        print(f"\n{'='*90}")
        print("Interactive Mode")
        print("Examples: '5+3', '10*2', 'floor(10/3)', 'abs(5-10)'")
        print(f"{'='*90}\n")
        
        while True:
            expr = input("Expression (or 'quit'): ").strip()
            if expr.lower() == 'quit':
                break
            
            pred_answer, confidence = self.trainer.predict(expr)
            print(f"Answer: {pred_answer:.4f}")
            print(f"Confidence: {confidence:.2%}\n")


def main():
    print("\n" + "="*90)
    print("Hybrid Math AI - Symbolic + Seq2Seq + Function Basis")
    print("="*90)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    tokenizer = MathTokenizer()
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}\n")
    
    model = HybridMathAI(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        hidden_dim=128
    )
    print(f"Model Parameters: {model.count_parameters():,}\n")
    
    trainer = MathAITrainer(model, tokenizer, device=device)
    game = MathAIGame(trainer, device)
    
    while True:
        print("\nMain Menu:")
        print("1. Train AI")
        print("2. Test AI")
        print("3. Interactive Mode")
        print("4. Exit")
        
        choice = input("\nPilih (1-4): ").strip()
        
        if choice == '1':
            difficulty = int(input("Difficulty (1-5, default 1): ") or "1")
            epochs = int(input("Epochs (default 50): ") or "50")
            batch_size = int(input("Batch size (default 32): ") or "32")
            trainer.train(num_epochs=epochs, batch_size=batch_size, difficulty=difficulty)
        
        elif choice == '2':
            difficulty = int(input("Difficulty (1-5, default 1): ") or "1")
            num_tests = int(input("Number of tests (default 20): ") or "20")
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
