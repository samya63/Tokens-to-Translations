import torch
import torch.nn.functional as F
import os
from Transformer import Transformer
from dataset import TranslationDataset 

# ==========================================
# 1. Configuration (Must match train.py exactly)
# ==========================================
# ==========================================
# 1. Configuration (Must match train.py exactly)
# ==========================================
d_model = 256      # Changed from 512
n_heads = 8        # This is likely still 8 (since 256 / 32 = 8)
n_blocks = 6       # Check your train.py to ensure this was 6!
max_len = 256      # Changed from 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Loading the Model
# ==========================================
def load_inference_model():
    print("Loading dataset vocabularies...")
    dataset = TranslationDataset("train")
    
    print("Initializing Transformer...")
    model = Transformer(
        d_model=d_model, 
        n_head=n_heads, 
        max_len=max_len, 
        n_blocks=n_blocks, 
        src_vocab_size=dataset.src_vocab_size, 
        tgt_vocab_size=dataset.tgt_vocab_size
    ).to(device)
    
    # Using os.path to ensure it finds the file regardless of terminal location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "checkpoints", "best_model.pt")
    
    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval() # Turns off dropout for inference
        return model, dataset
    else:
        print(f"\nError: Could not find checkpoint at {checkpoint_path}")
        print("Please ensure your 'checkpoints' folder contains 'best_model.pt'.")
        exit()

# ==========================================
# 3. Translation Logic (Multinomial + Temperature)
# ==========================================
def translate(model, dataset, text, temperature=0.7):
    model.eval()
    
    # 1. ENCODING: English string -> Token IDs
    src_tokens = [dataset.src_tokenizer.stoi["<SOS>"]] + \
                 dataset.src_tokenizer.encode(text) + \
                 [dataset.src_tokenizer.stoi["<EOS>"]]
                 
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(device)
    
    # 2. SETUP DECODING
    sos_idx = dataset.tgt_tokenizer.stoi["<SOS>"]
    eos_idx = dataset.tgt_tokenizer.stoi["<EOS>"]
    
    tgt_indices = [sos_idx]
    
    # 3. AUTOREGRESSIVE GENERATION
    for _ in range(50): # Max length cap
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(src_tensor, tgt_tensor)
        
        # Isolate the logits for the very last token generated
        last_logits = logits[0, -1, :] 
        
        if temperature > 0:
            # Multinomial Sampling with Temperature
            scaled_logits = last_logits / temperature
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            # Fallback to Greedy if temperature is exactly 0
            next_token = last_logits.argmax(dim=-1).item()
            
        tgt_indices.append(next_token)
        
        if next_token == eos_idx:
            break
            
    # 4. DECODING: Token IDs -> German string (Excluding SOS and EOS)
    # tgt_indices[1:-1] slices off the SOS at the start and the EOS at the end
    return dataset.tgt_tokenizer.decode(tgt_indices[1:-1])

# ==========================================
# 4. Interactive Terminal Loop
# ==========================================
if __name__ == "__main__":
    model, dataset = load_inference_model()
    
    print("\n" + "="*50)
    print("🌍 English to German Translation CLI 🌍")
    print("="*50)
    print("Type 'exit' or 'quit' at any time to stop.\n")

    while True:
        text = input("English: ")
        
        if text.lower() in ['exit', 'quit']:
            print("Exiting translator. Goodbye!")
            break
            
        if text.strip() == "":
            continue
        
        temp_input = input("Temperature [Press Enter for 0.7]: ")
        try:
            temp = float(temp_input) if temp_input.strip() else 0.7
        except ValueError:
            print("Invalid temperature. Defaulting to 0.7")
            temp = 0.7
        
        translation = translate(model, dataset, text, temperature=temp)
        
        print(f"\nGerman:  {translation}")
        print("-" * 50)