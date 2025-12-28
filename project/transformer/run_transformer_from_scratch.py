import os, json, math, argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import sentencepiece as spm
import sacrebleu
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

############################
# Arguments
############################
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--pos_emb", type=str,
                    choices=["absolute", "none"],
                    default="absolute")
parser.add_argument("--norm", type=str,
                    choices=["layernorm", "rmsnorm"],
                    default="layernorm")

args = parser.parse_args()

BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs

############################
# SentencePiece
############################
def train_spm():
    if os.path.exists("spm.model"):
        return
    with open("spm.txt","w",encoding="utf8") as f:
        for sp in ["train","valid"]:
            for l in open(f"data/{sp}.jsonl",encoding="utf8"):
                o=json.loads(l)
                f.write(o["en"]+"\n")
                f.write(o["zh_hy"]+"\n")
    spm.SentencePieceTrainer.Train(
        input="spm.txt",
        model_prefix="spm",
        vocab_size=8000,
        model_type="bpe",
        character_coverage=1.0
    )
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return self.scale * x / (rms + self.eps)
class IdentityPE(nn.Module):
    def forward(self, x):
        return x

############################
# Positional Encoding
############################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

############################
# Transformer
############################
class TransformerNMT(nn.Module):
    def __init__(self, vocab, d_model, pos_type, norm_type):
        super().__init__()

        self.emb = nn.Embedding(vocab, d_model)

        self.pos = PositionalEncoding(d_model) \
            if pos_type == "absolute" else IdentityPE()

        norm_layer = nn.LayerNorm(d_model) \
            if norm_type == "layernorm" else RMSNorm(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            norm_first=True
        )
        encoder_layer.norm1 = norm_layer
        encoder_layer.norm2 = norm_layer

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=2048,
            norm_first=True
        )
        decoder_layer.norm1 = norm_layer
        decoder_layer.norm2 = norm_layer
        decoder_layer.norm3 = norm_layer

        self.encoder = nn.TransformerEncoder(encoder_layer, 4)
        self.decoder = nn.TransformerDecoder(decoder_layer, 4)

        self.fc = nn.Linear(d_model, vocab)
    def forward(self, src, tgt):
        src = self.pos(self.emb(src))
        tgt = self.pos(self.emb(tgt))
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        return self.fc(out)

############################
# Main
############################
def main():
    train_spm()
    sp = spm.SentencePieceProcessor(model_file="spm.model")
    PAD = sp.pad_id()
    BOS = sp.bos_id()

    def load(path):
        data=[]
        for l in open(path,encoding="utf8"):
            o=json.loads(l)
            data.append((sp.EncodeAsIds(o["en"]),
                         sp.EncodeAsIds(o["zh_hy"])))
        return data

    train_data = load("data/train.jsonl")
    test_data  = load("data/test.jsonl")

    model = TransformerNMT(
        vocab=sp.get_piece_size(),
        d_model=512,
        pos_type=args.pos_emb,
        norm_type=args.norm
    ).to(DEVICE)

    opt = Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    print(f"Training: batch={BATCH_SIZE}, lr={LR}")

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0
        for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
            batch = train_data[i:i+BATCH_SIZE]
            src = torch.tensor([x[0] for x in batch], device=DEVICE)
            tgt = torch.tensor([x[1] for x in batch], device=DEVICE)

            opt.zero_grad()
            out = model(src, tgt[:, :-1])
            loss = crit(out.reshape(-1, out.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {ep+1} | Train Loss = {total_loss/len(train_data):.4f}")

    # BLEU
    model.eval()
    refs, preds = [], []
    with torch.no_grad():
        for src, tgt in test_data:
            src = torch.tensor([src], device=DEVICE)
            ys = torch.tensor([[BOS]], device=DEVICE)
            out = model(src, ys)
            pred = out.argmax(-1).squeeze().tolist()
            refs.append(sp.DecodeIds(tgt))
            preds.append(sp.DecodeIds(pred))

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    print(f"BLEU = {bleu:.2f}")

if __name__ == "__main__":
    main()
