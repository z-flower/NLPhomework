import os, json, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
import sentencepiece as spm
import sacrebleu
import matplotlib.pyplot as plt

########################################
# 全局配置
########################################
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMB_DIM = 256
HID_DIM = 512
NUM_LAYERS = 2

RNN_TYPE = "lstm"   # lstm / gru
ATTN_TYPE = "bahdanau"

LR = 1e-3
EPOCHS = 30
PATIENCE = 5

MAX_LEN = 100
SPM_VOCAB = 8000

########################################
# SentencePiece
########################################
def train_spm():
    if os.path.exists("spm.model"):
        return
    with open("spm.txt", "w", encoding="utf8") as f:
        for split in ["train", "valid"]:
            for line in open(f"./data/{split}.jsonl", encoding="utf8"):
                obj = json.loads(line)
                f.write(obj["en"] + "\n")
                f.write(obj["zh_hy"] + "\n")

    spm.SentencePieceTrainer.Train(
        input="spm.txt",
        model_prefix="spm",
        vocab_size=SPM_VOCAB,
        model_type="bpe",
        character_coverage=1.0
    )

########################################
# 数据加载
########################################
def load_data(path, sp):
    data = []
    for line in open(path, encoding="utf8"):
        obj = json.loads(line)
        data.append((
            sp.EncodeAsIds(obj["en"]),
            sp.EncodeAsIds(obj["zh_hy"])
        ))
    return data

########################################
# 模型定义
########################################
class Encoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB_DIM)
        RNN = nn.LSTM if RNN_TYPE == "lstm" else nn.GRU
        self.rnn = RNN(EMB_DIM, HID_DIM, NUM_LAYERS, batch_first=True)

    def forward(self, x):
        return self.rnn(self.emb(x))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        if ATTN_TYPE == "luong":
            self.W = nn.Linear(HID_DIM, HID_DIM, bias=False)
        elif ATTN_TYPE == "bahdanau":
            self.Ws = nn.Linear(HID_DIM, HID_DIM)
            self.Wh = nn.Linear(HID_DIM, HID_DIM)
            self.v = nn.Linear(HID_DIM, 1, bias=False)

    def forward(self, s, h):
        if ATTN_TYPE == "dot":
            score = torch.bmm(h, s.unsqueeze(2)).squeeze(2)
        elif ATTN_TYPE == "luong":
            score = torch.bmm(self.W(h), s.unsqueeze(2)).squeeze(2)
        else:
            T = h.size(1)
            s = s.unsqueeze(1).repeat(1, T, 1)
            score = self.v(torch.tanh(self.Ws(s) + self.Wh(h))).squeeze(2)

        alpha = F.softmax(score, dim=1)
        return torch.bmm(alpha.unsqueeze(1), h).squeeze(1)


class Decoder(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB_DIM)
        self.attn = Attention()
        RNN = nn.LSTM if RNN_TYPE == "lstm" else nn.GRU
        self.rnn = RNN(EMB_DIM + HID_DIM, HID_DIM, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HID_DIM * 2, vocab)

    def forward(self, tok, hid, enc):
        emb = self.emb(tok).unsqueeze(1)
        s = hid[0][-1] if isinstance(hid, tuple) else hid[-1]
        ctx = self.attn(s, enc)
        out, hid = self.rnn(torch.cat([emb, ctx.unsqueeze(1)], 2), hid)
        out = out.squeeze(1)
        return self.fc(torch.cat([out, ctx], 1)), hid


class Seq2Seq(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.enc = Encoder(vocab)
        self.dec = Decoder(vocab)

    def forward(self, src, trg, tf):
        enc, hid = self.enc(src)
        out = []
        inp = trg[:, 0]
        for t in range(1, trg.size(1)):
            logits, hid = self.dec(inp, hid, enc)
            out.append(logits)
            inp = trg[:, t] if random.random() < tf else logits.argmax(1)
        return torch.stack(out, 1)

########################################
# 解码
########################################
def greedy(model, src, bos, eos):
    enc, hid = model.enc(src)
    inp = torch.tensor([bos], device=DEVICE)
    out = []
    for _ in range(MAX_LEN):
        logit, hid = model.dec(inp, hid, enc)
        nxt = logit.argmax(1).item()
        if nxt == eos: break
        out.append(nxt)
        inp = torch.tensor([nxt], device=DEVICE)
    return out


def beam(model, src, bos, eos, k):
    enc, hid = model.enc(src)
    beams = [([bos], hid, 0.0)]
    for _ in range(MAX_LEN):
        new = []
        for seq, h, sc in beams:
            if seq[-1] == eos:
                new.append((seq, h, sc))
                continue
            inp = torch.tensor([seq[-1]], device=DEVICE)
            logit, nh = model.dec(inp, h, enc)
            lp = F.log_softmax(logit, -1)
            topk = torch.topk(lp, k)
            for i in range(k):
                new.append((seq+[topk.indices[0,i].item()],
                            nh, sc+topk.values[0,i].item()))
        beams = sorted(new, key=lambda x:x[2], reverse=True)[:k]
    return beams[0][0][1:]

########################################
# 单次实验
########################################
def run_experiment(attn, tf_mode):
    global ATTN_TYPE
    ATTN_TYPE = attn

    if tf_mode == "teacher":
        tf_start, tf_end = 1.0, 1.0
    elif tf_mode == "free":
        tf_start, tf_end = 0.0, 0.0
    else:
        tf_start, tf_end = 1.0, 0.5

    sp = spm.SentencePieceProcessor(model_file="spm.model")
    BOS, EOS, PAD = sp.bos_id(), sp.eos_id(), sp.pad_id()

    train = load_data("./data/train.jsonl", sp)
    valid = load_data("./data/valid.jsonl", sp)
    test  = load_data("./data/test.jsonl", sp)

    model = Seq2Seq(sp.get_piece_size()).to(DEVICE)
    opt = Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)

    best, patience = 1e9, 0
    train_losses, valid_losses = [], []

    for ep in range(EPOCHS):
        tf = tf_start - ep*(tf_start-tf_end)/EPOCHS
        model.train()
        tloss = 0
        for s,t in train:
            s = torch.tensor([s], device=DEVICE)
            t = torch.tensor([t], device=DEVICE)
            opt.zero_grad()
            out = model(s,t,tf)
            loss = crit(out.reshape(-1,out.size(-1)), t[:,1:].reshape(-1))
            loss.backward()
            opt.step()
            tloss += loss.item()
        train_losses.append(tloss/len(train))

        model.eval()
        vloss = 0
        with torch.no_grad():
            for s,t in valid:
                s = torch.tensor([s], device=DEVICE)
                t = torch.tensor([t], device=DEVICE)
                out = model(s,t,1.0)
                loss = crit(out.reshape(-1,out.size(-1)), t[:,1:].reshape(-1))
                vloss += loss.item()
        vloss /= len(valid)
        valid_losses.append(vloss)

        if vloss < best:
            best = vloss
            patience = 0
            torch.save(model.state_dict(), "best.pt")
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    plt.figure()
    plt.plot(train_losses,label="Train")
    plt.plot(valid_losses,label="Valid")
    plt.legend()
    plt.title(f"Loss Curve ({attn}, {tf_mode})")
    plt.savefig(f"loss_{attn}_{tf_mode}.png")
    plt.close()

    model.load_state_dict(torch.load("best.pt"))
    refs, g, b4, b8 = [], [], [], []

    for s,t in test:
        s = torch.tensor([s], device=DEVICE)
        refs.append(sp.DecodeIds(t))
        g.append(sp.DecodeIds(greedy(model,s,BOS,EOS)))
        b4.append(sp.DecodeIds(beam(model,s,BOS,EOS,4)))
        b8.append(sp.DecodeIds(beam(model,s,BOS,EOS,8)))

    return {
        "greedy": sacrebleu.corpus_bleu(g,[refs]).score,
        "beam4": sacrebleu.corpus_bleu(b4,[refs]).score,
        "beam8": sacrebleu.corpus_bleu(b8,[refs]).score
    }

########################################
# 主程序
########################################
if __name__ == "__main__":
    train_spm()

    results = {"attention":{}, "training":{}, "decoding":{}}

    for attn in ["dot","luong","bahdanau"]:
        res = run_experiment(attn,"teacher")
        results["attention"][attn] = res["beam4"]

    for mode in ["teacher","free"]:
        res = run_experiment("bahdanau",mode)
        results["training"][mode] = res["beam4"]

    res = run_experiment("bahdanau","scheduled")
    results["decoding"] = {
        "Greedy":res["greedy"],
        "Beam-4":res["beam4"],
        "Beam-8":res["beam8"]
    }

    for name,data in results.items():
        plt.figure()
        plt.bar(data.keys(), data.values())
        plt.title(name.capitalize()+" Comparison")
        plt.ylabel("BLEU")
        plt.savefig(f"{name}_comparison.png")
        plt.close()
