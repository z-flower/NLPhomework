import json, torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sacrebleu
from tqdm import tqdm

DEVICE="cuda" if torch.cuda.is_available() else "cpu"

model=T5ForConditionalGeneration.from_pretrained("t5-small").to(DEVICE)
tok=T5Tokenizer.from_pretrained("t5-small")

def load(path):
    return [json.loads(l) for l in open(path,encoding="utf8")]

train=load("./data/train.jsonl")
test =load("./data/test.jsonl")

opt=torch.optim.AdamW(model.parameters(),lr=3e-4)

for ep in range(3):
    model.train()
    for ex in tqdm(train):
        inp=tok(
            "translate Chinese to English: "+ex["zh_hy"],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)
        lab=tok(
            ex["en"],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).input_ids.to(DEVICE)
        out=model(**inp,labels=lab)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

model.eval()
refs,preds=[],[]
with torch.no_grad():
    for ex in test:
        inp=tok(
            "translate Chinese to English: "+ex["zh_hy"],
            return_tensors="pt"
        ).to(DEVICE)
        gen=model.generate(**inp,max_length=128)
        preds.append(tok.decode(gen[0],skip_special_tokens=True))
        refs.append(ex["en"])

print("T5 BLEU:",sacrebleu.corpus_bleu(preds,[refs]).score)
