def run_inference(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device) if args.model != 'pretrained' else None
    cfg = ckpt.get('config', {}) if ckpt else {}

    # --- 构建词表 ---
    if args.tokenizer == 'sp':
        # SentencePiece 模式
        assert args.src_model and args.tgt_model, "sp 模式需要 --src_model 和 --tgt_model"
        src_vocab = SubwordVocab(os.path.join(args.src_model, 'zh.model'), lang='zh')
        tgt_vocab = SubwordVocab(os.path.join(args.tgt_model, 'en.model'), lang='en')
    else:
        # word-level 模式
        src_vocab = Vocabulary(lang='zh')
        tgt_vocab = Vocabulary(lang='en')
        if args.src_model and args.tgt_model:
            # 从文件夹加载 .vocab 或 .txt
            with open(os.path.join(args.src_model, 'vocab.txt'), encoding='utf-8') as f:
                src_vocab.itos = [line.strip() for line in f]
            with open(os.path.join(args.tgt_model, 'vocab.txt'), encoding='utf-8') as f:
                tgt_vocab.itos = [line.strip() for line in f]
            src_vocab.stoi = {tok: idx for idx, tok in enumerate(src_vocab.itos)}
            tgt_vocab.stoi = {tok: idx for idx, tok in enumerate(tgt_vocab.itos)}
        elif ckpt:
            # fallback: 从 checkpoint 加载
            itos_src = ckpt['src_vocab']
            itos_tgt = ckpt['tgt_vocab']
            src_vocab.itos = itos_src
            src_vocab.stoi = {tok: idx for idx, tok in enumerate(src_vocab.itos)}
            tgt_vocab.itos = itos_tgt
            tgt_vocab.stoi = {tok: idx for idx, tok in enumerate(tgt_vocab.itos)}
        else:
            raise ValueError("没有提供词表文件或 checkpoint")

    src_tok = src_vocab.tokenizer
    tgt_tok = tgt_vocab.tokenizer
    ds = TranslationDataset(args.data_path, src_tok, tgt_tok, max_len=args.max_len)
    collate = Collate(src_vocab, tgt_vocab, device)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # --- 构建模型 ---
    if args.model == 'rnn':
        encoder = EncoderRNN(len(src_vocab), cfg.get('emb_dim', 256), cfg.get('hid_dim', 256), n_layers=2, dropout=cfg.get('dropout', 0.2))
        attention = Attention(enc_hid_dim=cfg.get('hid_dim', 256), dec_hid_dim=cfg.get('hid_dim', 256), method=cfg.get('attn', 'dot'))
        decoder = DecoderRNN(len(tgt_vocab), cfg.get('emb_dim', 256), cfg.get('hid_dim', 256), cfg.get('hid_dim', 256), n_layers=2, dropout=cfg.get('dropout', 0.2), attention=attention)
        model = Seq2Seq(encoder, decoder, device)
    else:
        model = TransformerNMT(
            src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab), d_model=cfg.get('d_model', 256), nhead=cfg.get('nhead', 4),
            num_encoder_layers=cfg.get('n_encoder', 4), num_decoder_layers=cfg.get('n_decoder', 4), dim_feedforward=cfg.get('ff', 512),
            dropout=cfg.get('dropout', 0.1), norm_type=cfg.get('norm', 'layer'), use_relative_bias=(cfg.get('pos', 'absolute')=='relative'), use_abs_pos=(cfg.get('pos', 'absolute')=='absolute')
        )

    if ckpt:
        model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    sos_idx = tgt_vocab.stoi['<sos>']
    eos_idx = tgt_vocab.stoi['<eos>']

    # --- 推理 ---
    outputs = []
    with torch.no_grad():
        for src, src_mask, tgt_inp, tgt_out in loader:
            if args.model == 'rnn':
                hyps = greedy_decode_rnn(model, src, args.max_len, sos_idx, eos_idx) if args.strategy == 'greedy' else beam_search_rnn(model, src, args.beam, args.max_len, sos_idx, eos_idx)
            else:
                hyps = greedy_decode_transformer(model, src, src_mask, args.max_len, sos_idx, eos_idx) if args.strategy == 'greedy' else beam_search_transformer(model, src, src_mask, args.beam, args.max_len, sos_idx, eos_idx)
            texts = [' '.join([tgt_vocab.itos[i] if i < len(tgt_vocab.itos) else '<unk>' for i in hyp]) for hyp in hyps]
            outputs.extend(texts)

    if args.out_file:
        with open(args.out_file, 'w', encoding='utf-8') as f:
            for line in outputs:
                f.write(line.strip() + '\n')
        print(f"Saved translations to {args.out_file}")
    else:
        for line in outputs[:10]:
            print(line)
# python translate.py --model rnn --tokenizer sp --src_model ./zh --tgt_model ./en --checkpoint ./rnn_checkpoint.pt --data_path test.jsonl