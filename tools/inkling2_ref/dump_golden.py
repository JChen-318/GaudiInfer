"""Golden reference dumps for the GaudiInfer Inkling-2-Mini port (HF transformers 5.24, CPU).

    python3 dump_golden.py OUT_DIR [--long N] [--decode N] [--dtype bfloat16|float32]

Writes OUT_DIR/<case>/: ids.npy, logits.npy (all rows, f32), hs.npy (per layer application output, [48+1, T, H] f32),
intermediates of selected modules (first/each call), and a decode continuation (greedy, with the KV cache) to check
the incremental path: dec_ids.npy, dec_logits.npy ([n, V] f32).
"""
import argparse, os, sys, time
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, Inkling2LoopForConditionalGeneration

M = os.environ.get('INKLING2_MODEL_DIR', '/models/Inkling-2-Mini')


def load(dtype=torch.bfloat16):
    cfg = AutoConfig.from_pretrained(M)
    if hasattr(cfg, 'quantization_config'):
        del cfg.quantization_config
    model = Inkling2LoopForConditionalGeneration.from_pretrained(M, config=cfg, dtype=dtype)
    model.eval()
    return model, AutoTokenizer.from_pretrained(M)


def f32(t):
    return t.detach().float().cpu().numpy()


def run_case(model, tok, out, ids, n_decode, hook_layers):
    os.makedirs(out, exist_ok=True)
    lm = model.model.language_model
    rec = {}
    handles = []

    def hook(name):
        def fn(mod, inp, outp):
            o = outp[0] if isinstance(outp, tuple) else outp
            if isinstance(o, torch.Tensor):
                rec.setdefault(name, []).append(f32(o))
        return fn

    for li in hook_layers:
        layer = lm.layers[li]
        for sub in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.relative_position_bias',
                    'self_attn.sink_proj', 'self_attn', 'post_attention_convolution', 'mlp', 'mlp.gate',
                    'post_mlp_convolution', 'attention_input', 'mlp_input', 'input_layernorm', 'q_input_layernorm',
                    'q_message_gate', 'mlp_message_gate']:
            try:
                m = layer.get_submodule(sub)
            except (AttributeError, TypeError):
                m = None
            if m is not None:
                handles.append(m.register_forward_hook(hook(f'L{li}.{sub}')))
        handles.append(layer.register_forward_hook(hook(f'L{li}.out')))
    handles.append(lm.message_input.register_forward_hook(hook('message_input')))
    handles.append(lm.final_input.register_forward_hook(hook('final_input')))
    handles.append(lm.embed_tokens.register_forward_hook(hook('embed')))

    x = torch.tensor([ids])
    t0 = time.time()
    with torch.no_grad():
        o = model(input_ids=x, use_cache=True, output_hidden_states=True)
    print(f'{out}: prefill T={len(ids)} {time.time() - t0:.1f} s', flush=True)
    for h in handles:
        h.remove()
    np.save(f'{out}/ids.npy', np.array(ids, np.int32))
    np.save(f'{out}/logits.npy', f32(o.logits[0]))
    hs = o.hidden_states
    print('  hidden_states entries', len(hs))
    np.save(f'{out}/hs.npy', np.stack([f32(h[0]) for h in hs]))
    for k, v in rec.items():
        np.save(f'{out}/i_{k}.npy', np.stack(v) if len(set(a.shape for a in v)) == 1 else np.array(v, dtype=object),
                allow_pickle=True)
    # greedy decode continuation with the cache
    cache = o.cache if hasattr(o, 'cache') else o.past_key_values
    nxt = int(o.logits[0, -1].argmax())
    dec_ids, dec_logits = [], []
    pos = len(ids)
    with torch.no_grad():
        for i in range(n_decode):
            dec_ids.append(nxt)
            od = model(input_ids=torch.tensor([[nxt]]), cache=cache, use_cache=True,
                       position_ids=torch.tensor([[pos]]))
            cache = od.cache if hasattr(od, 'cache') else od.past_key_values
            dec_logits.append(f32(od.logits[0, -1]))
            nxt = int(od.logits[0, -1].argmax())
            pos += 1
    np.save(f'{out}/dec_ids.npy', np.array(dec_ids, np.int32))
    np.save(f'{out}/dec_logits.npy', np.stack(dec_logits) if dec_logits else np.zeros((0,)))
    print('  decode:', repr(tok.decode(dec_ids)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--long', type=int, default=1600)
    ap.add_argument('--decode', type=int, default=16)
    ap.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    args = ap.parse_args()
    dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[args.dtype]
    torch.set_num_threads(16)
    model, tok = load(dtype)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, 'keys'):
        ids = ids['input_ids']
    ids = list(ids)
    print('prompt:', repr(tok.decode(ids)))
    run_case(model, tok, f'{args.out}/short', ids, args.decode, [0, 1, 5, 6, 11, 12, 17, 18, 23])
    if args.long:
        # any long text does (the dump stores its token ids): the repository's README
        text = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'README.md')).read() * 4
        body = tok.encode(text)[: args.long]
        msgs = [{"role": "user", "content": tok.decode(body) + "\n\nSummarize the text above in two sentences."}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        if isinstance(ids, dict) or hasattr(ids, 'keys'):
            ids = ids['input_ids']
        run_case(model, tok, f'{args.out}/long', list(ids), args.decode, [0, 5, 11, 23])


if __name__ == '__main__':
    main()
