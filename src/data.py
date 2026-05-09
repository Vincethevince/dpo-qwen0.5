"""UltraFeedback (H4 binarized) loader"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def _tokenize_pair(example,tokenizer):
    """Tokenize one (prompt, chosen, rejected) triple and build response masks.
    
    dataset fields: 
        chosen = [{"role":"user", "content":"..."}, {"role": "assistant", "content":"..."}]
        rejected = same shape, same prompt, different assistant content
    """
    prompt_messages = example["chosen"][:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    chosen_text = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
    rejected_text = tokenizer.apply_chat_template(example["rejected"], tokenize=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    chosen_ids = tokenizer(chosen_text, add_special_tokens=False).input_ids
    rejected_ids = tokenizer(rejected_text, add_special_tokens=False).input_ids

    plen = len(prompt_ids)
    valid = (chosen_ids[:plen] == prompt_ids and rejected_ids[:plen] == prompt_ids)

    return {
        "prompt_len":               plen,
        "chosen_input_ids":         chosen_ids,
        "chosen_response_mask":     [0] * plen + [1] * (len(chosen_ids) - plen),
        "rejected_input_ids":       rejected_ids,
        "rejected_response_mask":   [0] * plen + [1] * (len(rejected_ids) - plen),
        "valid":                    valid,
    }

def load_ultrafeedback(
        tokenizer, 
        split:str="train_prefs", 
        max_prompt_len:int=512, 
        max_total_len:int=1024,
        num_proc:int=4):
    """Load HuggingFaceH4/ultrafeedback_binarized dataset, tokenize, mask responses, drop oversize pairs
    """
    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized",split=split)

    dataset = dataset.map(
        lambda ex: _tokenize_pair(ex, tokenizer),
        remove_columns=dataset.column_names,
        num_proc=num_proc,
        desc=f"tokenizing {split}"
    )

    n0 = len(dataset)

    dataset = dataset.filter(
        lambda ex: ex["valid"] and 
                ex["prompt_len"] <= max_prompt_len and
                len(ex["chosen_input_ids"]) <= max_total_len and
                len(ex["rejected_input_ids"]) <= max_total_len
    )

    dataset = dataset.remove_columns(["valid"])
    print(f"[data] {split}: kept {len(dataset)}/{n0} pairs after length/template filter")

    return dataset


def dpo_collate(batch, pad_token_id:int):
    """Pad chosen/rejected (and their response mask) to per-batch max length"""

    def pad(key, fill):
        seqs = [torch.tensor(ex[key], dtype=torch.long) for ex in batch]
        return pad_sequence(seqs, batch_first=True, padding_value=fill)
    
    out = {}
    for side in ("chosen", "rejected"):
        ids = pad(f"{side}_input_ids", pad_token_id)
        mask = pad(f"{side}_response_mask", 0)
        attn = (ids != pad_token_id).long()
        out[f"{side}_input_ids"] = ids
        out[f"{side}_attention_mask"] = attn
        out[f"{side}_response_mask"] = mask
    
    return out

def make_dataloader(dataset, tokenizer, batch_size:int, shuffle: bool):
     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
     return DataLoader(
         dataset,
         batch_size=batch_size,
         shuffle=shuffle,
         collate_fn=lambda b: dpo_collate(b, pad_token_id=pad_id),
         drop_last=shuffle,
     )

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    dataset = load_ultrafeedback(tok)

    ex = dataset[0]
    plen = ex["prompt_len"]
    print(f"\n--- example 0 ---")
    print(f"prompt_len   : {plen}")
    print(f"chosen   len : {len(ex['chosen_input_ids'])}   (response tokens: {sum(ex['chosen_response_mask'])})")
    print(f"rejected len : {len(ex['rejected_input_ids'])} (response tokens: {sum(ex['rejected_response_mask'])})")

    print("\nDecoded prompt portion:")
    print(tok.decode(ex["chosen_input_ids"][:plen]))
    print("\nDecoded chosen response:")
    print(tok.decode(ex["chosen_input_ids"][plen:]))
    print("\nDecoded rejected response:")
    print(tok.decode(ex["rejected_input_ids"][plen:]))

    assert tok.decode(ex["chosen_input_ids"][:plen]) == tok.decode(ex["rejected_input_ids"][:plen])

    loader = make_dataloader(dataset.select(range(4)), tok, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print("\n--- batch shapes ---")
    for k,v in batch.items():
        print(f"    {k:30s} {tuple(v.shape)}")