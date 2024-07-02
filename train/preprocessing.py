import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, look_ahead_mask
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
        config: dictionary chứa cấu hình, bao gồm đường dẫn đến file tokenizer
        ds: dataset chứa các câu cần được token hóa
        lang: ngôn ngữ của dữ liệu

        [UNK]: unknown
        Whitespace(): tách từ dựa trên khoảng trắng
        trainer: WordLevelTrainer -> train tokenizer với token đặc biệt:
            + [UNK]: unknown
            + [PAD]:
            + [SOS]: Start of seq
            + [EOS]: End of seq
    """
    # config['tokenizer_file] = '../tokenizer/tokenizer_(0).json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    print(f"Loading dataset with config: {config['lang_src']}{config['lang_tgt']}")
    ds_raw = load_dataset('Helsinki-NLP/opus-100', f'{config["lang_src"]}-{config["lang_tgt"]}', split='test')

    # Build Tokenizers:
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90%: Training, 10% Validation
    train_size = int(0.9 * len(ds_raw))
    val_size = len(ds_raw) - train_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_size, val_size])

    # train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,\
    #                             config['lang_src'], config['lang_tgt'],\
    #                             config['seq_len'])
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
    #                             config['lang_src'], config['lang_tgt'],
    #                             config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target senetence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt