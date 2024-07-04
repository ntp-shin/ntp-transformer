import pandas as pd
import re

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader, random_split
from dataset import BilingualDataset, look_ahead_mask


class My_Dataset:
    def __init__(self, src_data_dir, tgt_data_dir):
        self.src_data_dir = src_data_dir
        self.tgt_data_dir = tgt_data_dir
        self.src_data = None
        self.tgt_data = None
        self.filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

    def read_data(self):
        # Open the source data
        try:
            with open(self.src_data_dir, 'r') as f:
                self.src_data = f.readlines()
        except:
            ValueError ("The source data file does not exist")

        # Open the target data
        try:
            with open(self.tgt_data_dir, 'r') as f:
                self.tgt_data = f.readlines()
        except:
            ValueError ("The target data file does not exist")

        return self.src_data, self.tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

    def preprocess(self, max_src = 200, max_tgt = 200):
        # Define the custom preprocessing function
        def preprocess_util(input_data):
            # Convert all text to lowercase
            lowercase = input_data.lower()
            # Remove newlines and double spaces
            removed_newlines = re.sub("\n|\r|\t", " ",  lowercase)
            removed_double_spaces = ' '.join(removed_newlines.split(' '))
            
            # Add start of sentence and end of sentence tokens
            # s = '[SOS] ' + removed_double_spaces + ' [EOS]'
            s = removed_double_spaces
            return s
        if self.src_data is None and self.tgt_data is None:
            self.read_data()
        assert len(self.src_data) == len(self.tgt_data), "The length of the source data and target data is not equal 01"
        index = []
        for i in range(len(self.src_data)):
            # if len seq > max => split the sequence to subsequences by '.' or '?' or '!'
            # src_data[0] = ['toi la sinh vien', 'toi hoc truong dai hoc bach khoa. Toi hoc nganh cong nghe thong tin. Toi hoc lop 60TH1.']
            # => src_data[0] = ['toi la sinh vien', 'toi hoc truong dai hoc bach khoa', 'Toi hoc nganh cong nghe thong tin', 'Toi hoc lop 60TH1']
            src = self.src_data[i].split(' ')
            tgt = self.tgt_data[i].split(' ')
            src_seqs = []
            tgt_seqs = []
            if len(src) > max_src:
                sub_src = ''
                for s in src:
                    if len(sub_src.split(' ')) < max_src:
                        sub_src += s + ' '
                    else:
                        src_seqs.append(sub_src)
                        sub_src = s + ' '
                src_seqs.append(sub_src)
            if len(tgt) > max_tgt:
                sub_tgt = ''
                for t in tgt:
                    if len(sub_tgt.split(' ')) < max_tgt:
                        sub_tgt += t + ' '
                    else:
                        tgt_seqs.append(sub_tgt)
                        sub_tgt = t + ' '
                tgt_seqs.append(sub_tgt)
            while(len(src_seqs) != len(tgt_seqs)):
                if len(src_seqs) > len(tgt_seqs):
                    # remove the last element of src_seqs
                    print("index: ", i)
                    print("src remove", src_seqs[-1])
                    src_seqs.pop()
                else:
                    # remove the last element of tgt_seqs
                    print("index: ", i)
                    print("tgt remove", tgt_seqs[-1])   
                    tgt_seqs.pop()
            # if len(src_seqs) > 1: Remove src_data[i] and tgt_data[i] and add src_seqs and tgt_seqs to src_data and tgt_data
            if len(src_seqs) > 1:
                index.append(i)
                self.src_data.extend(src_seqs)
                self.tgt_data.extend(tgt_seqs)

        for i in sorted(index, reverse=True):
            del self.src_data[i]
            del self.tgt_data[i]
        # Apply the preprocessing to the train and test datasets
        self.src_data = [preprocess_util(x) for x in self.src_data]
        self.tgt_data = [preprocess_util(x) for x in self.tgt_data]
        # Check len(src_data) == len(tgt_data)
        assert len(self.src_data) == len(self.tgt_data), "The length of the source data and target data is not equal 02"
        # Process the sequence has length greater than max_len_seq
    
                
        return self.src_data, self.tgt_data
    # Concat the src_data and tgt_data to a dictionary {{'en': src_data[i], 'vi': tgt_data[i]}}
    def to_dict(self):
        return [{'en': self.src_data[i], 'vi': self.tgt_data[i]} for i in range(len(self.src_data))]

class My_Tokenizer:
    def __init__(self, max_seq_len: int = 10000, vocab_size: int = 10000):
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

    def tokenizer(self, data):
        # Initialize the tokenizer
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(data, trainer=trainer)
        return tokenizer

if __name__ == '__main__':
    train_src_dir = 'data/train.en'
    train_tgt_dir = 'data/train.vi'

    train_data = My_Dataset(train_src_dir, train_tgt_dir)
    train_src_data, train_tgt_data = train_data.preprocess(200, 200)
    print(train_src_data[:15])
    print('-------------')
    print(train_tgt_data[0])

    # Tokenize the data
    my_tokenizer = My_Tokenizer()
    tokenizer_src = my_tokenizer.tokenizer(train_src_data)
    tokenizer_tgt = my_tokenizer.tokenizer(train_tgt_data)

    # save all vocabularies in tokenizer_SRC to file
    tokenizer_src.save("tokenizer_src.json")


    # print(f"Vocab size of tokenizer_tgt: {tokenizer_tgt.get_vocab_size()}")
    # Test with one sentence
    # sentence_vi = "nếu không ai cưới tôi và tôi cũng không nghĩ họ nên làm vậy , bà vú nói tôi không xinh đẹp , và bạn biết đấy tôi ít khi tốt , ít khi tốt -- nếu không ai cưới tôi tôi sẽ không bận tâm nhiều ; mua một con sóc trong lồng và một chiếc chuồng thỏ nhỏ . nếu không ai cưới tôi nếu không ai cưới tôi nếu không ai cưới tôi nếu không ai cưới tôi nếu không ai cưới tôi tôi sẽ có một căn nhà phía bìa rừng và một con ngựa non của riêng mình một chú cừu non hiền lành sạch sẽ mà tôi có thể đưa xuống phố . và khi tôi thực sự già đi -- 28 hay 29 -- tôi sẽ mua cho mình một cô bé mồ côi và nuôi nấng như con của mình . nếu không ai cưới tôi nếu không ai cưới tôi nếu không ai cưới tôi nếu không ai cưới tôi và , nếu không ai cưới tôi không ai cưới tôi đi nếu không ai cưới tôi không ai cưới tôi đi nếu không ai cưới tôi xin cảm ơn ."
    # sentence_en = "well , if no one ever marries me and i don &apos;t see why they should , nurse says i &apos;m not pretty , and you know i &apos;m seldom good , seldom good -- well , if no one ever marries me i shan &apos;t mind very much ; buy a squirrel in a cage and a little rabbit-hutch . if no one marries me if no one marries me if no one marries me if no one marries me if no one marries me i &apos;ll have a cottage near a wood and a pony all my own a little lamb quite clean and tame that i can take to town . and when i &apos;m really getting old -- and 28 or nine -- buy myself a little orphan girl and bring her up as mine . if no one marries me if no one marries me if no one marries me if no one marries me well , if no one marries me marries me well , if no one marries me marries me well , if no one marries me thank you . "
    # enc_input_tokens = tokenizer_src.encode(sentence_vi).ids
    # dec_input_tokens = tokenizer_tgt.encode(sentence_en).ids
    # print(f"enc_input_tokens: {enc_input_tokens}")
    # print(f"dec_input_tokens: {dec_input_tokens}")
    # print("Length of enc_input_tokens: ", len(enc_input_tokens))
    # print("Length of dec_input_tokens: ", len(dec_input_tokens))
    # # to dict
    # print("To dict:")
    # train_dict = train_data.to_dict()
    # print(train_dict[:5])

    # # To BilingualDataset
    # train_ds = BilingualDataset(train_dict, tokenizer_src, tokenizer_tgt, 'en', 'vi', 200)
    # print(f"train_ds: {train_ds[0]}")
    # print(type(train_ds[0]))