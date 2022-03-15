
from importlib_resources import path
from pathlib import Path
import torch
import random
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
from sacremoses import MosesDetokenizer, MosesTokenizer

from preprocessor import RESOURCES_DIR,tokenize, detokenize
# from main import parse_arguments
#from preprocessor import yield_sentence_pair
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL = 't5-base'
REPO_DIR = Path(__file__).resolve().parent
RESOURCES_DIR = REPO_DIR / 'resources'
DATASETS_DIR = RESOURCES_DIR / 'datasets'
WIKI_DATASET = 'wiki'
complex_train_filepath = DATASETS_DIR / WIKI_DATASET / 'wiki.train_small.complex'
simple_train_filepath = DATASETS_DIR / WIKI_DATASET / 'wiki.train_small.simple'

t5_tokenizer = T5Tokenizer.from_pretrained(MODEL)
t5_config = T5Config.from_pretrained(MODEL)
t5_model = T5ForConditionalGeneration.from_pretrained(MODEL).to(DEVICE)

def mask_span(sentence, tokenizer, ratio = 0.1):
    def verify(mask_span_start_id, span_length, masked_positions, seq_len):
        flag = True
        
        if mask_span_start_id-1 in masked_positions:
            return False
        
        for i in range(span_length+1):
            if mask_span_start_id+i in masked_positions or mask_span_start_id+i >= seq_len:
                flag = False
        return flag
    
    # with MosesTokenizer('en') as tokenize:
    #     input_tokens = tokenize(sentence)
    input_tokens = tokenize(sentence)
    seq_len = len(input_tokens)
    
    sample_prob = torch.ones(seq_len)
    sample_prob = sample_prob/torch.sum(sample_prob)
    
    num_spans = max(1, random.randint(round(seq_len*ratio)-1, round(seq_len*ratio)+1))
    
    masked_positions = []
    masked_start_positions = []
    masked_span_lengths = []
    total_span_length = 0
    
    for i in range(num_spans):
        span_length = random.randint(1,6)
        ### get span start id: tensor.size = 1
        mask_span_start_id = sample_prob.multinomial(1)
        trials = 0
        
        while not verify(mask_span_start_id,span_length,masked_positions, seq_len) and trials<=10:
            mask_span_start_id = sample_prob.multinomial(1)
            trials+=1
        if trials>=10:
            break
        
        for i in range(span_length):
            masked_positions.append(mask_span_start_id+i)
        masked_start_positions.append(mask_span_start_id)
        masked_span_lengths.append(span_length)
        total_span_length += span_length
    
    new_tokens = []
    span_idx = 0
    for idx in range(seq_len):
        if idx in masked_start_positions:
            new_tokens.append('<extra_id'+str(span_idx)+'>')
            span_idx+=1
        elif idx in masked_positions:
            continue
        else:
            new_tokens.append(input_tokens[idx])
    # with MosesDetokenizer('en') as detokenize:
    #     new_tokens = detokenize(new_tokens)
    new_tokens = detokenize(new_tokens)
    return new_tokens, total_span_length

def build_filled_inputs(original_tokens, filled_tokens):
    valid = True
    target_tokens = []
    input_tokens = []
    filling_index = 0
    
    for token in original_tokens:
        if token.startswith('<extra_id'):
            if filling_index >= len(filled_tokens):
                valid = False
                break
            if filled_tokens[filling_index].startswith('<extra_id'):
                valid = False
            while not filled_tokens[filling_index].startswith('<extra_id'):
                target_tokens.append(filled_tokens[filling_index])
                input_tokens.append(filled_tokens[filling_index])
                filling_index += 1
                
                if filling_index >= len(filled_tokens):
                    valid = False
                    break
            
            if filling_index >= len(filled_tokens):
                valid = False
                break
            target_tokens.append(filling_index)
            filling_index += 1
        else:
            input_tokens.append(token)
    return target_tokens, input_tokens, valid

def generate_one_batch(sentences, tokenizer, t5_model):
    batch_size = len(sentences)
    
    masked_texts = []
    max_length = 0
    
    for text in sentences:
        masked_text, length = mask_span(text.strip(), tokenizer)
        max_length = max(max_length, length)
        masked_texts.append(masked_text)
    
    input_ids = tokenizer.batch_encode_plus(
        masked_texts,
        add_special_tokens = True,
        return_tensors = 'pt',
        pad_to_max_length = True
    )
    
    outputs = t5_model.generate(
        input_ids = input_ids.to(DEVICE),
        attention_mask = input_ids['attention_mask'].to(DEVICE),
        do_sample = True,
        top_p =0.9,
        num_return_sequences = 10,
        max_length = round(max_length*2)
    )
    # 10 is the num_return_sequence
    outputs = outputs.reshape(batch_size, 10,-1).data
    
    input_text = []
    original_text = []
    masked_input = []
    output_text = []
    
    for i in range(batch_size):
        output_i = tokenizer.convert_ids_to_tokens(outputs[i,0,2:])
        original_inputs = tokenizer.convert_ids_to_tokens(input_ids['input_ids'][i])
        
        target_tokens, input_tokens, valid = build_filled_inputs(
            original_inputs, output_i
        )
        if valid:
            input_text.append(' '.join(input_tokens))
            original_text.append(' '.join(tokenizer.tokenize(sentences[i])))
            masked_input.append(' '.join(original_inputs))
            output_text.append(' '.join(target_tokens))
    
    return original_text, masked_input, output_text, input_text


def yield_sentence_pair(file1, file2):
    with Path(file1).open('r') as f1, Path(file2).open('r') as f2:
        for line1, line2 in zip(f1,f2):
            yield line1.rstrip(), line2.rstrip()

            
for complex_sent, simple_sent in yield_sentence_pair(complex_train_filepath, simple_train_filepath):
    original_text, masked_input, output_text, input_text = generate_one_batch(complex_sent, t5_tokenizer, t5_model)
    print('original text: ',original_text)
    print('masked input: ', masked_input)
    print('output text: ', output_text)
    print('input text: ', input_text)
    break
        
                


