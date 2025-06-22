import re
import copy
import json
import random
from tqdm.auto import tqdm
from collections import OrderedDict, namedtuple
from huggingface_tokenizer import HuggingFaceTokenizer

def process_batch(document):
    def preprocessing_document(text):
        processed_text = text.replace('"', '').replace("'", '')
        processed_text = re.sub(r'\s*\([^)]*\)\s*|\s*\[[^]]*\]\s*', ' ', processed_text)
        processed_text = processed_text.lower()

        paragraphs_raw = re.split(r'\n\s*\n+', processed_text)

        all_sentences = []
        for para_text in paragraphs_raw:
            current_para = re.sub(r'\s+', ' ', para_text).strip()
            if not current_para:
                continue

            sentences_raw = re.split(r'(?<=[.:?!])\s*', current_para)

            sentences = []
            for s in sentences_raw:
                s = s.strip()
                if s and s != '.' and(s.endswith('.') or s.endswith(':') or s.endswith('?') or s.endswith('!')):
                    sentences.append(s)
            if sentences:
                all_sentences.extend(sentences)
        return all_sentences
    return {'processed_text': [preprocessing_document(text) for text in document['maintext']]}

def tokenize_batch(batch):
    tokenizer = HuggingFaceTokenizer('bert-base-uncased')
    result = []
    for sentences in batch['processed_text']:
        if sentences:
            tokenized_sentences = [tokenizer.tokenize(sentence) for sentence in sentences]
            result.append(tokenized_sentences)

    return {'tokenized_text': result}

## MLM, NSP task
MlmSequence = namedtuple('mlm_token_sequence', ['mlm_sequence']) 
MlmIndexAndLabel = namedtuple('mlm_pred_positions_and_labels', ['mlm_index', 'mlm_label'])

def get_target_seq_length(max_num_tokens, config):
    target_seq_len = max_num_tokens

    if random.random() < config.short_seq_prob:
        target_seq_len = random.randint(2, max_num_tokens)
    return target_seq_len

def truncate_segment(segment_a, segment_b, max_num_tokens):
    while True:
        total_sequence_len = len(segment_a) + len(segment_b)
        if total_sequence_len <= max_num_tokens: break

        if len(segment_a) > len(segment_b):
            truncate_target = segment_a
            pop_random_head_or_tail(truncate_target, prob=0.5)
        else:
            truncate_target = segment_b
            pop_random_head_or_tail(truncate_target, prob=0.5)
    return segment_a, segment_b

def pop_random_head_or_tail(truncate_target, prob=0.5):
    if random.random() < prob:
        truncate_target.pop(0)
    else:
        truncate_target.pop()

def get_next_sentence(all_sentences, all_sentence_index, sentence_index, segment_b_len):
    if random.random() < 0.5 and sentence_index < len(all_sentences) - 1:
        is_next = True
        next_sentence = all_sentences[sentence_index + 1][:segment_b_len]
    else:
        is_next = False
        while True:
            random_sentence_index = random.choice(all_sentence_index)
            if random_sentence_index != sentence_index and random_sentence_index != sentence_index + 1: break
        next_sentence = all_sentences[random_sentence_index][:segment_b_len]
    return next_sentence, is_next

def apply_mlm_mixed_strategy(segment_pair, maskable_token_idx_list, num_mlm_pred_per_seq_list, vocab):
    mlm_sequence = []
    mlm_pred_positions_and_labels = []

    for sentence_index, sentence in enumerate(segment_pair):
        sentence_mlm_pred_positions_and_labels = []
        mlm_token_sequence = copy.deepcopy(sentence)
        maskable_token_indexes = copy.deepcopy(maskable_token_idx_list[sentence_index])
        num_mlm_pred_per_sequence = num_mlm_pred_per_seq_list[sentence_index]

        while len(maskable_token_indexes) > 0:
            if len(sentence_mlm_pred_positions_and_labels) >= num_mlm_pred_per_sequence:
                break

            random_pos = random.randrange(0, len(maskable_token_indexes))
            random_index = maskable_token_indexes.pop(random_pos)

            if len(sentence_mlm_pred_positions_and_labels) + len(random_index) > num_mlm_pred_per_sequence:
                continue

            masking_decision_rand = random.random()
            for mlm_index in random_index:
                original_token = mlm_token_sequence[mlm_index]

                if masking_decision_rand < 0.8:
                    masked_token = '[MASK]'
                else:
                    if masking_decision_rand < 0.9:
                        masked_token = original_token
                    else:
                        masked_token = random.choice(list(vocab.keys()))

                mlm_token_sequence[mlm_index] = masked_token
                sentence_mlm_pred_positions_and_labels.append(MlmIndexAndLabel(mlm_index=mlm_index, mlm_label=original_token))

        mlm_sequence.append(MlmSequence(mlm_sequence=mlm_token_sequence))
        sorted_mlm_pred_positions_and_labels = sorted(sentence_mlm_pred_positions_and_labels, key=lambda x: x.mlm_index)
        mlm_pred_positions_and_labels.append(sorted_mlm_pred_positions_and_labels)
    return mlm_sequence, mlm_pred_positions_and_labels

def generate_instances(tokenization_documents, vocab, config):
    max_num_tokens = config.max_seq_len - 3
    target_seq_len = get_target_seq_length(max_num_tokens, config)

    all_sentences = []
    for document in tokenization_documents:
        all_sentences.extend(document)

    count_all_sentence = 0
    for i in range(len(tokenization_documents)):
        count_all_sentence += len(tokenization_documents[i])
    assert len(all_sentences) == count_all_sentence

    segment_pair, segment_ids, nsp_labels = [], [], []
    all_sentence_index = list(range(count_all_sentence))
    for sentence_index in tqdm(range(len(all_sentence_index)), desc='1) Generating NSP instances...', position=1, leave=False):
        segment_a_len = len(all_sentences[sentence_index])
        segment_b_len = max(0, target_seq_len - segment_a_len)

        next_sentence, is_next = get_next_sentence(all_sentences, all_sentence_index, sentence_index, segment_b_len)
        segment_a, segment_b = truncate_segment(all_sentences[sentence_index], next_sentence, max_num_tokens)

        tokens = ['[CLS]'] + segment_a + ['[SEP]'] + segment_b + ['[SEP]']
        segment_id = [1] * (len(segment_a) + 2) + [2] * (len(segment_b) + 1)

        segment_pair.append(tokens)
        segment_ids.append(segment_id)
        nsp_labels.append(is_next)
    assert len(all_sentences) == len(segment_pair) == len(segment_ids) == len(nsp_labels)

    maskable_token_idx_list, num_mlm_pred_per_seq_list = [], []
    count_sequence_tokens = []
    for sentence_index, sentence in tqdm(enumerate(segment_pair), desc='2) Generating MLM instances...', position=2, leave=False):
        count_sequence_tokens.append(len(sentence))
        maskable_token_index = []

        for i, token in enumerate(sentence):
            if token in ('[CLS]', '[SEP]'): continue
            if config.whole_word_mask and len(maskable_token_index) > 0 and token.startswith("##"):
                maskable_token_index[-1].append(i)
            else:
                maskable_token_index.append([i])

        maskable_token_idx_list.append(maskable_token_index)
        num_mlm_pred_per_seq_list.append(min(config.max_mlm_pred_per_sequence, max(1, int(round(count_sequence_tokens[sentence_index] * config.mlm_prob)))))

    mlm_sequence, mlm_pred_positions_and_labels = apply_mlm_mixed_strategy(segment_pair, maskable_token_idx_list, num_mlm_pred_per_seq_list, vocab)

    mlm_token_sequences = [m.mlm_sequence for m in mlm_sequence]
    mlm_pred_positions = [[p.mlm_index for p in m] for m in mlm_pred_positions_and_labels]
    mlm_pred_labels    = [[p.mlm_label for p in m] for m in mlm_pred_positions_and_labels]
    return mlm_token_sequences, mlm_pred_positions, mlm_pred_labels, segment_ids, nsp_labels

def create_augmented_instances(tokenizer, tokenization_documents, config):
    vocab = tokenizer.get_vocab()
    all_mlm_token_sequences, all_mlm_pred_positions, all_mlm_pred_labels = [], [], []
    all_segment_ids, all_nsp_labels = [], []

    for i in tqdm(range(config.num_duplication), desc='Augmenting instances...', position=0):
        mlm_token_sequences, mlm_pred_positions,\
            mlm_pred_labels, segment_ids, nsp_labels = generate_instances(tokenization_documents['tokenized_text'], vocab, config)

        all_mlm_token_sequences.extend(mlm_token_sequences)
        all_mlm_pred_positions.extend(mlm_pred_positions)
        all_mlm_pred_labels.extend(mlm_pred_labels)
        all_segment_ids.extend(segment_ids)
        all_nsp_labels.extend(nsp_labels)

    max_len = max(len(seq) for seq in all_mlm_token_sequences)
    assert max_len <= config.max_seq_len
    return all_mlm_token_sequences, all_mlm_pred_positions, all_mlm_pred_labels, all_segment_ids, all_nsp_labels

def print_instance(instances):
    for key, value_list in instances.items():
        instance = value_list[:1]
        print(f'{key}: {instance}\n')

def add_padding_and_save_instances(all_mlm_token_sequences, all_mlm_pred_positions,
                                   all_mlm_pred_labels, all_segment_ids, all_nsp_labels,
                                   tokenizer, config):
    padding_token_id = tokenizer.get_token_id('[PAD]')

    instances = OrderedDict()
    instances['input_ids'] = []
    instances['input_mask'] = []
    instances['segment_ids'] = []
    instances['next_sentence_labels'] = []
    instances['mlm_pred_positions'] = []
    instances['mlm_pred_label_ids'] = []
    instances['mlm_weights'] = []

    #
    input_ids_list, input_mask_list = [], []
    for instance_index, instance in enumerate(all_mlm_token_sequences):
        input_id = tokenizer.convert_tokens_to_ids(instance)
        input_mask = [1] * len(input_id)

        assert len(input_id) <= config.max_seq_len

        while len(input_id) != config.max_seq_len:
            input_id.append(padding_token_id)  # padding_token_id: integer 0
            input_mask.append(0)
            all_segment_ids[instance_index].append(padding_token_id)
        input_ids_list.append(input_id)
        input_mask_list.append(input_mask)

    instances['input_ids'] = input_ids_list
    instances['input_mask'] = input_mask_list
    instances['segment_ids'] = all_segment_ids

    #
    next_sentence_labels_list = []
    for nsp_label in all_nsp_labels:
        next_sentence_labels_list.append(nsp_label)

    instances['next_sentence_labels'] = next_sentence_labels_list

    mlm_pred_positions_list, mlm_pred_labels_ids_list, mlm_weights_list = [], [], []
    for pred_positions_index, pred_positions in enumerate(all_mlm_pred_positions):
        mlm_pred_labels_id = tokenizer.convert_tokens_to_ids(all_mlm_pred_labels[pred_positions_index])
        mlm_weights_list.append([1.0] * len(mlm_pred_labels_id))

        while len(pred_positions) < config.max_mlm_pred_per_sequence:
            pred_positions.append(padding_token_id)
            mlm_pred_labels_id.append(padding_token_id)
            mlm_weights_list[pred_positions_index].append(0.0)
        mlm_pred_positions_list.append(pred_positions)
        mlm_pred_labels_ids_list.append(mlm_pred_labels_id)

    instances['mlm_pred_positions'] = mlm_pred_positions_list
    instances['mlm_pred_label_ids'] = mlm_pred_labels_ids_list
    instances['mlm_weights'] = mlm_weights_list

    print_instance(instances)

    with open(config.output_filename, 'w', encoding='utf-8') as f:
        json.dump(instances, f, ensure_ascii=False, indent='\t')
    return instances
