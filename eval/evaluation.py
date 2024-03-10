import sys
from transformers import BertModel, BertTokenizer
from collections import OrderedDict
from sklearn.metrics.pairwise import cosine_similarity
from math import log
import torch
from tqdm import tqdm
from readability import Readability
import nltk
from nltk.tokenize import sent_tokenize
import re
from collections import Counter, defaultdict
import nltk.data
from nltk.translate.bleu_score import SmoothingFunction
from transformers import BertTokenizer, TFBertModel
import re
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge import Rouge
from nltk.translate import meteor
import numpy as np
import copy
from nltk.translate.bleu_score import sentence_bleu
from itertools import chain
import json

nltk.download('punkt')
nltk.download('wordnet')

# receive from the command line the model name
# if len(sys.argv) < 2:
#     print("Usage: python evaluation.py <model_name>")
#     exit(1)
    
# model_name = sys.argv[1]
json_out = {}

# PATH = '/content/drive/MyDrive/' + model_name + '/'
PATH = 'texts_gpt/'

reference_file_path = PATH + 'reference.txt'
output_files = ['greedy', 'beam', 'topk', 'topp']

output_file_paths = []
for output_file in output_files:
    output_file_path = PATH + output_file + '.txt'
    output_file_paths.append(output_file_path)
    
reference_output = open(reference_file_path).readlines()
n = len(reference_output)

generated_outputs = [] 
reference_outputs = []

for output_file_path in output_file_paths:
    generated_outputs.append(open(output_file_path).readlines())
    reference_outputs.append(['' for _ in range(n)])
    
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
n_gen = len(generated_outputs)
print(generated_outputs)
for i in range(n_gen):
    # n = len(generated_outputs[i])
    for j in range(n):
        sentences = (tokenizer.tokenize(generated_outputs[i][j]))
        
        print("Sentences: ", sentences)
        # remove last sentence if it wasn't completed
        if sentences[-1][-1] not in ('.', '!', '?', '>'):
            sentences = sentences[:-1]
            
        n_sentences = len(sentences)
        
        reference_sentences = (tokenizer.tokenize(reference_output[j]))[:n_sentences]
        
        reference_outputs[i][j] = ' '.join(reference_sentences)
        generated_outputs[i][j] = ' '.join(sentences)

# print(generated_outputs[0][-1])
# print(generated_outputs[1][-1])
# print(generated_outputs[2][-1])
# print(generated_outputs[3][-1])

for i in range(n_gen):
    f_gen = open('data_files_bleurt/' + output_files[i] + '.txt', 'w')
    f_ref = open('data_files_bleurt/' + output_files[i] + '_reference.txt', 'w')
    
    f_gen.write('\n'.join(generated_outputs[i]))
    f_ref.write('\n'.join(reference_outputs[i]))
    
# sanity check
# print("Reference output: ", reference_outputs[0][0])
# print("Generated output: ", generated_outputs[0][0])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # METRICS # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# helper method to get rid of special characters
def clean_tokens(sentence_tokenized):
  for i in range(len(sentence_tokenized)):
    sentence_tokenized[i] = re.sub('[^A-Za-z1-9 ]', '', sentence_tokenized[i])
  return sentence_tokenized

# # # # # # # # # # # # BLEU 1-4 # # # # # # # # # # # # # 
json_out = {}

bleu1, bleu2, bleu3, bleu4 = {}, {}, {}, {}
# bleu1 = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
# bleu2 = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
# bleu3 = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
# bleu4 = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]

reference_output = 0
  
for ds in range(len(generated_outputs)):
    bleu1[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    bleu2[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    bleu3[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    bleu4[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    for hl in range(len(reference_outputs[0])):
        reference_tokens = [word_tokenize(t) for t in clean_tokens(sent_tokenize(reference_outputs[ds][hl]))]
        generated_tokens = [word_tokenize(t) for t in clean_tokens(sent_tokenize(generated_outputs[ds][hl]))]

        if len(reference_tokens) > len(generated_tokens):
            reference_tokens = reference_tokens[:len(generated_tokens)]
        else:
            generated_tokens = generated_tokens[:len(reference_tokens)]
        
        n_sents = len(reference_tokens) 
        for sent in range(n_sents):
        # print("REF::: ", reference_tokens)
        # print("GEN::: ", generated_tokens)
            chencherry = SmoothingFunction()
            bleu_score1 = sentence_bleu([reference_tokens[sent]], generated_tokens[sent], smoothing_function=chencherry.method1, weights=(1, 0))
            bleu_score2 = sentence_bleu([reference_tokens[sent]], generated_tokens[sent], smoothing_function=chencherry.method1, weights=(0.5, 0.5))
            bleu_score3 = sentence_bleu([reference_tokens[sent]], generated_tokens[sent], smoothing_function=chencherry.method1, weights=(0.333, 0.333, 0.334))
            bleu_score4 = sentence_bleu([reference_tokens[sent]], generated_tokens[sent], smoothing_function=chencherry.method1, weights=(0.25, 0.25, 0.25, 0.25))

            bleu1[output_files[ds]][hl] += bleu_score1
            bleu2[output_files[ds]][hl] += bleu_score2
            bleu3[output_files[ds]][hl] += bleu_score3
            bleu4[output_files[ds]][hl] += bleu_score4
        bleu1[output_files[ds]][hl] /= n_sents
        bleu2[output_files[ds]][hl] /= n_sents
        bleu3[output_files[ds]][hl] /= n_sents
        bleu4[output_files[ds]][hl] /= n_sents
        
# avg_bleu1, avg_bleu2, avg_bleu3, avg_bleu4 = [0 for _ in range(len(generated_outputs[0])]
# for ds in range(len(generated_outputs[0])):
#     avg_bleu1[ds] = np.mean(bleu1[ds])
#     avg_bleu2[ds] = np.mean(bleu2[ds])
#     avg_bleu3[ds] = np.mean(bleu3[ds])
#     avg_bleu4[ds] = np.mean(bleu4[ds])

json_out['bleu1'] = bleu1
json_out['bleu2'] = bleu2
json_out['bleu3'] = bleu3
json_out['bleu4'] = bleu4

print('Bleu1: ', bleu1)
print('Bleu2: ', bleu2)
print('Bleu3: ', bleu3)
print('Bleu4: ', bleu4)

# # # # # # # # # # # # ROUGE # # # # # # # # # # # # # 
# https://pypi.org/project/rouge/?ref=blog.paperspace.com
# maybe necessary: pip install rouge

rouge = Rouge()
# rouge_score1 = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
# rouge_score2 = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
# rouge_scoreL = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]

rouge_score1, rouge_score2, rouge_scoreL = {}, {}, {}
    # candidate, reference: generated and ground-truth sentences

for ds in range(len(generated_outputs)):
    rouge_score1[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    rouge_score2[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    rouge_scoreL[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    for hl in range(len(reference_outputs[0])):
        # Tokenize your reference and generated texts into lists of words or tokens
        rscore = rouge.get_scores(generated_outputs[ds][hl], reference_outputs[ds][hl])

        # Access the F-score from rouge-1/2/l
        rouge_score1[output_files[ds]][hl] = rscore[0]["rouge-1"]["f"]
        rouge_score2[output_files[ds]][hl] = rscore[0]["rouge-2"]["f"]
        rouge_scoreL[output_files[ds]][hl] = rscore[0]["rouge-l"]["f"]
    
# avg_rouge1 = [len(generated_outputs[0])] # should be 4
# avg_rouge2 = [len(generated_outputs[0])] # should be 4
# avg_rougeL = [len(generated_outputs[0])] # should be 4
# for ds in range(len(generated_outputs[0])):
#     avg_rouge1[ds] = np.mean(rouge_score1[ds]) # get the mean over all headlines per decoding strategy
#     avg_rouge1[ds] = np.mean(rouge_score1[ds]) # get the mean over all headlines per decoding strategy
#     avg_rouge1[ds] = np.mean(rouge_score1[ds]) # get the mean over all headlines per decoding strategy

print('Rouge-1: ', rouge_score1)
print('Rouge-2: ', rouge_score2)
print('Rouge-L: ', rouge_scoreL)

json_out['rouge1'] = rouge_score1
json_out['rouge2'] = rouge_score2
json_out['rougeL'] = rouge_scoreL

# # # # # # # # # # # # METEOR # # # # # # # # # # # # # 

# meteor_scores = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
meteor_scores = {}

for ds in range(len(generated_outputs)):
    meteor_scores[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    for hl in range(len(reference_outputs[0])):
        reference_tokens = [word_tokenize(t) for t in clean_tokens(sent_tokenize(reference_outputs[ds][hl]))]
        generated_tokens = [word_tokenize(t) for t in clean_tokens(sent_tokenize(generated_outputs[ds][hl]))]

        if len(reference_tokens) > len(generated_tokens):
            reference_tokens = reference_tokens[:len(generated_tokens)]
        else:
            generated_tokens = generated_tokens[:len(reference_tokens)]
        
        n_sents = len(reference_tokens) 
        for sent in range(n_sents):
            meteor_score = meteor([reference_tokens[sent]], generated_tokens[sent])
            meteor_scores[output_files[ds]][hl] += meteor_score
            # # have all references == candidate text without one sentence we want to use for score
            # remaining_sentences = generated_tokens.copy()
            # remaining_sentences.pop(sentence)
            # hypothesis = generated_tokens[sentence]
            # print("SENTENCE: ",hypothesis)
            # print("REMAINING: ", remaining_sentences)

            # if remaining_sentences != []:
            #     hypothesis = word_tokenize(hypothesis)
            #     remaining_sentences = [word_tokenize(element) for element in remaining_sentences]

            #     meteor_score = round(meteor(remaining_sentences, hypothesis), 4)
            #     meteor_scores[ds][hl] += meteor_score
                
        meteor_scores[output_files[ds]][hl] /= n_sents

# avg_meteor = [len(generated_outputs[0])] # should be 4
# for ds in range(len(generated_outputs[0])):
#     avg_meteor[ds] = np.mean(meteor_scores[ds]) # get the mean over all headlines per decoding strategy

print('METEOR: ', meteor_scores)

json_out['meteor'] = meteor_scores

with open(PATH + 'metrics_gpt_reference.json', 'w') as f:
    json.dump(json_out, f)

json_out = {}
# # # # # # # # # # # # SELF-BLEU # # # # # # # # # # # # # 

def get_bleu_score(sentence, remaining_sentences):
    lst = []
    chencherry = SmoothingFunction()
    for i in remaining_sentences:
        bleu = sentence_bleu([sentence], i, smoothing_function=chencherry.method1)
        lst.append(bleu)
    return lst


# sentences - list of sentences generated by NLG system -> in our case candidate_tokens

sbleu = {}
#sbleu = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]

for ds in range(len(generated_outputs)):
    sbleu[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    for hl in range(len(generated_outputs[0])):
        reference_tokens = [word_tokenize(t) for t in clean_tokens(sent_tokenize(reference_outputs[ds][hl]))]
        generated_tokens = [word_tokenize(t) for t in clean_tokens(sent_tokenize(generated_outputs[ds][hl]))]
        scores = []
        for sentence in range(len(generated_tokens)): # compare each sentence with all other sentences in generated text for a headline
            remaining_sentences = generated_tokens.copy()
            remaining_sentences.pop(sentence)

            bleu = get_bleu_score(generated_tokens[sentence], remaining_sentences) # calculate bleu score
            scores.append(bleu)

        sbleu[output_files[ds]][hl] = np.mean(scores) # mean over bleu score = self bleu, BUT we do not use reference_output in self-bleu

# avg_self_bleu = [len(generated_outputs[0])] # should be 4
# for ds in range(len(generated_outputs[0])):
#     avg_self_bleu[ds] = np.mean(sbleu[ds]) # get the mean over all headlines per decoding strategy

print('Self-Bleu: ', sbleu)

json_out['self_bleu'] = sbleu

# # # # # # # # # # # # N-Gram Repitition Rates aka Dist-N # # # # # # # # # # # # # ==> does not use reference outputs!

# A low diversity score suggests the model suffers from repetition, and a high diversity score means the
# model generated text is lexically diverse. - https://arxiv.org/pdf/2210.15097.pdf Lisa Li, page 3

# code from https://github.com/neural-dialogue-metrics/Distinct-N/tree/main

#helper methods
def pad_sequence(sequence, n, pad_left=False, pad_right=False,
                 left_pad_symbol=None, right_pad_symbol=None):
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    sequence = pad_sequence(sequence, n, pad_left, pad_right,
                            left_pad_symbol, right_pad_symbol)

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

# the real deal
def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    return sum(distinct_n_sentence_level(sentence, n) for sentence in sentences) / len(sentences)

# # # # # # # # # # # # # #  my implementation haha # # # # # # # # # # # # # #
# dist_n_score = [[ 0 for _ in range(len(generated_outputs[0]))] for _ in range(len(generated_outputs))]
dist_n_score = {}
ngram_num = 2

for ds in range(len(reference_outputs)):
    dist_n_score[output_files[ds]] = [0 for _ in range(len(generated_outputs[0]))]
    for hl in range(len(generated_outputs[0])):
        score = distinct_n_corpus_level(clean_tokens(sent_tokenize(generated_outputs[ds][hl])), ngram_num)
        dist_n_score[output_files[ds]][hl] = score # have a score for each (decoding strategy, headline) pair

print('Dist-N Gram Score: ', dist_n_score)

json_out['dist_n'] = dist_n_score

with open(PATH + 'metrics_gpt_diversity.json', 'w') as f:
    json.dump(json_out, f)
    
json_out = {}
    
# # # # # # # # # # # # BLEURT # # # # # # # # # # # # # 

# PREPARATION:
    # Download and install the model
    # pip install --upgrade pip  # ensures that pip is current
    # git clone https://github.com/google-research/bleurt.git
    # cd bleurt
    # pip install .

    # Download the BLEURT-base checkpoint (pretrained model).
    # wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
    # unzip BLEURT-20.zip
    # python -m bleurt.score_files -candidate_file=../data_files_bleurt_gpt/greedy.txt -reference_file=../data_files_bleurt_gpt/reference.txt bleurt_checkpoint=BLEURT-20 > ../data_files_bleurt_gpt/greedy_BLEURT.txt
    # python -m bleurt.score_files -candidate_file=../data_files_bleurt_gpt/beam.txt   -reference_file=../data_files_bleurt_gpt/reference.txt bleurt_checkpoint=BLEURT-20   > ../data_files_bleurt_gpt/beam_BLEURT.txt
    # python -m bleurt.score_files -candidate_file=../data_files_bleurt_gpt/topk.txt   -reference_file=../data_files_bleurt_gpt/reference.txt bleurt_checkpoint=BLEURT-20   > ../data_files_bleurt_gpt/topk_BLEURT.txt
    # python -m bleurt.score_files -candidate_file=../data_files_bleurt_gpt/topp.txt   -reference_file=../data_files_bleurt_gpt/reference.txt bleurt_checkpoint=BLEURT-20   > ../data_files_bleurt_gpt/topp_BLEURT.txt

bleurt_scores = {}
for i in range(len(output_files)):
    f = open('data_files_bleurt_gpt/' + output_files[i] + '_BLEURT.txt', 'r')
    bleurt_score = f.readlines()
    bleurt_scores[output_files[i]] = []
    s = 0
    n = len(bleurt_score)
    for line in bleurt_score:
        s = float(line.strip())
        bleurt_scores[output_files[i]].append(s)
    # print('BLEURT: ', s / n)

print('BLEURT: ', bleurt_scores)

json_out['bleurt'] = bleurt_scores

# # # # # # # # # # # # COMET # # # # # # # # # # # # # #

# # # # # # # # # # # # SEScore # # # # # # # # # # # # #

# # # # # # # # # # # # # EmbSim # # # # # # # # # # # #

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = TFBertModel.from_pretrained("bert-base-cased")

ref_context_embeddings, gen_context_embeddings = {}, {}

n_gen = len(generated_outputs)
n = len(generated_outputs[0])

for i in range(n_gen):
    ref_context_embeddings[output_files[i]] = []
    gen_context_embeddings[output_files[i]] = []
    for j in range(n):
        ref_sample = reference_outputs[i][j]
        encoded_input = tokenizer(ref_sample, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
        ref_context_embeddings[output_files[i]].append(model(encoded_input))

        gen_sample = generated_outputs[i][j]
        encoded_input = tokenizer(gen_sample, return_tensors='tf', padding='max_length', max_length=512, truncation=True)
        gen_context_embeddings[output_files[i]].append(model(encoded_input))

# sanity check
print("Reference context embeddings: ", ref_context_embeddings[output_files[0]][0])
print("Generated context embeddings: ", gen_context_embeddings[output_files[0]][0])

s = 0

embsim_scores = {}
for i in range(n_gen):
    embsim_scores[output_files[i]] = []
    for j in range(n):
        gen_emb = gen_context_embeddings[output_files[i]][j]['last_hidden_state'].numpy().reshape(1, -1)
        ref_emb = ref_context_embeddings[output_files[i]][j]['last_hidden_state'].numpy().reshape(1, -1)

        s = cosine_similarity(gen_emb, ref_emb)
        res = log(s[0][0])
        embsim_scores[output_files[i]].append(res)
    
print('EmbSim: ', embsim_scores)
json_out['embsim'] = embsim_scores

with open(PATH + 'metrics_gpt_similarity.json', 'w') as f:
    json.dump(json_out, f)

json_out = {}

# # # # # # # # # # # # # NLLTest # # # # # # # # # # # #

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# # # # # # # # # Gunning Fog Index (GFI) # # # # # # # # 
# # # # # # # # # Flesch Reading Ease (FRE) # # # # # # # 
# # # # # Simple Measure of Gobbledygook (SMOG) # # # # #

# PREPARATION:
    # Install readability metrics
    # pip install py-readability-metrics
    # python -m nltk.downloader punkt 

gunning_fog_scores = {}
flesch_scores = {}
smog_scores = {}
    
n_generations = len(generated_outputs)
n_samples_per_generation = len(generated_outputs[0])

for i in range(n_generations):
    gunning_fog_scores[output_files[i]] = []
    flesch_scores[output_files[i]] = []
    smog_scores[output_files[i]] = []
    for j in range(n_samples_per_generation):
        
        text = generated_outputs[i][j]
        reference_sentences = tokenizer.tokenize(text)
        # print(reference_sentences)
        
        text_ = ''
        for num in range(50):
            text_ += reference_sentences[(num) % len(reference_sentences)] + ' '
        
        # print(text_)
        
        r = Readability(text_)
        try:
            gf = r.gunning_fog()
            gunning_fog_scores[output_files[i]].append(gf.score)
            # print(gf.score)
            # print(gf.grade_level)

            f = r.flesch()
            flesch_scores[output_files[i]].append(f.score)
            # print(f.score)
            # print(f.ease)
            # print(f.grade_levels)
            
            s = r.smog(all_sentences=True)
            smog_scores[output_files[i]].append(s.score)
        except:
            print("Error: ", text_)
            gunning_fog_scores[output_files[i]].append(-1)
            flesch_scores[output_files[i]].append(-1)
            smog_scores[output_files[i]].append(-1)
            
print('Gunning Fog Index: ', gunning_fog_scores)
print('Flesch Reading Ease: ', flesch_scores)
print('Simple Measure of Gobbledygook: ', smog_scores)

json_out['gfi'] = gunning_fog_scores
json_out['fre'] = flesch_scores
json_out['smog'] = smog_scores

# # # # # # # Wiener Sachtextformel (WSF) # # # # # # # # 

# PREPARATION:
    # pip install syllables
    # pip install pyphen
    
# def count_sentences_english(text):
#     sentences = sent_tokenize(text)
#     return len(sentences)

# def count_syllables_english(text):
#     dic = pyphen.Pyphen(lang='en_US')
#     text = re.sub("\s+", " ", text)
    
#     number_of_syllables = 0
#     syllables_per_word = defaultdict(int)
#     characters_per_word = defaultdict(int)
 
#     for word in text.split(" "):
# 		# print(word)
#         syllable_counter = 0
#         # hyphenate word
#         syllables = dic.inserted(word)
#         # count first syllable of word
#         syllable_counter += 1
#         # and count the other syllables
#         syllable_counter += syllables.count("-")
#         number_of_syllables += syllable_counter
#         syllables_per_word[syllable_counter] += 1
#         characters_per_word[len(word)] += 1
# 		# print("  Chars: " + str(len(word)))
# 		# print("  Syllables: " + str(syllable_counter))

#     return number_of_syllables, syllables_per_word, characters_per_word

# def wiener_sachtext_formel(ms, sl, iw, es):
# 	wsf = 0.1935 * ms + 0.1672 * sl + 0.1297 * iw - 0.0327 * es - 0.875
# 	return wsf

# wsf = {}
# for i in range(n_generations):
#     wsf[output_files[i]] = []
#     for j in range(n_samples_per_generation):
#         text = generated_outputs[i][j]
        
#         text = re.sub("\s+", " ", text)
#         number_of_words = text.count(" ") + 1
#         demotext = re.sub("\.+", ".", text)
#         number_of_sentences = count_sentences_english(text)
#         number_of_syllables, syllables_per_word, characters_per_word = count_syllables_english(text)
#         avg_sentence_length = number_of_words / number_of_sentences
#         avg_number_of_syllables_per_word = number_of_syllables / number_of_words

#         number_of_words_with_three_or_more_syllables = sum([v for k, v in syllables_per_word.items() if k >= 3])
#         number_of_words_with_six_or_more_characters = sum([v for k, v in characters_per_word.items() if k >= 6])
        
#         wsf_score = wiener_sachtext_formel(
#             number_of_words_with_three_or_more_syllables / number_of_words * 100,
#             avg_sentence_length,
#             number_of_words_with_six_or_more_characters / number_of_words * 100,
#             syllables_per_word[1] / number_of_words * 100)
        
#         wsf[output_files[i]].append(wsf_score)
        
# print('Wiener Sachtextformel: ', wsf)

# json_out['wsf'] = wsf

with open(PATH + 'metrics_gpt_readability.json', 'w') as f:
    json.dump(json_out, f)
'''    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Save results

# with open(PATH + 'results.txt', 'w') as f:
#     f.write('BLEURT: ' + str(bleurt_scores) + '\n')
#     f.write('EmbSim: ' + str(embsim_scores) + '\n')
#     f.write('Gunning Fog Index: ' + str(gunning_fog_scores) + '\n')
#     f.write('Flesch Reading Ease: ' + str(flesch_scores) + '\n')
#     f.write('Simple Measure of Gobbledygook: ' + str(smog_scores) + '\n')
#     f.write('Wiener Sachtextformel: ' + str(wsf_scores) + '\n')
'''