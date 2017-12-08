import random

import torch
from torch.autograd import Variable

from dataset import TextDataset
from model.seq2seq import AttnDecoderRNN, DecoderRNN, EncoderRNN
import matplotlib.pyplot as plt
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
use_attn = True
use_cuda = torch.cuda.is_available()
lang_dataset = TextDataset()
print('*' * 10)


def evaluate(encoder, decoder, in_lang, max_length=MAX_LENGTH):
    if use_cuda:
        in_lang = in_lang.cuda()
    input_variable = Variable(in_lang)
    input_variable = input_variable.unsqueeze(0)
    input_length = input_variable.size(1)
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[:, ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    if use_attn:
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang_dataset.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    else:
        for di in range(max_length):
            decoder_output, decoder_hidden = decoder(decoder_input,
                                                     decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang_dataset.output_lang.index2word[ni])

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    if use_attn:
        return decoded_words, decoder_attentions[:di + 1]
    else:
        return decoded_words


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair_idx = random.choice(list(range(len(lang_dataset))))
        pair = lang_dataset.pairs[pair_idx]
        in_lang, out_lang = lang_dataset[pair_idx]
        print('>', pair[0])
        print('=', pair[1])
        if use_attn:
            output_words, attentions = evaluate(encoder, decoder, in_lang)
        else:
            output_words = evaluate(encoder, decoder, in_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words

encoder = EncoderRNN(input_size, hidden_size)
encoder.load_state_dict(torch.load('./encoder.pth'))
if use_attn:
    decoder = AttnDecoderRNN(hidden_size, output_size, n_layers=2)
    decoder.load_state_dict(torch.load('./attn_decoder.pth'))
else:
    decoder = DecoderRNN(hidden_size, output_size, n_layers=2)
    decoder.load_state_dict(torch.load('./decoder.pth'))

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

evaluateRandomly(encoder, decoder)

if use_attn:
    pair_idx = random.choice(list(range(len(lang_dataset))))
    pairs = lang_dataset.pairs[pair_idx]
    print('>')
    print(pairs[0])
    in_lang, out_lang = lang_dataset[pair_idx]
    output_words, attentions = evaluate(encoder, decoder, in_lang)
    plt.matshow(attentions.cpu().numpy())
    plt.show()