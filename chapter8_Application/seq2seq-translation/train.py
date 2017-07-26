import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import TextDataset
from model.seq2seq import AttnDecoderRNN, DecoderRNN, EncoderRNN

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
lang_dataset = TextDataset()
lang_dataloader = DataLoader(lang_dataset, shuffle=True)
print()

input_size = lang_dataset.input_lang_words
hidden_size = 256
output_size = lang_dataset.output_lang_words
total_epoch = 20

encoder = EncoderRNN(input_size, hidden_size)
decoder = DecoderRNN(hidden_size, output_size, n_layers=2)
attn_decoder = AttnDecoderRNN(hidden_size, output_size, n_layers=2)
use_attn = True

if torch.cuda.is_available():
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    attn_decoder = attn_decoder.cuda()


def showPlot(points):
    plt.figure()
    x = np.arange(len(points))
    plt.plot(x, points)
    plt.show()


def train(encoder, decoder, total_epoch, use_attn):

    param = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(param, lr=1e-3)
    criterion = nn.NLLLoss()
    plot_losses = []
    for epoch in range(total_epoch):
        since = time.time()
        running_loss = 0
        print_loss_total = 0
        total_loss = 0
        for i, data in enumerate(lang_dataloader):
            in_lang, out_lang = data
            if torch.cuda.is_available():
                in_lang = in_lang.cuda()
                out_lang = out_lang.cuda()
            in_lang = Variable(in_lang)  # batch=1, length
            out_lang = Variable(out_lang)

            encoder_outputs = Variable(
                torch.zeros(MAX_LENGTH, encoder.hidden_size))
            if torch.cuda.is_available():
                encoder_outputs = encoder_outputs.cuda()
            encoder_hidden = encoder.initHidden()
            for ei in range(in_lang.size(1)):
                encoder_output, encoder_hidden = encoder(
                    in_lang[:, ei], encoder_hidden)
                encoder_outputs[ei] = encoder_output[0][0]

            decoder_input = Variable(torch.LongTensor([[SOS_token]]))
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
            decoder_hidden = encoder_hidden
            loss = 0
            if use_attn:
                for di in range(out_lang.size(1)):
                    decoder_output, decoder_hidden, decoder_attention = attn_decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, out_lang[:, di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    if torch.cuda.is_available():
                        decoder_input = decoder_input.cuda()
                    if ni == EOS_token:
                        break
            else:
                for di in range(out_lang.size(1)):
                    decoder_output, decoder_hidden = decoder(
                        decoder_input, decoder_hidden)
                    loss += criterion(decoder_output, out_lang[:, di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi[0][0]

                    decoder_input = Variable(torch.LongTensor([[ni]]))
                    if torch.cuda.is_available():
                        decoder_input = decoder_input.cuda()
                    if ni == EOS_token:
                        break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            print_loss_total += loss.data[0]
            total_loss += loss.data[0]
            if (i + 1) % 5000 == 0:
                print('{}/{}, Loss:{:.6f}'.format(
                    i + 1, len(lang_dataloader), running_loss / 5000))
                running_loss = 0
            if (i + 1) % 100 == 0:
                plot_loss = print_loss_total / 100
                plot_losses.append(plot_loss)
                print_loss_total = 0
        during = time.time() - since
        print('Finish {}/{} , Loss:{:.6f}, Time:{:.0f}s'.format(
            epoch + 1, total_epoch, total_loss / len(lang_dataset), during))
        print()
    showPlot(plot_losses)


if use_attn:
    train(encoder, attn_decoder, total_epoch, use_attn=True)
else:
    train(encoder, decoder, total_epoch, use_attn=False)

print('finish training!')
if use_attn:
    torch.save(encoder.state_dict(), './encoder.pth')
    torch.save(attn_decoder.state_dict(), './attn_decoder.pth')
else:
    torch.save(encoder.state_dict(), './encoder.pth')
    torch.save(decoder.state_dict(), './decoder.pth')
