import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Attractor(nn.Module):
    def __init__(self):
        super(Attractor, self).__init__()
        self.encoder = nn.LSTM(input_size=768,
                               hidden_size=768,
                               num_layers=1,
                               batch_first=True)
        self.decoder = nn.LSTM(input_size=768,
                               hidden_size=768,
                               num_layers=1,
                               batch_first=False)
        
    def forward(self, x):
        '''
        _, (h_0, c_0) = Encoder(e_1, ..., e_T)
        _, (h_s, c_s) = Decoder(h_{s-1}, c_{s-1}, 0)
                        here input is zeros
        '''
        # x: shape(batch, seq_len, 768)
        _, (h_n, c_n) = self.encoder(x)
        # h_n: shape(1, B, 768)   --- last time sequence hidden state
        # c_n: shape(1, B, 768)   --- last time sequence cell state
        
        # for decoder I need to have zeros as inputs
        batch_size = h_n.shape[1]
        inputs = torch.zeros(1, batch_size, 768).to(torch.device('cuda:1'))
        
        attractors = []
        
        # run decoder for 3 time steps
        for i in range(3):
            _, (h_n, c_n) = self.decoder(inputs, (h_n, c_n))
            # stack obtained attractors
            attractors = attractors + [h_n]
        
        return torch.cat(attractors, dim=2)
        
class BertAttractor(nn.Module):
    def __init__(self, max_len=64):
        super(BertAttractor, self).__init__()
        self.max_len = max_len
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                  num_labels=3,
                                                                  output_hidden_states=True)
        self.attractor = Attractor()
        
    def forward(self, input_ids, attention_mask):
        # get sentence lengths
        lengths = self.input_lens(attention_mask)
        
        # bert outputs
        cls_output, output_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        final_hidden_states = output_hidden_states[-1]
        
        '''
        cls_output: shape(B, C)
        final_hidden_state: shape(B, T, F)
        '''
        
        # get cls embedding
        cls_embedding = [states[0] for states in final_hidden_states]
        
        # get attended hidden_states
        hidden_states = [states[1: sent_len+1] for states, sent_len in zip(final_hidden_states, lengths)]
        pad_hidden_states = pad_sequence(hidden_states, batch_first=True)
        packed_hidden_states = pack_padded_sequence(pad_hidden_states, lengths, batch_first=True, enforce_sorted=False) 
                
        # get attractors
        attractors = self.attractor(packed_hidden_states)[0]
        
        '''
        cls_embedding: [(768)]_B
        attractors: shape(B, F*3)
        '''
        
        # attractor labels  --- matmul(cls_embed, attractors)
        attractor_embeddings = []
        for embed, att in zip(cls_embedding, attractors):
            # get the matrices aligned
            att_matrix = att.view(768, -1)
            embed_matrix = embed.view(1,-1)
            
            # get the labels
            att_embeds = torch.matmul(embed_matrix, att_matrix)
            
            '''
            att_labels: shape(1, 3)
            '''
            
            attractor_embeddings.append(att_embeds)
        attractor_labels = torch.cat(attractor_embeddings, dim=0)
        
        return cls_output, attractor_labels
    
    def input_lens(self, attention_mask):
        lengths = []
        for masks in attention_mask:
            sent_len = torch.where(masks==0)[0]
            sent_len = sent_len[0] if sent_len.shape[0] else self.max_len
            lengths.append(sent_len - 1)
        return lengths

class SeqBert(nn.Module):
    def __init__(self):
        super(SeqBert, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                                  num_labels=3,
                                                                  output_hidden_states=True)

    def forward(self, input_ids, attention_mask):
        cls_output, output_hidden_states = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return cls_output
