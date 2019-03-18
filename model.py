# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from enums import *
from utils import log_sum_exp
from constants import Constants
from utils import *


class BiLSTM_CRF(nn.Module):
    def __init__(self, word_set_size, tag_to_id, word_embedding_dim, word_lstm_dim, word_cnn_dim,
                 word_lstm_bidirect=True, pre_word_embeds=None, encoder_mode=EncoderSchema.LSTM,
                 char_mode=CharEmbeddingSchema.CNN, char_embedding_dim=25, char_lstm_dim=25, char_lstm_bidirect=True,
                 char_cnn_win=3, char_cnn_output=25, char_to_id=None, use_gpu=False, dropout=0.5, use_crf=True):
        super(BiLSTM_CRF, self).__init__()

        self.word_set_size = word_set_size
        self.tag_to_id = tag_to_id
        self.word_embedding_dim = word_embedding_dim
        self.word_lstm_dim = word_lstm_dim
        self.word_lstm_bidirect = word_lstm_bidirect
        self.char_embedding_dim = char_embedding_dim
        self.char_mode = char_mode
        self.char_lstm_dim = char_lstm_dim
        self.char_lstm_bidirect = char_lstm_bidirect
        self.char_cnn_win = char_cnn_win
        self.char_cnn_output = char_cnn_output
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_crf = use_crf
        self.tag_set_size = len(tag_to_id)
        self.encoder_mode = encoder_mode
        self.word_cnn_dim = word_cnn_dim

        print('char_mode: %s, char_embedding_out: %d, word_lstm_dim: %d, ' %
              (char_mode, char_cnn_output if char_mode == CharEmbeddingSchema.CNN else
              ((char_lstm_dim * 2) if char_lstm_bidirect else char_lstm_dim), word_lstm_dim))

        if char_embedding_dim is not None:
            self.char_embeds = nn.Embedding(len(char_to_id), char_embedding_dim)

            if char_mode == CharEmbeddingSchema.LSTM:
                self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim, num_layers=1,
                                         bidirectional=char_lstm_bidirect, batch_first=True)
                init_lstm_(self.char_lstm)
            if char_mode == CharEmbeddingSchema.CNN:
                self.char_cnn = nn.Conv2d(in_channels=1, out_channels=char_cnn_output,
                                          kernel_size=(char_cnn_win, char_embedding_dim),
                                          padding=(char_cnn_win // 2, 0))
                init_cnn_(self.char_cnn, char_cnn_win, 1, dropout)


        self.word_embeds = nn.Embedding(word_set_size, word_embedding_dim)
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
            self.word_embeds.weight = nn.Parameter(torch.FloatTensor(pre_word_embeds))
            # self.word_embeds.weight.requires_grad = False
        else:
            self.pre_word_embeds = False

        self.dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(word_embedding_dim+char_embedding_dim, word_lstm_dim,
                            bidirectional=word_lstm_bidirect, batch_first=True)
        in_features = word_lstm_dim * (2 if word_lstm_bidirect else 1)
        self.hidden2tag = nn.Linear(in_features, self.tag_set_size)
        init_linear_(self.hidden2tag, in_features, dropout)

        if use_crf:
            self.transitions = nn.Parameter(
                torch.zeros(self.tag_set_size, self.tag_set_size))

            self.transitions.data[tag_to_id[Constants.Tag_Start], :] = Constants.Invalid_Transition
            self.transitions.data[:, tag_to_id[Constants.Tag_End]] = Constants.Invalid_Transition


    def _score_sentence(self, feats, tags, sentence_masks, device):
        """
        Get the CRF score for the ground-truth tags
        :param feats: LSTM output, batch x max_seq x tag
        :param tags: ground-truth tags, batch x max_seq
        :param tags: ground-truth tags, batch x max_seq
        :param sentence_masks: binary (0,1) int matrix, batch x max_seq
        :param device: device info
        :return:
        """

        batch_size, max_seq_length, tag_num = feats.size()

        # batch x tag
        tag_wise_score = torch.gather(feats, 2, tags.view(batch_size, max_seq_length, 1)).squeeze(2)

        pad_ones = torch.ones((batch_size, 1), dtype=torch.long, requires_grad=False)
        tag_start = pad_ones * self.tag_to_id[Constants.Tag_Start]
        tag_end = pad_ones * self.tag_to_id[Constants.Tag_End]

        if self.use_gpu:
            pad_ones = pad_ones.to(device)
            tag_start = tag_start.to(device)
            tag_end = tag_end.to(device)

        pad_start_tags = torch.cat([tag_start, tags], dim=1)
        pad_end_tags = torch.cat([tags, tag_end], dim=1)
        transition_masks = torch.cat([pad_ones, sentence_masks], dim=1)

        transition_score = self.transitions[pad_end_tags, pad_start_tags]

        invalid_sentence_masks = (1 - sentence_masks).byte()
        invalid_transition_masks = (1 - transition_masks).byte()

        tag_wise_score.masked_fill_(invalid_sentence_masks, 0)
        transition_score.masked_fill_(invalid_transition_masks, 0)

        score = torch.sum(tag_wise_score, 1) + torch.sum(transition_score, 1)

        return score

    def _get_lstm_features(self, words, sentence_masks, chars, chars_length, char_position_map, device):
        """
        calculate the LSTM features for each step
        :param words: int matrix, batch x max_seq
        :param sentence_masks: binary (0,1) int matrix, batch x max_seq
        :param chars: int matrix, all_words x max_word
        :param chars_length: int matrix, all_words
        :param chars_position_map: dict, all_words
        :param device: torch.cuda.device
        :return: LSTM features
        """

        batch_size, max_seq_length = words.size()
        sentence_lengths = torch.sum(sentence_masks, 1)

        # all_chars x max_word x char_embed
        chars_embeds = self.char_embeds(chars)

        if self.char_mode == CharEmbeddingSchema.LSTM:
            # packed char embeddings
            packed = torch.nn.utils.rnn.pack_padded_sequence(chars_embeds, chars_length, batch_first=True)
            chars_lstm_out_packed, _ = self.char_lstm(packed)
            chars_lstm_out, chars_lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(chars_lstm_out_packed,
                                                                                            batch_first=True)

            # restore
            chars_embeds_temp = torch.zeros((batch_size, max_seq_length, chars_lstm_out.size(2)), dtype=torch.float)
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.to(device)
            for i, index in enumerate(chars_lstm_out_lengths):
                last_char_lstm_out = torch.cat(
                    (chars_lstm_out[i, index - 1, :self.char_lstm_dim], chars_lstm_out[i, 0, self.char_lstm_dim:])) \
                    if self.char_lstm_bidirect else chars_lstm_out[i, index - 1, :self.char_lstm_dim]
                chars_embeds_temp[char_position_map[i][0], char_position_map[i][1]] = last_char_lstm_out

            chars_embeds = chars_embeds_temp.clone()

        if self.char_mode == CharEmbeddingSchema.CNN:
            # all_word x 1 x max_word x char_embed
            chars_embeds = chars_embeds.unsqueeze(1)
            # all_word x cnn_output x max_word x 1
            chars_cnn_out = self.char_cnn(chars_embeds)
            chars_cnn_out = chars_cnn_out[:, :, :chars_embeds.size(2), :]
            # all_word x cnn_output x 1 x 1 -> all_word x cnn_output
            max_pool_out = F.max_pool2d(chars_cnn_out, kernel_size=(chars_cnn_out.size(2), 1))
            chars_embeds = max_pool_out.squeeze(2).squeeze(2)

            # restore
            chars_embeds_temp = torch.zeros((batch_size, max_seq_length, chars_embeds.size(1)), dtype=torch.float)
            if self.use_gpu:
                chars_embeds_temp = chars_embeds_temp.to(device)
            for i in range(chars_embeds.size(0)):
                chars_embeds_temp[char_position_map[i][0], char_position_map[i][1]] = chars_embeds[i]

            chars_embeds = chars_embeds_temp.clone()

        # batch x max_seq x word_embed
        embeds = self.word_embeds(words)
        # batch x max_seq x [word_embed + char_embed]
        embeds = torch.cat([embeds, chars_embeds], 2)

        embeds = self.dropout(embeds)

        # pack sentences for LSTM
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_lengths, batch_first=True)
        lstm_out_packed, _ = self.lstm(packed_embeds)
        lstm_out, lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)

        # batch x max_seq x hidden_state
        lstm_out = self.dropout(lstm_out)
        # batch x max_seq x tag
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats



    def _forward_alg(self, feats, sentence_masks, device):
        """
        Get alpha values for CRF
        :param feats: LSTM output, batch x max_seq x tag
        :param sentence_masks: binary (0,1) int matrix, batch x max_seq
        :param device: device info
        :return: alpha values for each sentence
        """

        batch_size, max_seq_length, tag_num = feats.size()
        sentence_lengths = torch.sum(sentence_masks, 1)

        # initialize alpha with a Tensor with values all equal to Constants.Invalid_Transition, 1 x tag
        init_alphas = torch.Tensor(1, self.tag_set_size).fill_(Constants.Invalid_Transition)
        init_alphas[0][self.tag_to_id[Constants.Tag_Start]] = 0.

        # batch x 1 x tag
        forward_var = init_alphas.view(1, 1, tag_num).expand(batch_size, 1, tag_num)

        all_alphas = torch.zeros((max_seq_length, batch_size, tag_num), dtype=torch.float)
        if self.use_gpu:
            forward_var = forward_var.to(device)
            all_alphas = all_alphas.to(device)

        for i in range(max_seq_length):
            # batch x tag
            feat = feats[:, i, :]
            # batch x tag x 1
            emit_score = feat.view(batch_size, tag_num, 1)
            # batch x tag x tag
            transition_expanded = self.transitions.view(1, tag_num, tag_num).expand(batch_size, tag_num, tag_num)
            # batch x tag x tag
            tag_var = forward_var + transition_expanded + emit_score
            # batch x tag --> batch x 1 x tag
            new_forward_var = log_sum_exp(tag_var, dim=2)
            forward_var = new_forward_var.unsqueeze(1)
            all_alphas[i] = new_forward_var

        # max_seq x batch x tag
        forward_var_selection = (sentence_lengths - 1).view(1, -1, 1).expand(1, -1, tag_num)
        # batch x tag
        forward_var_last = torch.gather(all_alphas, 0, forward_var_selection).squeeze(0)

        terminal_var = forward_var_last + self.transitions[self.tag_to_id[Constants.Tag_End], :].view(1, -1)
        # batch
        Z = log_sum_exp(terminal_var, dim=1)

        return Z


    def viterbi_decode(self, feats, sentence_masks, device):
        """
        Viterbi decoding
        :param feats: LSTM output, batch x max_seq x tag
        :param sentence_masks: binary (0,1) int matrix, batch x max_seq
        :param device: device info
        :return: tagging result
        """

        sentence_lengths = torch.sum(sentence_masks, 1)
        batch_size, max_seq_length, tag_num = feats.size()

        scores = []
        tag_seqs = []

        for si in range(batch_size):
            # sentence_length x tag
            feat = feats[si, :sentence_lengths[si], :]

            back_pointers = []

            # analogous to forward, 1 x tag
            forward_var = torch.Tensor(1, self.tag_set_size).fill_(Constants.Invalid_Transition)
            forward_var[0][self.tag_to_id[Constants.Tag_Start]] = 0

            if self.use_gpu:
                forward_var = forward_var.to(device)

            for tag_feat in feat:
                # tag x tag
                next_tag_var = forward_var + self.transitions
                # tag
                viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)
                # 1 x tag
                forward_var = viterbivars_t.view(1, -1) + tag_feat.view(1, -1)
                bptrs_t = bptrs_t.cpu().numpy().tolist()
                back_pointers.append(bptrs_t)

            # 1 x tag
            terminal_var = forward_var + self.transitions[self.tag_to_id[Constants.Tag_End], :].view(1, -1)
            terminal_var[0][self.tag_to_id[Constants.Tag_End]] = Constants.Invalid_Transition
            terminal_var[0][self.tag_to_id[Constants.Tag_Start]] = Constants.Invalid_Transition

            path_score, best_tag_id = torch.max(terminal_var, 1)
            path_score = path_score.cpu().data.item()
            best_tag_id = best_tag_id.cpu().data.item()
            best_path = [best_tag_id]
            for bptrs_t in reversed(back_pointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)

            start = best_path.pop()
            assert start == self.tag_to_id[Constants.Tag_Start]
            best_path.reverse()

            scores.append(path_score)
            tag_seqs.append(best_path)

        return scores, tag_seqs

    def neg_log_likelihood(self, words, sentence_masks, tags, chars, chars_length, chars_position_map, device):
        """
        Loss function
        :param words: int matrix, batch x max_seq
        :param sentence_masks: binary (0,1) int matrix, batch x max_seq
        :param tags: int matrix, batch x max_seq
        :param chars: int matrix, batch x max_seq x max_word
        :param chars_length: int matrix, batch x max_seq
        :param chars_position_map: dict list, batch x dict_size
        :param device: torch.cuda.device
        :return: loss of CRF/Softmax
        """

        sentence_lengths = torch.sum(sentence_masks, 1)

        # batch x max_seq x tag
        feats = self._get_lstm_features(words, sentence_masks, chars, chars_length, chars_position_map, device)

        if self.use_crf:
            forward_score = self._forward_alg(feats, sentence_masks, device)
            gold_score = self._score_sentence(feats, tags, sentence_masks, device)
            crf_loss = torch.sum(forward_score - gold_score) / sentence_lengths.size(0)
            return crf_loss
        else:
            all_scores = torch.zeros(sentence_lengths.size(0), dtype=torch.float)
            if self.use_gpu:
                all_scores = all_scores.to(device)

            for i, length in enumerate(sentence_lengths):
                sentence_feat = feats[i, :length, :]
                sentence_tags = tags[i, :length]
                all_scores[i] = F.cross_entropy(sentence_feat, sentence_tags, size_average=False)

            scores = torch.sum(all_scores)
            return scores

    def forward(self, words, sentence_masks, chars, chars_length, chars_position_map, device):
        """
        Derive the tagging results
        :param words: int matrix, batch x max_seq
        :param sentence_masks: binary (0,1) int matrix, batch x max_seq
        :param chars: int matrix, batch x max_seq x max_word
        :param chars_length: int matrix, batch x max_seq
        :param chars_position_map: dict list, batch x dict_size
        :param device: torch.cuda.device
        :return: tagging results
        """

        sentence_lengths = torch.sum(sentence_masks, 1)

        # batch x max_seq x tag
        feats = self._get_lstm_features(words, sentence_masks, chars, chars_length, chars_position_map, device)

        # sentence_lengths = torch.sum(sentence_masks, 1)
        if self.use_crf:
            score, tag_seq = self.viterbi_decode(feats, sentence_masks, device)
        else:
            score_temp, tag_seq_temp = torch.max(feats, 2)
            tag_seq = []
            score = []
            for i, length in enumerate(sentence_lengths):
                tag_seq.append(list(tag_seq_temp[i, :length].cpu().numpy()))
                score.append(list(score_temp[i, :length].cpu().numpy()))

        return score, tag_seq
