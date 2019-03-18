# coding:utf-8
import codecs
import os
import math
import numpy as np
from constants import Constants
from data_format_util import iobes_to_iob2
from data_processing import generate_mini_batch_input
from datetime import datetime


def get_perf_metric(name, best_results):
    """
    Evalute the result file and get the new f1 value
    :param name: name of the model
    :param best_results: the current best results: (f1, accuracy, recall, precision)
    :return: new best f1 value, the new f1 value, whether the new best f1 value is updated
    """
    should_save = False
    new_f1 = 0.0

    eval_path = Constants.Eval_Folder
    eval_tmp_folder = Constants.Eval_Temp_Folder
    eval_script = Constants.Eval_Script

    prediction_file = eval_tmp_folder + '/predition.' + name
    score_file = eval_tmp_folder + '/score.' + name

    os.system('perl %s <%s >%s' % (eval_script, prediction_file, score_file))

    evaluation_lines = [line.rstrip() for line in codecs.open(score_file, 'r', 'utf8')]

    for i, line in enumerate(evaluation_lines):  # accuracy:  96.46%; precision:  74.61%; recall:  47.99%; FB1:  58.41
        if i == 1:
            tmp = line.strip().split()
            new_f1 = float(tmp[-1])
            new_a = float(tmp[1][:tmp[1].find("%")])
            new_p = float(tmp[3][:tmp[3].find("%")])
            new_r = float(tmp[5][:tmp[5].find("%")])
            new_results = [new_f1, new_a, new_p, new_r]
            best_f1 = best_results[0]
            if new_f1 > best_f1:
                best_results = [new_f1, new_a, new_p, new_r]
                should_save = True

    return best_results, new_results, should_save


def get_NAM_NOM(golden_label, pred_label):

    num_golden_NAM = 0
    num_pred_NAM = 0
    TP_NAM = 0

    num_golden_NOM = 0
    num_pred_NOM = 0
    TP_NOM = 0

    for g_label, p_label in zip(golden_label, pred_label):
        if g_label.endswith("NAM"):
            num_golden_NAM += 1
        elif g_label.endswith("NOM"):
            num_golden_NOM += 1

        if p_label.endswith("NAM"):
            num_pred_NAM += 1
        elif p_label.endswith("NOM"):
            num_pred_NOM += 1

        if g_label == p_label and g_label != "O":
            if g_label.endswith("NAM"):
                TP_NAM += 1
            elif g_label.endswith("NOM"):
                TP_NOM += 1

    P_NAM = TP_NAM / num_pred_NAM if num_pred_NAM != 0 else np.nan
    R_NAM = TP_NAM / num_golden_NAM

    P_NOM = TP_NOM / num_pred_NOM if num_pred_NOM != 0 else np.nan
    R_NOM = TP_NOM / num_golden_NOM

    F1_NAM = 2 * P_NAM * R_NAM / (P_NAM + R_NAM) if (P_NAM + R_NAM) != 0 else np.nan
    F1_NOM = 2 * P_NOM * R_NOM / (P_NOM + R_NOM) if (P_NOM + R_NOM) != 0 else np.nan

    return P_NAM, R_NAM, F1_NAM, P_NOM, R_NOM, F1_NOM



def evaluating(model, datas, best_results, name, mappings, char_mode, use_gpu, device, mini_batch_size):
    prediction = []

    eval_tmp_folder = Constants.Eval_Temp_Folder
    id_to_tag = mappings["id_to_tag"]
    char_to_id = mappings["char_to_id"]

    train_indecies = list(range(len(datas)))
    batch_count = math.ceil(len(datas) / mini_batch_size)

    total_time = 0
    for batch_i in range(batch_count):  # batch_count
        start_idx = batch_i * mini_batch_size
        end_idx = min((batch_i + 1) * mini_batch_size, len(datas))

        mini_batch_idx = train_indecies[start_idx:end_idx]
        sentence_masks, words, chars, tags, \
        sentence_char_lengths, sentence_char_position_map, str_words, unaligned_tags = \
            generate_mini_batch_input(datas, mini_batch_idx, mappings, char_mode)

        if use_gpu:
            sentence_masks = sentence_masks.to(device)
            words = words.to(device)
            chars = chars.to(device)
            tags = tags.to(device)
            sentence_char_lengths = sentence_char_lengths.to(device)

        start_time = datetime.now()
        val, out = model(words, sentence_masks, chars, sentence_char_lengths, sentence_char_position_map, device)
        end_time = datetime.now()
        total_time += (end_time - start_time).total_seconds()

        predicted_tags = [[id_to_tag[id] for id in predicted_id] for predicted_id in out]
        predicted_tags_bio = [iobes_to_iob2(predicted_tags_sentence) for predicted_tags_sentence in predicted_tags]

        ground_truth_tags = [[id_to_tag[id] for id in ground_truth_id] for ground_truth_id in unaligned_tags]
        ground_truth_tags_bio = [iobes_to_iob2(ground_truth_tags_sentence) for ground_truth_tags_sentence in ground_truth_tags]

        for si in range(end_idx - start_idx):
            for (str_word, true_tag, pred_tag) in zip(str_words[si], ground_truth_tags_bio[si], predicted_tags_bio[si]):
                line = ' '.join([str_word, true_tag, pred_tag])
                prediction.append(line)
            prediction.append('')

    predf = os.path.join(eval_tmp_folder, 'predition.' + name)

    with codecs.open(predf, mode='w', encoding="utf-8") as f:
        f.write('\n'.join(prediction))
        f.flush()

    best_results, new_results, save = get_perf_metric(name, best_results)

    return best_results, new_results, save, total_time