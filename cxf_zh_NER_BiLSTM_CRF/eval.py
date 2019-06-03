import os


def eval_ner_char(label_predict):
    cnt_right = 0
    cnt_total = 0
    org_tp = 0
    org_fp = 0
    org_fn = 0

    per_tp = 0
    per_fp = 0
    per_fn = 0

    loc_tp = 0
    loc_fp = 0
    loc_fn = 0
    for sent_result in label_predict:
        for char, tag, tag_ in sent_result:
            cnt_total += 1
            tag = '0' if tag == 'O' else tag

            tag = str(tag)
            tag_ = str(tag_)

            if tag == tag_:
                cnt_right += 1

            # ORG
            if 'ORG' in tag and tag_ == tag:
                org_tp += 1
            if 'ORG' in tag and 'ORG' not in tag_:
                org_fn += 1
            if 'ORG' in tag_ and 'ORG' not in tag:
                org_fp += 1

            # PER
            if 'PER' in tag and tag_ == tag:
                per_tp += 1
            if 'PER' in tag and 'PER' not in tag_:
                per_fn += 1
            if 'PER' in tag_ and 'PER' not in tag:
                per_fp += 1

            # LOC
            if 'LOC' in tag and tag_ == tag:
                loc_tp += 1
            if 'LOC' in tag and 'LOC' not in tag_:
                loc_fn += 1
            if 'LOC' in tag_ and 'LOC' not in tag:
                loc_fp += 1

    # org
    org_p = org_tp / (org_tp + org_fp)
    org_recall = org_tp / (org_tp + org_fn)
    org_f1 = 2 * (org_p * org_recall) / (org_p + org_recall)

    # per
    per_p = per_tp / (per_tp + per_fp)
    per_recall = per_tp / (per_tp + per_fn)
    per_f1 = 2 * (per_p * per_recall) / (per_p + per_recall)

    # loc
    loc_p = loc_tp / (loc_tp + loc_fp)
    loc_recall = loc_tp / (loc_tp + loc_fn)
    loc_f1 = 2 * (loc_p * loc_recall) / (loc_p + loc_recall)

    print(org_p, org_recall, org_f1)
    print(per_p, per_recall, per_f1)
    print(loc_p, loc_recall, loc_f1)

    accuracy = cnt_right / cnt_total
    print('accuracy: ' + str(accuracy))

    print('eval end!')


# unfinished function
def eval_ner_word(label_predict):
    cnt_right = 0
    cnt_total = 0
    org_tp = 0
    org_fp = 0
    org_fn = 0

    per_tp = 0
    per_fp = 0
    per_fn = 0

    loc_tp = 0
    loc_fp = 0
    loc_fn = 0
    for sent_result in label_predict:
        org_index_start = 0
        org_index_end = 0
        org_tag = []
        for i in range(len(sent_result)):
            char = sent_result[i][0]
            tag = str(sent_result[i][1])
            tag_ = str(sent_result[i][2])

        for char, tag, tag_ in sent_result:
            cnt_total += 1
            tag = '0' if tag == 'O' else tag

            tag = str(tag)
            tag_ = str(tag_)

            if tag == tag_:
                cnt_right += 1

            # ORG
            if 'ORG' in tag and tag_ == tag:
                org_tp += 1
            if 'ORG' in tag and 'ORG' not in tag_:
                org_fn += 1
            if 'ORG' in tag_ and 'ORG' not in tag:
                org_fp += 1

            # PER
            if 'PER' in tag and tag_ == tag:
                per_tp += 1
            if 'PER' in tag and 'PER' not in tag_:
                per_fn += 1
            if 'PER' in tag_ and 'PER' not in tag:
                per_fp += 1

            # LOC
            if 'LOC' in tag and tag_ == tag:
                loc_tp += 1
            if 'LOC' in tag and 'LOC' not in tag_:
                loc_fn += 1
            if 'LOC' in tag_ and 'LOC' not in tag:
                loc_fp += 1

    # org
    org_p = org_tp / (org_tp + org_fp)
    org_recall = org_tp / (org_tp + org_fn)
    org_f1 = 2 * (org_p * org_recall) / (org_p + org_recall)

    # per
    per_p = per_tp / (per_tp + per_fp)
    per_recall = per_tp / (per_tp + per_fn)
    per_f1 = 2 * (per_p * per_recall) / (per_p + per_recall)

    # loc
    loc_p = loc_tp / (loc_tp + loc_fp)
    loc_recall = loc_tp / (loc_tp + loc_fn)
    loc_f1 = 2 * (loc_p * loc_recall) / (loc_p + loc_recall)

    print(org_p, org_recall, org_f1)
    print(per_p, per_recall, per_f1)
    print(loc_p, loc_recall, loc_f1)

    accuracy = cnt_right / cnt_total
    print('accuracy: ' + str(accuracy))

    print('eval end!')


def conlleval(label_predict, label_path, metric_path):
    """

    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """

    # eval_ner_char(label_predict)
    # eval_ner_word(label_predict)
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                # char = char.encode("utf-8")
                line.append("{} {} {}\n".format(char, tag, tag_))
            line.append("\n")
        fw.writelines(line)
    # Calculate the matrix using the conlleval_rev.pl file
    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics
