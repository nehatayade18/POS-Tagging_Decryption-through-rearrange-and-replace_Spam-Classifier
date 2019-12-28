import os
import os.path as osp
import sys
from collections import Counter
import math
import pickle

"""
Authors: Andrei Amatuni, Viral Prajapati, Neha Tayade
"""

import email
from email.policy import default

from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):

    text = ""

    def handle_data(self, data):
        self.text += data


STOP_WORDS = [
    "the",
    "and",
    "in",
    "of",
    "if",
    "or",
    "than",
    "there",
    "this"
]


def increment_counts(counts):
    for k, v in counts.items():
        counts[k] += 1

    return counts


def build_likelihood_table(spam_counts, not_spam_counts):
    table = {}

    num_spam = sum(v for k, v in spam_counts.items())
    num_not_spam = sum(v for k, v in not_spam_counts.items())

    all_words = set(list(spam_counts.keys()) + list(not_spam_counts.keys()))

    for word in all_words:
        s_count = spam_counts[word]
        ns_count = not_spam_counts[word]
        if s_count == 0:
            s_count += 1
        if ns_count == 0:
            ns_count += 1
        table[word] = {
            "spam": math.log(float(s_count)/num_spam, 2),
            "notspam": math.log(float(ns_count)/num_not_spam, 2)
            }

    return table


def likelihood(msg, table, classif="spam"):
    """
    Compute log likelihood of the words given a spam/notspam classification

    :param msg: list of words in the message
    :param table: table containing word likelihood values across the spam/notspam classifications
    :param classif: classification to pull likelihood for
    :return: sum of log likelihoods across words in the message
    """
    likelihood = 0

    for word in msg:
        if word not in table:
            continue
        likelihood += table[word][classif]

    return likelihood


def parse_words(file):
    text = ""
    with open(file, "rb") as input:
        msg = email.message_from_binary_file(input, policy=default)

        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload()
            if part.get_content_type() == "text/html":
                parser = MyHTMLParser()
                parser.feed(part.get_payload())
                text += parser.text

        words = [x.lower() for x in text.split()]
        words = [x for x in words if x not in STOP_WORDS] # filter stop words
        words = [x for x in words if not any(y.isnumeric() for y in x)] # filter tokens with numbers in them

    return words


def get_count_distribution(dir):
    counts = Counter()

    num_files = 0

    for root, dirs, files in os.walk(dir):
        for file in files:
            if not file.startswith("."):
                # print(file)
                words = parse_words(osp.join(root, file))
                counts.update(words)
                num_files += 1

    return counts, num_files


def train(spam_dir, not_spam_dir):

    print("Training...")
    spam_counts, num_spam = get_count_distribution(spam_dir)
    not_spam_counts, num_notspam = get_count_distribution(not_spam_dir)

    return spam_counts, not_spam_counts, num_spam, num_notspam


def test(test_dir, likelihood_table, num_spam, num_notspam, thresh=1.0):

    print("Testing...")

    P_s = num_spam / (num_spam + num_notspam)
    P_ns = num_notspam / (num_spam + num_notspam)

    results = []

    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if not file.startswith("."):
                # print(file)
                words = set(parse_words(osp.join(root, file)))
                spam_like = likelihood(words, likelihood_table, classif="spam")
                notspam_like = likelihood(words, likelihood_table, classif="notspam")

                spam_posterior = spam_like + math.log(P_s, 2)
                notspam_posterior = notspam_like + math.log(P_ns, 2)

                odds = spam_posterior / notspam_posterior

                if odds <= thresh:
                    results.append((file, "spam"))
                else:
                    results.append((file, "notspam"))

    return results


def compare_groundtruth(gt_file, test_results):
    gt = {}
    with open(gt_file, "r") as input:
        for line in input:
            l = line.split()
            gt[l[0]] = l[1]

    hits = 0
    for r in test_results:
        gt_val = gt[r[0]]
        if gt_val == r[1]:
            hits += 1

    return float(hits) / len(test_results)


def output(results, path):

    print("Writing output: {}".format(path))
    with open(path, "w") as out:
        for r in results:
            out.write("{} {}\n".format(r[0], r[1]))


if __name__ == "__main__":

    train_dir = sys.argv[1]
    test_dir = sys.argv[2]
    out_file = sys.argv[3]

    spam_counts, not_spam_counts, \
    num_spam, num_notspam = train(osp.join(train_dir, "spam"),
                                  osp.join(train_dir, "notspam"))

    spam_counts = increment_counts(spam_counts)
    not_spam_counts = increment_counts(not_spam_counts)

    likelihood_table = build_likelihood_table(spam_counts, not_spam_counts)

    #
    # with open("table.pkl", "wb") as out:
    #     pickle.dump(likelihood_table, out)

    # with open("table.pkl", "rb") as input:
    #     likelihood_table = pickle.load(input)

    # num_spam = 1214
    # num_notspam = 1432

    results = test(test_dir, likelihood_table, num_spam, num_notspam)


    model_perf = compare_groundtruth("test-groundtruth.txt", results)

    print("\nModel performance: {}\n".format(model_perf))

    output(results, out_file)

