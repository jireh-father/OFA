import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64
import json


def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# tokenizers
# https://wikidocs.net/166826
# https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer
# https://misconstructed.tistory.com/80

def convert_ptn_item_to_simple_html(item):
    table_tag = []

    text_set = set()
    total_texts = []
    i = 0
    tags = []
    max_col_span = 0
    max_row_span = 0
    while i < len(item['html']['structure']['tokens']):
        tag = item['html']['structure']['tokens'][i]
        tag = tag.strip()
        i += 1
        if tag in ["<thead>", "</thead>", "<tbody>", "</tbody>"] or tag.startswith("</t"):
            continue
        if tag == "<td":
            split_tag = item['html']['structure']['tokens'][i].strip().split('"')
            num_spans = int(split_tag[1])
            if "col" in split_tag[0]:
                if num_spans > max_col_span:
                    max_col_span = num_spans
            else:
                if num_spans > max_row_span:
                    max_row_span = num_spans

            tag += item['html']['structure']['tokens'][i].strip() + item['html']['structure']['tokens'][i + 1]
            i += 2
            tags.append(tag.strip())
        else:
            tags.append(tag)

    i = 0
    for tag in tags:
        table_tag.append(tag)
        if tag.startswith("<td"):
            text = remove_html_tags("".join(item['html']['cells'][i]['tokens'])).strip()
            if text:
                table_tag.append(text)
                text_set.update(set(text))
                total_texts.append(text)
            i += 1
    return table_tag, total_texts, text_set, max_row_span, max_col_span


def main(args):
    total_chars = set()
    total_texts = []
    os.makedirs(args.output_dir, exist_ok=True)
    chars_output = os.path.join(args.output_dir, "chars.json")
    corpus_output = os.path.join(args.output_dir, "total_corpus.txt")

    output_json_train = os.path.join(args.output_dir, "train_metadata.jsonl")
    output_json_val = os.path.join(args.output_dir, "val_metadata.jsonl")
    max_row_span = 0
    max_col_span = 0
    with open(output_json_train, "w+", encoding='utf-8') as output_train, open(output_json_val, "w+",
                                                                               encoding='utf-8') as output_val:
        for i, line in enumerate(open(args.label_path, encoding='utf-8')):
            if i % 10 == 0:
                print(i)
            if i >= args.test_cnt:
                break
            item = json.loads(line)
            table_tag, tmp_total_texts, text_set, tmp_max_row_span, tmp_max_col_span = convert_ptn_item_to_simple_html(
                item)
            total_texts.append(" ".join(tmp_total_texts))
            if max_row_span < tmp_max_row_span:
                max_row_span = tmp_max_row_span
            if max_col_span < tmp_max_col_span:
                max_col_span = tmp_max_col_span
            total_chars.update(text_set)
            if item['split'] == "train":
                outf = output_train
            else:
                outf = output_val
            gt_parse = {
                "gt_parse": {"text_sequence": " ".join(table_tag)}
            }
            result_line = {
                "file_name": item['filename'],
                "ground_truth": json.dumps(gt_parse)
            }
            outf.write("{}\n".format(json.dumps(result_line)))

    total_chars = list(total_chars)
    total_chars.sort()
    json.dump(total_chars, open(chars_output, "w+"))
    with open(corpus_output, "w+", encoding="utf-8") as corpus_outf:
        for text in total_texts:
            corpus_outf.write("{}\n".format(text))

    print("max_row_span", max_row_span)
    print("max_col_span", max_col_span)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\ofa_dataset")

    parser.add_argument('--test_cnt', type=int,
                        default=None)
    main(parser.parse_args())
