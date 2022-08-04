import argparse
import json
import os
from PIL import Image
from io import BytesIO
import base64


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
    i = 0
    max_col_span = 0
    max_row_span = 0
    num_rows = 0
    num_cols = 0
    while i < len(item['html']['structure']['tokens']):
        tag = item['html']['structure']['tokens'][i]
        tag = tag.strip()
        i += 1
        if tag in ["<thead>", "</thead>", "<tbody>", "</tbody>"]:
            continue
        if tag == "<td":
            split_tag = item['html']['structure']['tokens'][i].strip().split('"')
            num_spans = int(split_tag[1])
            num_cols += num_spans
            if "col" in split_tag[0]:
                if num_spans > max_col_span:
                    max_col_span = num_spans
            else:
                if num_spans > max_row_span:
                    max_row_span = num_spans
            i += 2
        else:
            if tag == "<tr>":
                num_rows += 1
                num_cols = 0
            elif tag == "<td>":
                num_cols += 1

    return max_row_span, max_col_span, num_rows, num_cols


def main(args):
    max_row_span = 0
    max_col_span = 0
    max_square = 0
    max_square_data = None
    max_row_span_data = None
    max_col_span_data = None

    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        item = json.loads(line)
        tmp_max_row_span, tmp_max_col_span, num_rows, num_cols = convert_ptn_item_to_simple_html(item)
        if max_row_span < tmp_max_row_span:
            max_row_span = tmp_max_row_span
            max_row_span_data = line
        if max_col_span < tmp_max_col_span:
            max_col_span = tmp_max_col_span
            max_col_span_data = line
        if max_square < num_rows * num_cols:
            max_square = num_rows * num_cols
            max_square_data = line

    print("max_row_span", max_row_span)
    print("max_col_span", max_col_span)
    print("max_square", max_square)
    print("max_row_span_data", max_row_span_data)
    print("max_col_span_data", max_col_span_data)
    print("max_square_data", max_square_data)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")

    main(parser.parse_args())
