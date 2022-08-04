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
    table_tag = []

    text_set = set()
    i = 0
    tags = []
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

            tag += item['html']['structure']['tokens'][i].strip() + item['html']['structure']['tokens'][i + 1]
            i += 2
            tags.append(tag.strip())
        else:
            if tag == "<tr>":
                num_rows += 1
                num_cols = 0
            elif tag == "<td>":
                num_cols += 1
            tags.append(tag)

    i = 0
    for tag in tags:
        table_tag.append(tag)
        if tag.startswith("<td"):
            text = remove_html_tags("".join(item['html']['cells'][i]['tokens'])).strip()
            if text:
                table_tag.append(text)
                text_set.update(set(text))
            i += 1
    return table_tag, text_set, max_row_span, max_col_span, num_rows, num_cols


def main(args):
    total_chars = set()
    os.makedirs(args.output_dir, exist_ok=True)
    train_output = os.path.join(args.output_dir, "ofa_dataset_train.tsv")
    val_output = os.path.join(args.output_dir, "ofa_dataset_val.tsv")
    chars_output = os.path.join(args.output_dir, "chars.json")
    train_image_dir = os.path.join(args.image_root, "train")
    val_image_dir = os.path.join(args.image_root, "val")
    trainf = open(train_output, "w+", encoding='utf-8')
    valf = open(val_output, "w+", encoding='utf-8')
    max_row_span = 0
    max_col_span = 0
    max_square = 0

    for i, line in enumerate(open(args.label_path, encoding='utf-8')):
        if i % 10 == 0:
            print(i)
        item = json.loads(line)
        table_tag, text_set, tmp_max_row_span, tmp_max_col_span, num_rows, num_cols = convert_ptn_item_to_simple_html(
            item)
        if max_row_span < tmp_max_row_span:
            max_row_span = tmp_max_row_span
        if max_col_span < tmp_max_col_span:
            max_col_span = tmp_max_col_span
        if max_square < num_rows * num_cols:
            max_square = num_rows * num_cols
        total_chars.update(text_set)
        if item['split'] == "train":
            image_dir = train_image_dir
            outf = trainf
        else:
            image_dir = val_image_dir
            outf = valf
        image_path = os.path.join(image_dir, item['filename'])
        img = Image.open(image_path)  # path to file
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_buffer = BytesIO()
        img.save(img_buffer, format=img.format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)  # bytes
        base64_str = base64_str.decode("utf-8")  # str
        outf.write("{}\n".format("\t".join([str(i + 1), base64_str, "".join(table_tag)])))

    total_chars = list(total_chars)
    total_chars.sort()
    json.dump(total_chars, open(chars_output, "w+"))
    trainf.close()
    valf.close()
    print("max_row_span", max_row_span)
    print("max_col_span", max_col_span)
    print("max_square", max_square)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\PubTabNet_2.0.0.jsonl")
    parser.add_argument('--image_root', type=str,
                        default="D:\dataset\\table_ocr\pubtabnet\pubtabnet")
    parser.add_argument('--output_dir', type=str, default="D:\dataset\\table_ocr\pubtabnet\pubtabnet\ofa_dataset")

    main(parser.parse_args())
