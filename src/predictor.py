import argparse
import os
import pandas as pd
import json
import collections
import numpy as np
import collections
import subprocess
import sys
import pdb

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("source_code/download_files/setuptools-45.2.0-py3-none-any.whl")
install("source_code/download_files/PyYAML-6.0.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
install("source_code/download_files/filelock-3.4.1-py3-none-any.whl")

install("source_code/download_files/huggingface_hub-0.4.0-py3-none-any.whl")
install("source_code/download_files/sacremoses-0.0.53.tar.gz")
install("source_code/download_files/tokenizers-0.12.1-cp36-cp36m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl")

install("source_code/download_files/charset_normalizer-3.0.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
install("source_code/download_files/attrs-22.2.0-py3-none-any.whl")

install("source_code/download_files/asynctest-0.13.0-py3-none-any.whl")
install("source_code/download_files/async_timeout-4.0.2-py3-none-any.whl")
install("source_code/download_files/frozenlist-1.2.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl")
install("source_code/download_files/multidict-5.2.0-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl")
install("source_code/download_files/yarl-1.7.2-cp36-cp36m-manylinux_2_5_x86_64.manylinux1_x86_64.manylinux_2_12_x86_64.manylinux2010_x86_64.whl")
install("source_code/download_files/idna-ssl-1.1.0.tar.gz")

install("source_code/download_files/aiosignal-1.2.0-py3-none-any.whl")
install("source_code/download_files/aiohttp-3.8.5-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
install("source_code/download_files/dill-0.3.4-py2.py3-none-any.whl")
install("source_code/download_files/fsspec-2022.1.0-py3-none-any.whl")

install("source_code/download_files/multiprocess-0.70.12.2-py36-none-any.whl")
install("source_code/download_files/pyarrow-6.0.1-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
install("source_code/download_files/urllib3-1.26.16-py2.py3-none-any.whl")
install("source_code/download_files/responses-0.17.0-py2.py3-none-any.whl")
install("source_code/download_files/xxhash-3.2.0-cp36-cp36m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")

install("source_code/download_files/transformers-4.18.0-py3-none-any.whl")
install("source_code/download_files/datasets-2.4.0-py3-none-any.whl")
install("source_code/download_files/adapter_transformers-3.0.1-py3-none-any.whl")

from tqdm.auto import tqdm
from transformers import AutoTokenizer, RobertaTokenizerFast
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, AdapterTrainer, Trainer
from datasets import Dataset, load_metric
#from torch.utils.data import Dataset

max_length = 384 # The maximum length of a feature (question and context)
doc_stride = 128

load_model = "source_code/QAModel"
tokenizer = RobertaTokenizerFast.from_pretrained(load_model)
model = AutoModelForQuestionAnswering.from_pretrained(load_model)

pad_on_right = tokenizer.padding_side == "right"

task_name = "Squad"
load_adapter = "source_code/Adapter_model"
adapter_config = load_adapter + "/adapter_config.json"
model.load_adapter(
    load_adapter,
    config=adapter_config,
    load_as=task_name,
    with_head = True
)
model.train_adapter(task_name)
model.set_active_adapters = task_name

def load_squad_dataset(file_name):
	f = open(file_name, encoding="utf-8")
	data = json.load(f)

	# Iterating through the json list
	entry_list = list()
	id_list = list()

	for row in data['data']:
	  title = row['title']

	  for paragraph in row['paragraphs']:
	      context = paragraph['context']

	      for qa in paragraph['qas']:
	          entry = {}

	          qa_id = qa['id']
	          question = qa['question']
	          answers = qa['answers']

	          entry['id'] = qa_id
	          entry['title'] = title.strip()
	          entry['context'] = context.strip()
	          entry['question'] = question.strip()

	          answer_starts = [answer["answer_start"] for answer in answers]
	          answer_texts = [answer["text"].strip() for answer in answers]
	          entry['answers'] = {}
	          entry['answers']['answer_start'] = answer_starts
	          entry['answers']['text'] = answer_texts

	          entry_list.append(entry)

	reverse_entry_list = entry_list[::-1]

	# for entries with same id, keep only last one (corrected texts by the group Deep Learning Brasil)
	unique_ids_list = list()
	unique_entry_list = list()
	for entry in reverse_entry_list:
	  qa_id = entry['id']
	  if qa_id not in unique_ids_list:
	      unique_ids_list.append(qa_id)
	      unique_entry_list.append(entry)

	# Closing file
	f.close()
	new_dict = {}
	new_dict['data'] = unique_entry_list
	return new_dict

def prepare_validation_features(examples):
	# Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
	# in one example possible giving several features when a context is long, each of those features having a
	# context that overlaps a bit the context of the previous feature.
	examples["question"] = [q.lstrip() for q in examples["question"]]
	tokenized_examples = tokenizer(
	    examples["question" if pad_on_right else "context"],
	    examples["context" if pad_on_right else "question"],
	    truncation="only_second" if pad_on_right else "only_first",
	    max_length=max_length,
	    stride=doc_stride,
	    return_overflowing_tokens=True,
	    return_offsets_mapping=True,
	    padding="max_length",
	)

	# Since one example might give us several features if it has a long context, we need a map from a feature to
	# its corresponding example. This key gives us just that.
	sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

	# We keep the example_id that gave us this feature and we will store the offset mappings.
	tokenized_examples["example_id"] = []

	for i in range(len(tokenized_examples["input_ids"])):
	    # Grab the sequence corresponding to that example (to know what is the context and what is the question).
	    sequence_ids = tokenized_examples.sequence_ids(i)
	    context_index = 1 if pad_on_right else 0

	    # One example can give several spans, this is the index of the example containing this span of text.
	    sample_index = sample_mapping[i]
	    tokenized_examples["example_id"].append(examples["id"][sample_index])

	    # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
	    # position is part of the context or not.
	    tokenized_examples["offset_mapping"][i] = [
	        (o if sequence_ids[k] == context_index else None)
	        for k, o in enumerate(tokenized_examples["offset_mapping"][i])
	    ]

	return tokenized_examples

def postprocess_qa_predictions(examples, features, raw_predictions, n_best_size = 20, max_answer_length = 30):
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []

        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer (only for squad_v2)
        #if not squad_v2:
        #    predictions[example["id"]] = best_answer["text"]
        #else:
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions[example["id"]] = answer

    return predictions

def parser_args():
	parser = argparse.ArgumentParser(description ='Process Squad data')
	parser.add_argument("input_file_path", type=str, help="Input File Path")
	parser.add_argument("output_file_path", type=str, help="Output File Path")
	args = parser.parse_args()
	return args

def save_predictions(output_pred_file, final_predictions):
	with open(output_pred_file, "w") as my_file:
		json.dump(final_predictions, my_file)

def main():
	args = parser_args()
	input_file_path = args.input_file_path
	output_pred_file = args.output_file_path
	if os.path.isfile(input_file_path):
		valid_data_dict = load_squad_dataset(input_file_path)
	#pdb.set_trace()
	valid_dataset = Dataset.from_pandas(pd.DataFrame(data=valid_data_dict["data"]))
	#valid_dataset = Dataset(pd.DataFrame(data=valid_data_dict["data"]))
	validation_features = valid_dataset.map(
	    prepare_validation_features,
	    batched=True,
	    remove_columns=valid_dataset.column_names
	)

	path_to_outputs = "./outputs"
	val_args = TrainingArguments(do_predict=True, fp16=True, output_dir=path_to_outputs)
	#val_args = TrainingArguments(do_predict=True, fp16=False, output_dir=path_to_outputs)
	data_collator = default_data_collator
	trainer = AdapterTrainer(
	    model=model,
	    args=val_args,
	    tokenizer=tokenizer,
	    data_collator=data_collator,
	    )

	raw_predictions = trainer.predict(validation_features)
	final_predictions = postprocess_qa_predictions(valid_dataset, validation_features, raw_predictions.predictions)
	#formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in final_predictions.items()]
	#references = [{"id": ex["id"], "answers": ex["answers"]} for ex in valid_dataset]
	#metric = load_metric("squad_v2")
	#print(metric.compute(predictions=formatted_predictions, references=references))
	save_predictions(output_pred_file, final_predictions)

if __name__ == "__main__":
	main()



