import argparse
import torch
import os
import json
from tqdm import tqdm
import re

from tinyllava.eval.choice_parsing import parse_choice

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from tinyllava.model.configuration_compodistill import CompoDistillConfig
from tinyllava.model.modeling_compodistill import CompoDistillForConditionalGeneration

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from a list split into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k]


def load_sugarcrepe_data(data_dir):
    """
    Load all SugarCrepe dataset files from the specified directory.
    Returns a dictionary mapping split names to their data.
    """
    split_files = [
        'add_att.json',
        'add_obj.json',
        'replace_att.json',
        'replace_obj.json',
        'replace_rel.json',
        'swap_att.json',
        'swap_obj.json'
    ]

    all_data = {}
    for split_file in split_files:
        split_name = split_file.replace('.json', '')
        file_path = os.path.join(data_dir, split_file)

        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                all_data[split_name] = data
                print(f"Loaded {len(data)} samples from {split_name}")
        else:
            print(f"Warning: {file_path} not found, skipping...")

    return all_data


def prepare_questions(all_data, image_folder):
    """
    Convert SugarCrepe data format to question format.
    For each sample, we create two questions (positive-first and negative-first).
    """
    questions = []

    for split_name, split_data in all_data.items():
        for idx, item in split_data.items():
            filename = item['filename']
            positive_caption = item['caption']
            negative_caption = item['negative_caption']

            # Positive-first variant
            prompt_pos_first = (
                f"Which caption best describes the image?\n"
                f"(1) {positive_caption}\n"
                f"(2) {negative_caption}\n"
                f"Output (1) or (2)."
            )

            questions.append({
                'question_id': f"{split_name}_{idx}_pos_first",
                'image': filename,
                'text': prompt_pos_first,
                'split': split_name,
                'variant': 'positive_first',
                'ground_truth': 1,
                'positive_caption': positive_caption,
                'negative_caption': negative_caption
            })

            # Negative-first variant
            prompt_neg_first = (
                f"Which caption best describes the image?\n"
                f"(1) {negative_caption}\n"
                f"(2) {positive_caption}\n"
                f"Output (1) or (2)."
            )

            questions.append({
                'question_id': f"{split_name}_{idx}_neg_first",
                'image': filename,
                'text': prompt_neg_first,
                'split': split_name,
                'variant': 'negative_first',
                'ground_truth': 2,
                'positive_caption': positive_caption,
                'negative_caption': negative_caption
            })

    return questions


class SugarCrepeDataset(Dataset):
    """Custom dataset class for SugarCrepe evaluation"""

    def __init__(self, questions, image_folder, text_processor, image_processor):
        self.questions = questions
        self.image_folder = image_folder
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]

        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_processor(image)

        # Add default image token
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, image_folder, text_processor, image_processor, batch_size=1, num_workers=4):
    """Create DataLoader for SugarCrepe dataset"""
    assert batch_size == 1, "batch_size must be 1"
    dataset = SugarCrepeDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def calculate_accuracy(predictions):
    """
    Calculate accuracy for each split and variant.
    Returns a dictionary with accuracy statistics.
    """
    results = {}

    # Group predictions by split and variant
    grouped = {}
    for pred in predictions:
        split = pred['split']
        variant = pred['variant']
        key = f"{split}_{variant}"

        if key not in grouped:
            grouped[key] = []
        grouped[key].append(pred)

    # Calculate accuracy for each group
    for key, preds in grouped.items():
        correct = sum(1 for p in preds if p['predicted'] == p['ground_truth'])
        total = len(preds)
        accuracy = correct / total if total > 0 else 0

        results[key] = {
            'correct': correct,
            'total': total,
            'accuracy': accuracy
        }

    # Calculate overall accuracy per split (average of both variants)
    split_names = set(pred['split'] for pred in predictions)
    split_averages = []
    for split_name in split_names:
        pos_key = f"{split_name}_positive_first"
        neg_key = f"{split_name}_negative_first"

        if pos_key in results and neg_key in results:
            avg_accuracy = (results[pos_key]['accuracy'] + results[neg_key]['accuracy']) / 2
            results[f"{split_name}_average"] = {
                'accuracy': avg_accuracy
            }
            split_averages.append(avg_accuracy)

    # Calculate overall average across all splits
    if split_averages:
        overall_average = sum(split_averages) / len(split_averages)
        results['overall_average'] = {
            'accuracy': overall_average,
            'num_splits': len(split_averages)
        }

    return results


def eval_model(args):
    """Main evaluation function"""

    # Load model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model, tokenizer, image_processor, context_len = load_pretrained_model(model_path)

    text_processor = TextPreprocess(tokenizer, args.conv_mode)
    data_args = model.config
    image_processor = ImagePreprocess(image_processor, data_args)

    # Load SugarCrepe data
    print(f"Loading SugarCrepe data from {args.data_dir}")
    all_data = load_sugarcrepe_data(args.data_dir)

    # Prepare questions
    print("Preparing questions...")
    questions = prepare_questions(all_data, args.image_folder)
    print(f"Total questions prepared: {len(questions)}")

    # Handle chunking for multi-GPU evaluation
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Processing chunk {args.chunk_idx}/{args.num_chunks} with {len(questions)} questions")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare output files
    predictions_file = os.path.join(args.output_dir, f"predictions_{args.chunk_idx}.jsonl")
    pred_file = open(predictions_file, "w")

    # Create data loader
    data_loader = create_data_loader(questions, args.image_folder, text_processor, image_processor)

    # Move model to GPU
    model.to(device='cuda')
    model.eval()

    # Evaluation loop
    predictions = []
    for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                image_sizes=image_sizes,
                use_cache=True
            )

        # Decode output
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # Parse prediction. Some models generate the selected caption text instead of "(1)" or "(2)".
        # Captions in the order they appear in the prompt.
        if line['variant'] == 'positive_first':
            option_1, option_2 = line['positive_caption'], line['negative_caption']
        else:
            option_1, option_2 = line['negative_caption'], line['positive_caption']
        predicted_choice = parse_choice(output_text, option_1, option_2)

        # Record prediction
        prediction = {
            'question_id': line['question_id'],
            'split': line['split'],
            'variant': line['variant'],
            'ground_truth': line['ground_truth'],
            'predicted': predicted_choice,
            'model_output': output_text,
            'correct': predicted_choice == line['ground_truth'] if predicted_choice is not None else False,
            'prompt': line['text'],
            'image': line['image']
        }

        predictions.append(prediction)
        pred_file.write(json.dumps(prediction) + "\n")

    pred_file.close()

    # Calculate and save accuracy (only for single chunk or last chunk)
    if args.num_chunks == 1 or args.chunk_idx == args.num_chunks - 1:
        print("\nCalculating accuracy...")
        accuracy_results = calculate_accuracy(predictions)

        # Save accuracy results
        accuracy_file = os.path.join(args.output_dir, f"accuracy_{args.chunk_idx}.json")
        with open(accuracy_file, 'w') as f:
            json.dump(accuracy_results, f, indent=2)

        # Print results
        print("\n" + "="*60)
        print("SugarCrepe Evaluation Results")
        print("="*60)

        # Print overall average first
        if 'overall_average' in accuracy_results:
            print(f"\n*** OVERALL AVERAGE: {accuracy_results['overall_average']['accuracy']:.4f} ***")
            print(f"    (Average across {accuracy_results['overall_average']['num_splits']} splits)")
            print()

        # Print individual split results
        for key, value in sorted(accuracy_results.items()):
            if key == 'overall_average':
                continue  # Already printed above
            elif 'average' in key:
                print(f"{key}: {value['accuracy']:.4f}")
            else:
                print(f"{key}: {value['correct']}/{value['total']} = {value['accuracy']:.4f}")
        print("="*60)

    print(f"\nPredictions saved to: {predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TinyLLaVA on SugarCrepe dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the TinyLLaVA model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing SugarCrepe JSON files")
    parser.add_argument("--image-folder", type=str, required=True, help="Directory containing COCO val2017 images")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--conv-mode", type=str, default="qwen2_base", help="Conversation mode")
    parser.add_argument("--num-chunks", type=int, default=1, help="Number of chunks for multi-GPU evaluation")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Current chunk index")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    args = parser.parse_args()

    eval_model(args)
