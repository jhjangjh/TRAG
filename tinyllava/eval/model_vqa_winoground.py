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


def load_winoground_data(data_file):
    """
    Load Winoground dataset from JSONL file.
    Returns a list of data entries.
    """
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))

    print(f"Loaded {len(data)} samples from Winoground dataset")
    return data


def prepare_questions(winoground_data, image_folder):
    """
    Convert Winoground data format to question format.
    For each sample, we create two questions (forward and reverse order).
    We only use image_0 as specified in the requirements.
    """
    questions = []

    for item in winoground_data:
        item_id = item['id']
        image_0 = item['image_0']
        caption_0 = item['caption_0']
        caption_1 = item['caption_1']

        # Add .png extension to image filename
        image_filename = f"{image_0}.png"

        # Forward order: caption_0 first (correct answer is 1)
        prompt_forward = (
            f"Which caption best describes the image?\n"
            f"(1) {caption_0}\n"
            f"(2) {caption_1}\n"
            f"Output (1) or (2)."
        )

        questions.append({
            'question_id': f"{item_id}_forward",
            'image': image_filename,
            'text': prompt_forward,
            'winoground_id': item_id,
            'order': 'forward',
            'ground_truth': 1,
            'caption_0': caption_0,
            'caption_1': caption_1,
            'tag': item.get('tag', ''),
            'collapsed_tag': item.get('collapsed_tag', '')
        })

        # Reverse order: caption_1 first (correct answer is 2)
        prompt_reverse = (
            f"Which caption best describes the image?\n"
            f"(1) {caption_1}\n"
            f"(2) {caption_0}\n"
            f"Output (1) or (2)."
        )

        questions.append({
            'question_id': f"{item_id}_reverse",
            'image': image_filename,
            'text': prompt_reverse,
            'winoground_id': item_id,
            'order': 'reverse',
            'ground_truth': 2,
            'caption_0': caption_0,
            'caption_1': caption_1,
            'tag': item.get('tag', ''),
            'collapsed_tag': item.get('collapsed_tag', '')
        })

    return questions


class WinogroundDataset(Dataset):
    """Custom dataset class for Winoground evaluation"""

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
    """Create DataLoader for Winoground dataset"""
    assert batch_size == 1, "batch_size must be 1"
    dataset = WinogroundDataset(questions, image_folder, text_processor, image_processor)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def calculate_accuracy(predictions):
    """
    Calculate accuracy for forward, reverse, and average.
    Returns a dictionary with accuracy statistics.
    """
    results = {}

    # Separate forward and reverse predictions
    forward_preds = [p for p in predictions if p['order'] == 'forward']
    reverse_preds = [p for p in predictions if p['order'] == 'reverse']

    # Calculate forward accuracy
    forward_correct = sum(1 for p in forward_preds if p['predicted'] == p['ground_truth'])
    forward_total = len(forward_preds)
    forward_accuracy = forward_correct / forward_total if forward_total > 0 else 0

    results['forward'] = {
        'correct': forward_correct,
        'total': forward_total,
        'accuracy': forward_accuracy
    }

    # Calculate reverse accuracy
    reverse_correct = sum(1 for p in reverse_preds if p['predicted'] == p['ground_truth'])
    reverse_total = len(reverse_preds)
    reverse_accuracy = reverse_correct / reverse_total if reverse_total > 0 else 0

    results['reverse'] = {
        'correct': reverse_correct,
        'total': reverse_total,
        'accuracy': reverse_accuracy
    }

    # Calculate average accuracy
    average_accuracy = (forward_accuracy + reverse_accuracy) / 2

    results['average'] = {
        'accuracy': average_accuracy
    }

    # Calculate accuracy by tag (optional analysis)
    tag_stats = {}
    for pred in predictions:
        tag = pred.get('collapsed_tag', 'Unknown')
        if tag not in tag_stats:
            tag_stats[tag] = {'correct': 0, 'total': 0}

        tag_stats[tag]['total'] += 1
        if pred['predicted'] == pred['ground_truth']:
            tag_stats[tag]['correct'] += 1

    for tag, stats in tag_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    results['by_tag'] = tag_stats

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

    # Load Winoground data
    print(f"Loading Winoground data from {args.data_file}")
    winoground_data = load_winoground_data(args.data_file)

    # Prepare questions
    print("Preparing questions...")
    questions = prepare_questions(winoground_data, args.image_folder)
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
        if line['order'] == 'forward':
            option_1, option_2 = line['caption_0'], line['caption_1']
        else:
            option_1, option_2 = line['caption_1'], line['caption_0']
        predicted_choice = parse_choice(output_text, option_1, option_2)

        # Record prediction
        prediction = {
            'question_id': line['question_id'],
            'winoground_id': line['winoground_id'],
            'order': line['order'],
            'ground_truth': line['ground_truth'],
            'predicted': predicted_choice,
            'model_output': output_text,
            'correct': predicted_choice == line['ground_truth'] if predicted_choice is not None else False,
            'prompt': line['text'],
            'image': line['image'],
            'tag': line['tag'],
            'collapsed_tag': line['collapsed_tag']
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
        print("Winoground Evaluation Results")
        print("="*60)
        print(f"Forward Order:  {accuracy_results['forward']['correct']}/{accuracy_results['forward']['total']} = {accuracy_results['forward']['accuracy']:.4f}")
        print(f"Reverse Order:  {accuracy_results['reverse']['correct']}/{accuracy_results['reverse']['total']} = {accuracy_results['reverse']['accuracy']:.4f}")
        print(f"\nAverage Accuracy: {accuracy_results['average']['accuracy']:.4f}")

        # Print accuracy by tag
        if 'by_tag' in accuracy_results:
            print("\nAccuracy by Tag:")
            for tag, stats in sorted(accuracy_results['by_tag'].items()):
                print(f"  {tag}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

        print("="*60)

    print(f"\nPredictions saved to: {predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TinyLLaVA on Winoground dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the TinyLLaVA model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name")
    parser.add_argument("--data-file", type=str, required=True, help="Path to Winoground examples.jsonl file")
    parser.add_argument("--image-folder", type=str, required=True, help="Directory containing Winoground images")
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
