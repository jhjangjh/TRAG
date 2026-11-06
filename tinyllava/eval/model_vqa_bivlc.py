import argparse
import torch
import os
import json
from tqdm import tqdm
import re

from tinyllava.eval.choice_parsing import parse_choice
import glob

from tinyllava.utils import *
from tinyllava.data import *
from tinyllava.model import *

from tinyllava.model.configuration_compodistill import CompoDistillConfig
from tinyllava.model.modeling_compodistill import CompoDistillForConditionalGeneration

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import math
import io

# Try to import parquet loading libraries
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' library not available, trying pandas...")

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: 'pandas' library not available")

if not DATASETS_AVAILABLE and not PANDAS_AVAILABLE:
    raise ImportError("Either 'datasets' or 'pandas' library is required to load parquet files")


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    """Get the k-th chunk from a list split into n chunks"""
    chunks = split_list(lst, n)
    return chunks[k]


def load_bivlc_data(data_dir):
    """
    Load BiVLC dataset from parquet files in the specified directory.
    Returns a list of data entries with PIL images.
    """
    print(f"Loading BiVLC data from {data_dir}")

    # Find all parquet files in the directory
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet file(s): {[os.path.basename(f) for f in parquet_files]}")

    data = []

    # Try using datasets library first (preferred for HuggingFace datasets)
    if DATASETS_AVAILABLE:
        try:
            print("Loading data using 'datasets' library...")
            dataset = load_dataset('parquet', data_files=parquet_files, split='train')

            for idx, item in enumerate(dataset):
                data.append({
                    'id': idx,
                    'image': item['image'],  # PIL Image object
                    'caption': item['caption'],
                    'negative_caption': item['negative_caption'],
                    'type': item.get('type', ''),
                    'subtype': item.get('subtype', '')
                })

            print(f"Successfully loaded {len(data)} samples using 'datasets' library")
            return data

        except Exception as e:
            print(f"Failed to load with 'datasets' library: {e}")
            if not PANDAS_AVAILABLE:
                raise

    # Fall back to pandas
    if PANDAS_AVAILABLE:
        try:
            print("Loading data using 'pandas' library...")
            dfs = []
            for parquet_file in parquet_files:
                df = pd.read_parquet(parquet_file)
                dfs.append(df)

            # Concatenate all dataframes
            full_df = pd.concat(dfs, ignore_index=True)

            for idx, row in full_df.iterrows():
                data.append({
                    'id': idx,
                    'image': row['image'],  # PIL Image object
                    'caption': row['caption'],
                    'negative_caption': row['negative_caption'],
                    'type': row.get('type', ''),
                    'subtype': row.get('subtype', '')
                })

            print(f"Successfully loaded {len(data)} samples using 'pandas' library")
            return data

        except Exception as e:
            print(f"Failed to load with 'pandas' library: {e}")
            raise

    raise RuntimeError("Failed to load parquet files with any available method")


def prepare_questions(bivlc_data):
    """
    Convert BiVLC data format to question format.
    For each sample, we create two questions (forward and reverse order).
    """
    questions = []

    for item in bivlc_data:
        item_id = item['id']
        pil_image = item['image']
        positive_caption = item['caption']
        negative_caption = item['negative_caption']
        item_type = item.get('type', '')
        item_subtype = item.get('subtype', '')

        # Forward order: positive caption first (correct answer is 1)
        prompt_forward = (
            f"Which caption best describes the image?\n"
            f"(1) {positive_caption}\n"
            f"(2) {negative_caption}\n"
            f"Output (1) or (2)."
        )

        questions.append({
            'question_id': f"{item_id}_forward",
            'bivlc_id': item_id,
            'image_pil': pil_image,  # Store PIL image directly
            'text': prompt_forward,
            'order': 'forward',
            'ground_truth': 1,
            'positive_caption': positive_caption,
            'negative_caption': negative_caption,
            'type': item_type,
            'subtype': item_subtype
        })

        # Reverse order: negative caption first (correct answer is 2)
        prompt_reverse = (
            f"Which caption best describes the image?\n"
            f"(1) {negative_caption}\n"
            f"(2) {positive_caption}\n"
            f"Output (1) or (2)."
        )

        questions.append({
            'question_id': f"{item_id}_reverse",
            'bivlc_id': item_id,
            'image_pil': pil_image,  # Store PIL image directly
            'text': prompt_reverse,
            'order': 'reverse',
            'ground_truth': 2,
            'positive_caption': positive_caption,
            'negative_caption': negative_caption,
            'type': item_type,
            'subtype': item_subtype
        })

    return questions


class BiVLCDataset(Dataset):
    """Custom dataset class for BiVLC evaluation"""

    def __init__(self, questions, text_processor, image_processor):
        self.questions = questions
        self.text_processor = text_processor
        self.image_processor = image_processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_data = line["image_pil"]
        qs = line["text"]

        # Convert image data to PIL Image
        if isinstance(image_data, Image.Image):
            # Already a PIL Image
            pil_image = image_data.convert('RGB')
        elif isinstance(image_data, dict):
            # Image stored as dict with 'bytes' and 'path' keys
            if 'bytes' in image_data and image_data['bytes']:
                # Load from bytes
                pil_image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
            elif 'path' in image_data and image_data['path']:
                # Fall back to path if bytes not available
                pil_image = Image.open(image_data['path']).convert('RGB')
            else:
                raise ValueError(f"Invalid image dict format: {image_data.keys()}")
        else:
            # Try to convert from array
            import numpy as np
            if isinstance(image_data, np.ndarray):
                pil_image = Image.fromarray(image_data).convert('RGB')
            else:
                raise ValueError(f"Unsupported image type: {type(image_data)}")

        image_tensor = self.image_processor(pil_image)

        # Add default image token
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        msg = Message()
        msg.add_message(qs)
        result = self.text_processor(msg.messages, mode='eval')
        input_ids = result['input_ids']

        return input_ids, image_tensor, pil_image.size

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids, image_tensors, image_sizes = zip(*batch)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors, image_sizes


def create_data_loader(questions, text_processor, image_processor, batch_size=1, num_workers=4):
    """Create DataLoader for BiVLC dataset"""
    assert batch_size == 1, "batch_size must be 1"
    dataset = BiVLCDataset(questions, text_processor, image_processor)
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

    # Calculate accuracy by type (optional analysis)
    type_stats = {}
    for pred in predictions:
        pred_type = pred.get('type', 'Unknown')
        if pred_type not in type_stats:
            type_stats[pred_type] = {'correct': 0, 'total': 0}

        type_stats[pred_type]['total'] += 1
        if pred['predicted'] == pred['ground_truth']:
            type_stats[pred_type]['correct'] += 1

    for pred_type, stats in type_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0

    results['by_type'] = type_stats

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

    # Load BiVLC data
    print(f"Loading BiVLC data from {args.data_dir}")
    try:
        bivlc_data = load_bivlc_data(args.data_dir)
    except Exception as e:
        print(f"Error loading BiVLC data: {e}")
        raise

    # Prepare questions
    print("Preparing questions...")
    questions = prepare_questions(bivlc_data)
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
    data_loader = create_data_loader(questions, text_processor, image_processor)

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
            option_1, option_2 = line['positive_caption'], line['negative_caption']
        else:
            option_1, option_2 = line['negative_caption'], line['positive_caption']
        predicted_choice = parse_choice(output_text, option_1, option_2)

        # Record prediction
        prediction = {
            'question_id': line['question_id'],
            'bivlc_id': line['bivlc_id'],
            'order': line['order'],
            'ground_truth': line['ground_truth'],
            'predicted': predicted_choice,
            'model_output': output_text,
            'correct': predicted_choice == line['ground_truth'] if predicted_choice is not None else False,
            'prompt': line['text'],
            'type': line['type'],
            'subtype': line['subtype']
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
        print("BiVLC Evaluation Results")
        print("="*60)
        print(f"Forward Order:  {accuracy_results['forward']['correct']}/{accuracy_results['forward']['total']} = {accuracy_results['forward']['accuracy']:.4f}")
        print(f"Reverse Order:  {accuracy_results['reverse']['correct']}/{accuracy_results['reverse']['total']} = {accuracy_results['reverse']['accuracy']:.4f}")
        print(f"\nAverage Accuracy: {accuracy_results['average']['accuracy']:.4f}")

        # Print accuracy by type
        if 'by_type' in accuracy_results and accuracy_results['by_type']:
            print("\nAccuracy by Type:")
            for pred_type, stats in sorted(accuracy_results['by_type'].items()):
                print(f"  {pred_type}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.4f}")

        print("="*60)

    print(f"\nPredictions saved to: {predictions_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TinyLLaVA on BiVLC dataset")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the TinyLLaVA model")
    parser.add_argument("--model-base", type=str, default=None, help="Base model name")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing BiVLC parquet files")
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
