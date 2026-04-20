import json
import argparse

def compute_confidence_from_records(records):
    """按照原始逻辑计算置信度：过滤到token_id=126081为止，取平均confidence"""
    filtered_records = []
    for record in sorted(records, key=lambda x: x.get('position', 0)):
        if record.get('token_id') == 126081:
            break
        filtered_records.append(record)
    
    token_confidences = [r.get('confidence', 0.0) for r in filtered_records]
    if token_confidences:
        return sum(token_confidences) / len(token_confidences)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)
    args = parser.parse_args()

    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    for sample_idx, sample in enumerate(dataset):
        if 'rollouts_records' not in sample:
            continue

        rollouts_records = sample['rollouts_records']
        rollouts_confidence = []

        for rollout_idx, records in enumerate(rollouts_records):
            avg_confidence = compute_confidence_from_records(records)
            rollouts_confidence.append(avg_confidence)
            print(f"Sample {sample_idx}, rollout {rollout_idx + 1}: avg_confidence = {avg_confidence:.4f}")

        sample['rollouts_confidence'] = rollouts_confidence
        del sample['rollouts_records']

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved {len(dataset)} samples to {args.output_file}")


if __name__ == "__main__":
    main()