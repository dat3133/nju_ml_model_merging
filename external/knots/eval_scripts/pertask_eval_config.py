import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from task_merger import get_merge_handler
from utils import (
    evaluate_cliphead,
    get_clip_encodings,
    get_config_from_name,
    prepare_experiment_config,
    set_seed,
    write_to_csv,
)


FINE_TUNED_ACC_RANK16_VITB32 = {
    'stanford_cars': 74.0,
    'dtd': 58.3,
    'eurosat': 99.0,
    'gtsrb': 92.7,
    'mnist': 99.3,
    'resisc45': 88.4,
    'sun397': 64.5,
    'svhn': 96.2,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', default='vitB_r16_knots_ties_mnist')
    parser.add_argument('--eval-split', default='test', choices=['val', 'test'])
    parser.add_argument('--seed', type=int, default=420)
    parser.add_argument('--output-csv', default='')
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    raw_config = get_config_from_name(args.config_name, device=device)

    print('Running with config:', args.config_name)
    print('Eval split:', args.eval_split)
    print('Merge config:', raw_config['task_merge_config'])

    all_clip_encodings = [
        get_clip_encodings(dataset_config['clip_encodings'])
        for dataset_config in raw_config['dataset']
    ]
    config = prepare_experiment_config(raw_config)
    dataset_names = np.array([dataset_config['name'] for dataset_config in raw_config['dataset']])
    dataloaders = np.array([loader_dict for loader_dict in config['data']])

    with torch.no_grad():
        models = np.array([model.cpu() for model in config['models']['bases']])
        merge_cls = get_merge_handler(config['task_merge_config']['representation'])
        merge = merge_cls(
            deepcopy(models),
            pretrained_model=deepcopy(config['models']['new']),
            param_handler=config['param_handler'],
            device=device,
            merge_config=config['task_merge_config'],
        )
        merge.transform(config['task_merge_config'])
        merge.set_scaling_coeffs(config['task_merge_config']['scaling_coeffs'])
        merged_model = merge.merge(config['task_merge_config'])

        results = deepcopy(config['task_merge_config'])
        avg_accuracy = 0.0
        avg_norm_accuracy = 0.0
        evaluated = 0

        for i, loader_dict in enumerate(dataloaders):
            dataset_name = dataset_names[i]
            if args.eval_split not in loader_dict['test']:
                print(f'skipping {dataset_name}: split {args.eval_split} not available')
                continue

            loader = loader_dict['test'][args.eval_split]
            class_vectors = all_clip_encodings[i].to(device)
            acc = evaluate_cliphead(merged_model.to(device), loader, class_vectors=class_vectors)
            acc_pct = acc * 100
            ref_acc = FINE_TUNED_ACC_RANK16_VITB32.get(dataset_name)
            norm_acc = acc_pct / ref_acc * 100 if ref_acc else float('nan')

            print(f'{dataset_name} accuracy is {np.round(acc_pct, 3)}')
            print(f'{dataset_name} normalized accuracy is {np.round(norm_acc, 3)}')

            results[dataset_name] = acc_pct
            results[f'{dataset_name}_norm_acc'] = norm_acc
            avg_accuracy += acc_pct
            avg_norm_accuracy += norm_acc
            evaluated += 1

        avg_accuracy /= max(evaluated, 1)
        avg_norm_accuracy /= max(evaluated, 1)
        results['Average_acc'] = avg_accuracy
        results['Average_norm_acc'] = avg_norm_accuracy
        results['config_name'] = args.config_name
        results['eval_split'] = args.eval_split

        print(f'Average Accuracy is {np.round(avg_accuracy, 3)}')
        print(f'Average Normalized Accuracy is {np.round(avg_norm_accuracy, 3)}')

        if args.output_csv:
            os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
            write_to_csv(results, args.output_csv)
            print(f'wrote {args.output_csv}')


if __name__ == '__main__':
    main()
