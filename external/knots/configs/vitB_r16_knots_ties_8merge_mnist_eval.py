import os

VIT_ARCH = 'ViT-B-32-CLIP'
CACHE_DIR = os.environ.get('HF_HOME', '/root/autodl-tmp/hf-cache')
HEAD_DIR = ''

config = {
    'dataset': [
        {
            'name': 'mnist',
            'shuffle_train': True,
            'crop_ratio': 1.0,
            'clip_encodings': os.path.join(HEAD_DIR, VIT_ARCH, 'mnist_head.pt'),
            'val_fraction': 0.2,
            'batch_size': 128,
            'num_workers': 4,
            'shuffled_idxs': os.path.join(os.getcwd(), 'dataset/shuffled_idxs/mnist_shuffled_idxs.pt'),
        },
    ],
    'model': {
        'name': 'hf_clip',
        'base_type': 'openai/clip-vit-base-patch32',
        'cachedir': CACHE_DIR,
        'bases': [
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_stanford_cars',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_dtd',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_eurosat',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_gtsrb',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_mnist',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_resisc45',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_sun397',
            'hoffman-lab/KnOTS-ViT-B-32_lora_R16_svhn',
        ],
        'ft_config': {
            'type': 'lora',
            'r': 16,
            'lora_alpha': 16,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'out_proj'],
            'lora_dropout': 0.1,
            'bias': 'none',
        },
    },
    'task_merge_config': {
        'representation': 'svd-vector',
        'sign_resolve_mode': 'sum_of_values',
        'scaling_coeffs': 0.6,
        'topK': 20,
        'merge_method': 'ties',
        'merging_type': 'mean',
        'concat_across_output': True,
        'dare': False,
        'dare_pruning_coeffs': 0.0,
    },
    'eval_type': 'clip',
}
