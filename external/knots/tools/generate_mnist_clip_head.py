import argparse
import os
import sys

import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.templates import get_templates


MNIST_CLASSNAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def build_classification_head(model, tokenizer, classnames, templates, device):
    logit_scale = model.logit_scale
    zeroshot_weights = []

    print('Building MNIST CLIP classification head.')
    with torch.no_grad():
        for classname in tqdm(classnames):
            embeddings = []
            for template in templates:
                tokenized = tokenizer(template(classname))
                tokenized = {
                    key: torch.tensor(value).to(device).reshape(1, -1)
                    for key, value in tokenized.items()
                }
                embedding = model.text_projection(model.text_model(**tokenized)[1])
                embeddings.append(embedding)

            embeddings = torch.concat(embeddings, dim=0)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            embedding = embeddings.mean(dim=0, keepdim=True)
            embedding /= embedding.norm()
            zeroshot_weights.append(embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
        zeroshot_weights *= logit_scale.exp()
        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    return zeroshot_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--cache-dir', default=os.environ.get('HF_HOME', '/root/autodl-tmp/hf-cache'))
    parser.add_argument('--output', default='ViT-B-32-CLIP/mnist_head.pt')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model, cache_dir=args.cache_dir).eval().to(device)
    processor = CLIPProcessor.from_pretrained(args.model, cache_dir=args.cache_dir)
    templates = get_templates('mnist')

    head = build_classification_head(
        model=model,
        tokenizer=processor.tokenizer,
        classnames=MNIST_CLASSNAMES,
        templates=templates,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(head, args.output)
    print(f'wrote {args.output} with shape {tuple(head.shape)}')


if __name__ == '__main__':
    main()
