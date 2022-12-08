#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transformers import ViTFeatureExtractor, ViTModel
import torch
import datasets
import numpy as np

# dataset = datasets.load_dataset("huggingface/cats-image")
# dataset = datasets.load_dataset("data/oxford")
dataset = datasets.load_dataset("data/vehicle")
image = dataset["train"]["image"][0]

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTModel.from_pretrained("google/vit-base-patch16-224")


def image_to_vec(image):
    ls = {}
    inputs = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        hidden_states = model(**inputs, return_dict=True, output_hidden_states=True).hidden_states

        output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        vec = output_hidden_state.cpu().numpy()[0]
        ls['first_last_avg'] = vec

        output_hidden_state = (hidden_states[-1]).mean(dim=1)
        vec = output_hidden_state.cpu().numpy()[0]
        ls['last_avg'] = vec

        output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        vec = output_hidden_state.cpu().numpy()[0]
        ls['last2avg'] = vec

        output_hidden_state = (hidden_states[-1])[:, 0, :]
        vec = output_hidden_state.cpu().numpy()[0]
        ls['cls'] = vec

    return ls


print(image_to_vec(image))
