# Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation

Resources for ACL 2023 paper "Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation"

## Dependencies
- python==3.9.7
- torch==1.10.1
- configargparse==1.4
- numpy==1.21.2
- transformers==4.11.3
- scikit-learn==1.0.2
- tqdm==4.62.3
- pandas=1.3.5

## Training a model
In `run.sh`, change `CUDA_DEVICE`, `CACHE_DIR` and `OUTPUT_DIR` to your local setting.

Run

```
sh ./run.sh
```

Option explanations:
- `wiki_path`: path to the Wikipedia snippet
- `predict_topic`: topic prediction training
- `predict_stance_neg`: unlikelihood training for stance label
- `predict_topic_neg`: unlikelihood training for topic words (errata: we forgot to include learning topics via unlikelihood in paper description, which shares the same motivation and formulation as the one with stance label)


## Reference
```
@inproceedings{acl-2023-stance-detection,
	title = "Zero-Shot and Few-Shot Stance Detection on Varied Topics via Conditional Generation",
	author = "Wen, Haoyang and Hauptmann, Alexander G.",
	booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics",
	year = 2023,
}
```

## Acknowledgement

The Wikipedia snippet is directly following the repository of the paper ["Infusing Wikipedia Knowledge to Enhance Stance Detection"](https://github.com/zihaohe123/wiki-enhanced-stance-detection) Zihao He, Negar Mokhberian, Kristina Lerman
