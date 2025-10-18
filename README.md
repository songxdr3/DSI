# DSI
Code for paper "Dual-Space Intervention for Mitigating Bias in Robust Visual Question Answering"

# Abstract


Visual Question Answering (VQA) evaluates the visual-textual reasoning capabilities of intelligent agents. However, existing methods often exploit biases between questions and answers rather than demonstrating genuine reasoning abilities. These methods heavily rely on dataset-specific knowledge, with spurious correlations and dataset biases compromising their generalization ability. To address these long-standing challenges, we propose the Dual-Space Intervention (DSI) method, which categorizes biases into language bias and distribution bias. Two key innovations are included in our work: (1) To mitigate language bias, we introduce an adaptive question shuffling approach that dynamically determines the optimal shuffling proportion based on question difficulty, ensuring models develop a deeper understanding of the problem context, rather than relying on spurious word-answer correlations; (2) To tackle distribution bias, we develop a novel label rebalancing method that integrates long-tailed distribution recognition into robust VQA frameworks, specifically targeting head classes. This approach reduces the disproportionately high variance in head logits relative to tail logits, improving tail class recognition accuracy. Extensive experiments on four benchmarks (VQA-CP v1, VQA-CP v2, VQA-CE, and SLAKE-CP) demonstrate our method's superiority, with VQA-CP v1 and SLAKE-CP achieving state-of-the-art performance at 63.14\% and 37.61\% respectively. 


## Prerequisites
Please make sure you are using a NVIDIA GPU with Python==3.8.8 and about 100 GB of disk space.


## Data Setup
Download UpDn features from [google drive](https://drive.google.com/drive/folders/111ipuYC0BeprYZhHXLzkRGeYAHcTT0WR), which is the link from this repo, into /data/detection_features folder

Download questions/answers for VQA v2, VQA-CP v1 and VQA-CP v2 by executing bash tools/download.sh

Preprocess process the data with bash tools/process.sh

## Training
Run python main.py to run DSI.

## Evaluating
Run python eval.py --load_path DIRNAME to evaluate your model.
