# DSI
Code for paper "Dual-Space Intervention for Mitigating Bias in Robust Visual Question Answering"

# Abstract


Visual Question Answering (VQA) evaluates the visual-textual reasoning capabilities of intelligent agents. However, existing methods often exploit biases between questions and answers rather than demonstrating genuine reasoning abilities. These methods heavily rely on dataset-specific knowledge, with spurious correlations and dataset biases compromising their generalization ability. To address these long-standing challenges, we propose the Dual-Space Intervention (DSI) method, which categorizes biases into language bias and distribution bias. Two key innovations are included in our work: (1) To mitigate language bias, we introduce an adaptive question shuffling approach that dynamically determines the optimal shuffling proportion based on question difficulty, ensuring models develop a deeper understanding of the problem context, rather than relying on spurious word-answer correlations; (2) To tackle distribution bias, we develop a novel label rebalancing method that integrates long-tailed distribution recognition into robust VQA frameworks, specifically targeting head classes. This approach reduces the disproportionately high variance in head logits relative to tail logits, improving tail class recognition accuracy. Extensive experiments on four benchmarks (VQA-CP v1, VQA-CP v2, VQA-CE, and SLAKE-CP) demonstrate our method's superiority, with VQA-CP v1 and SLAKE-CP achieving state-of-the-art performance at 63.14\% and 37.61\% respectively. 


## Prerequisites
Please make sure you are using a NVIDIA GPU with Python==3.8.8 and about 100 GB of disk space.

Install all requirements with `pip install -r requirements.txt`

## Data Setup
Download UpDn features from [google drive](https://drive.google.com/drive/folders/111ipuYC0BeprYZhHXLzkRGeYAHcTT0WR), which is the link from [this repo](https://github.com/GeraldHan/GGE), into `/data/detection_features` folder

Download questions/answers for VQA v2, VQA-CP v1 and VQA-CP v2 by executing `bash tools/download.sh`

### GloVe Vectors
```
wget -P data http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove.6B.zip -d data/glove
rm data/glove.6B.zip
```
### VQA-CP v2
```
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_annotations.json
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_annotations.json
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_train_questions.json
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v2_test_questions.json
```
### VQA-CP v1
```
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_train_annotations.json
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_test_annotations.json
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_train_questions.json
wget -P data https://computing.ece.vt.edu/~aish/vqacp/vqacp_v1_test_questions.json
```

### VQA v2
```
wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
unzip data/v2_Questions_Train_mscoco.zip -d data
rm data/v2_Questions_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
unzip data/v2_Questions_Val_mscoco.zip -d data
rm data/v2_Questions_Val_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip
unzip data/v2_Questions_Test_mscoco.zip -d data
rm data/v2_Questions_Test_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
unzip data/v2_Annotations_Train_mscoco.zip -d data
rm data/v2_Annotations_Train_mscoco.zip

wget -P data https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
unzip data/v2_Annotations_Val_mscoco.zip -d data
rm data/v2_Annotations_Val_mscoco.zip
```
Preprocess process the data with `bash tools/process.sh`

## Training
Run `python main.py` to run our DSI.

## Evaluating
Run `python eval.py --load_path DIRNAME` to evaluate your model.

## Acknowledgements
The code framework for this repository is primarily sourced from [here](https://github.com/chojw/genb).
