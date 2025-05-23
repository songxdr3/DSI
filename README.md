# DSI
Code for paper "Dual-Space Intervention for Mitigating Bias in Robust Visual Question Answering"

# Abstract


Visual Question Answering (VQA) evaluates the visual-textual reasoning capabilities of intelligent agents. However, existing methods often exploit biases between questions and answers rather than demonstrating genuine reasoning abilities. These methods heavily rely on dataset-specific knowledge, with spurious correlations and dataset biases compromising their generalization ability. To address these long-standing challenges, we propose the Dual-Space Intervention (DSI) method, which categorizes biases into language bias and distribution bias. Two key innovations are included in our work: (1) To mitigate language bias, we introduce an adaptive question shuffling approach that dynamically determines the optimal shuffling proportion based on question difficulty, ensuring models develop a deeper understanding of the problem context, rather than relying on spurious word-answer correlations; (2) To tackle distribution bias, we develop a novel label rebalancing method that integrates long-tailed distribution recognition into robust VQA frameworks, specifically targeting head classes. This approach reduces the disproportionately high variance in head logits relative to tail logits, improving tail class recognition accuracy. Extensive experiments on four benchmarks (VQA-CP v1, VQA-CP v2, VQA-CE, and SLAKE-CP) demonstrate our method's superiority, with VQA-CP v1 and SLAKE-CP achieving state-of-the-art performance at 63.14\% and 37.06\% respectively. 

The code will be released soon.
