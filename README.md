### Encoding Heterogeneous Social and Political Context for Entity Stance Prediction
This repository serves as a code listing for the paper 'Encoding Heterogeneous Social and Political Context for Entity Stance Prediction'. Arxiv version of the paper is available at https://arxiv.org/abs/2108.03881. Work in progress.

#### Abstract
Political stance detection is an important and challenging task, while existing works mainly focus on identifying perspectives in news articles or social media posts. However, semantic ambiguity and quote abundance prevent text-based methods from capturing the full picture of one's beliefs, thus we should shift the focus of stance detection from text to social entities. In this paper, we propose the novel task of entity stance prediction, which aims to predict entities' stances given their social and political context. Specifically, we retrieve facts from Wikipedia about social entities regarding contemporary U.S. politics. We then annotate social entities' stances towards political ideologies with the help of domain experts. After defining the task of entity stance prediction, we propose a graph-based solution, which constructs a heterogeneous information network from collected facts and adopts gated relational graph convolutional networks for representation learning. Our model is then trained with a combination of supervised, self-supervised and unsupervised loss functions, which are motivated by multiple social and political phenomena. We conduct extensive experiments to compare our method with existing text and graph analysis baselines. Our model achieves highest stance detection accuracy and yields inspiring insights regarding social entity stances. We further conduct ablation study and parameter analysis to study the mechanism and effectiveness of our proposed approach.

#### Entity Stance Prediction Data Set
Our collected entity stance prediction data set is avaiable in /data.
