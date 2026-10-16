# RAG Evaluation: Evidence Review

## Question 1

**Question:** Why does the Transformer eliminate recurrence?

**Original category:** answerable

### Rank 1 — 1_2_1 (page 2, similarity 0.346)

**Source:** Attention.pdf

through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains. Attention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs. 2 Background The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for

---

### Rank 2 — 1_5_1 (page 5, similarity 0.338)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 3 — 1_3_0 (page 3, similarity 0.297)

**Source:** Attention.pdf

Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 3.1 Encoder and Decoder Stacks Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output

---

### Rank 4 — 1_2_3 (page 2, similarity 0.284)

**Source:** Attention.pdf

in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22]. End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34]. To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9]. 3 Model Architecture Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a

---

### Rank 5 — 1_2_4 (page 2, similarity 0.267)

**Source:** Attention.pdf

encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next. 2

---

## Question 2

**Question:** How does scaled dot-product attention work?

**Original category:** answerable

### Rank 1 — 1_4_2 (page 4, similarity 0.667)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 2 — 1_4_1 (page 4, similarity 0.642)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

### Rank 3 — 1_4_0 (page 4, similarity 0.617)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

### Rank 4 — 1_1_2 (page 1, similarity 0.555)

**Source:** Attention.pdf

of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research. †Work performed while at Google Brain. ‡Work performed while at Google Research. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. arXiv:1706.03762v7 [cs.CL] 2 Aug 2023

---

### Rank 5 — 1_5_1 (page 5, similarity 0.447)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

## Question 3

**Question:** What is the purpose of multi-head attention?

**Original category:** answerable

### Rank 1 — 1_5_0 (page 5, similarity 0.561)

**Source:** Attention.pdf

output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. MultiHead(Q, K, V ) = Concat(head1, ..., headh)W O where headi = Attention(QW Q i , KW K i , V W V i ) Where the projections are parameter matrices W Q i ∈Rdmodel×dk, W K i ∈Rdmodel×dk, W V i ∈Rdmodel×dv and W O ∈Rhdv×dmodel. In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the

---

### Rank 2 — 1_13_0 (page 13, similarity 0.460)

**Source:** Attention.pdf

Attention Visualizations It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS> <pad> <pad> <pad> <pad> <pad> <pad> It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS> <pad> <pad> <pad> <pad> <pad> <pad> Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color. 13

---

### Rank 3 — 1_4_2 (page 4, similarity 0.453)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 4 — 1_5_1 (page 5, similarity 0.443)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 5 — 1_3_1 (page 3, similarity 0.435)

**Source:** Attention.pdf

= 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i. 3.2 Attention An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum 3

---

## Question 4

**Question:** How does Bahdanau attention calculate alignment scores?

**Original category:** answerable

### Rank 1 — 3_12_3 (page 12, similarity 0.474)

**Source:** NMT.pdf

the output probabilities (one for each word) with a softmax function (see Eq. (6)). A.1.2 ALIGNMENT MODEL The alignment model should be designed considering that the model needs to be evaluated Tx × Ty times for each sentence pair of lengths Tx and Ty. In order to reduce computation, we use a single- layer multilayer perceptron such that a(si−1, hj) = v⊤ a tanh (Wasi−1 + Uahj) , where Wa ∈Rn×n, Ua ∈Rn×2n and va ∈Rn are the weight matrices. Since Uahj does not depend on i, we can pre-compute it in advance to minimize the computational cost. 8 Here, we show the formula of the decoder. The same formula can be used in the encoder by simply ignoring the context vector ci and the related terms. 12

---

### Rank 2 — 3_4_0 (page 4, similarity 0.465)

**Source:** NMT.pdf

Published as a conference paper at ICLR 2015 the alignment is not considered to be a latent variable. Instead, the alignment model directly com- putes a soft alignment, which allows the gradient of the cost function to be backpropagated through. This gradient can be used to train the alignment model as well as the whole translation model jointly. We can understand the approach of taking a weighted sum of all the annotations as computing an expected annotation, where the expectation is over possible alignments. Let αij be a probability that the target word yi is aligned to, or translated from, a source word xj. Then, the i-th context vector ci is the expected annotation over all the annotations with probabilities αij. The probability αij, or its associated energy eij, reﬂects the importance of the annotation hj with respect to the previous hidden state si−1 in deciding the next state si and generating yi. Intuitively, this implements a mechanism of attention in the decoder. The decoder decides parts of the source sentence to pay attention to. By letting the decoder have

---

### Rank 3 — 3_3_3 (page 3, similarity 0.420)

**Source:** NMT.pdf

The weight αij of each annotation hj is computed by αij = exp (eij) PTx k=1 exp (eik) , (6) where eij = a(si−1, hj) is an alignment model which scores how well the inputs around position j and the output at position i match. The score is based on the RNN hidden state si−1 (just before emitting yi, Eq. (4)) and the j-th annotation hj of the input sentence. We parametrize the alignment model a as a feedforward neural network which is jointly trained with all the other components of the proposed system. Note that unlike in traditional machine translation, 3

---

### Rank 4 — 3_7_3 (page 7, similarity 0.409)

**Source:** NMT.pdf

of the soft align- ment is that it naturally deals with source and target phrases of different lengths, without requiring a counter-intuitive way of mapping some words to or from nowhere ([NULL]) (see, e.g., Chapters 4 and 5 of Koehn, 2010). 5.2.2 LONG SENTENCES As clearly visible from Fig. 2 the proposed model (RNNsearch) is much better than the conventional model (RNNencdec) at translating long sentences. This is likely due to the fact that the RNNsearch does not require encoding a long sentence into a ﬁxed-length vector perfectly, but only accurately encoding the parts of the input sentence that surround a particular word. As an example, consider this source sentence from the test set: An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital. The RNNencdec-50 translated this sentence into: Un privil`ege d’admission est le droit d’un m´edecin de reconnaˆıtre un patient `a l’hˆopital ou un centre m´edical d’un diagnostic

---

### Rank 5 — 3_7_0 (page 7, similarity 0.406)

**Source:** NMT.pdf

Published as a conference paper at ICLR 2015 Model All No UNK◦ RNNencdec-30 13.93 24.19 RNNsearch-30 21.50 31.44 RNNencdec-50 17.82 26.71 RNNsearch-50 26.75 34.16 RNNsearch-50⋆ 28.45 36.15 Moses 33.30 35.63 Table 1: BLEU scores of the trained models com- puted on the test set. The second and third columns show respectively the scores on all the sentences and, on the sentences without any unknown word in them- selves and in the reference translations. Note that RNNsearch-50⋆was trained much longer until the performance on the development set stopped improv- ing. (◦) We disallowed the models to generate [UNK] tokens when only the sentences having no unknown words were evaluated (last column). 5.2 QUALITATIVE ANALYSIS 5.2.1 ALIGNMENT The proposed approach provides an intuitive way to inspect the (soft-)alignment between the words in a generated translation and those in a source sentence. This is done by visualizing the annotation weights αij from Eq. (6), as in Fig. 3. Each row of a matrix in each plot indicates the weights associated with the annotations. From this we see which positions in the source sentence

---

## Question 5

**Question:** What is the difference between sparse and dense retrieval?

**Original category:** answerable

### Rank 1 — 4_2_4 (page 2, similarity 0.524)

**Source:** ODQA.pdf

et al., 2017).4 Formally speaking, a retriever R : (q, C) →CF is a function that takes as input a question q and a corpus C and returns a much smaller ﬁlter set of texts CF ⊂C, where |CF| = k ≪|C|. For a ﬁxed k, a retriever can be evaluated in isolation on top-k retrieval accuracy, which is the fraction of ques- tions for which CF contains a span that answers the question. 3 Dense Passage Retriever (DPR) We focus our research in this work on improv- ing the retrieval component in open-domain QA. Given a collection of M text passages, the goal of our dense passage retriever (DPR) is to index all the passages in a low-dimensional and continuous space, such that it can retrieve efﬁciently the top k passages relevant to the input question for the reader at run-time. Note that M can be very large (e.g., 21 million passages in our experiments, de- scribed in Section 4.1) and k is usually small, such as 20–100. 3.1 Overview Our dense passage retriever (DPR) uses a dense encoder

---

### Rank 2 — 4_1_2 (page 1, similarity 0.464)

**Source:** ODQA.pdf

been released at https://github.com/facebookresearch/DPR. 2For instance, the exact match score on SQuAD v1.1 drops from above 80% to less than 40% (Yang et al., 2019a). Retrieval in open-domain QA is usually imple- mented using TF-IDF or BM25 (Robertson and Zaragoza, 2009), which matches keywords efﬁ- ciently with an inverted index and can be seen as representing the question and context in high- dimensional, sparse vectors (with weighting). Con- versely, the dense, latent semantic encoding is com- plementary to sparse representations by design. For example, synonyms or paraphrases that consist of completely different tokens may still be mapped to vectors close to each other. Consider the question “Who is the bad guy in lord of the rings?”, which can be answered from the context “Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy.” A term-based system would have difﬁculty retrieving such a context, while a dense retrieval system would be able to better match “bad guy” with “villain” and fetch the cor- rect context. Dense encodings are also learnable by adjusting the embedding

---

### Rank 3 — 5_19_1 (page 19, similarity 0.464)

**Source:** RAG.pdf

728 dimensional vectors, consisting of 15.3B values. These can be easily be stored at 8-bit ﬂoating point precision to manage memory and disk footprints. H Retrieval Collapse In preliminary experiments, we observed that for some tasks such as story generation [11], the retrieval component would “collapse” and learn to retrieve the same documents regardless of the input. In these cases, once retrieval had collapsed, the generator would learn to ignore the documents, and the RAG model would perform equivalently to BART. The collapse could be due to a less-explicit requirement for factual knowledge in some tasks, or the longer target sequences, which could result in less informative gradients for the retriever. Perez et al. [46] also found spurious retrieval results when optimizing a retrieval component in order to improve performance on downstream tasks. I Number of instances per dataset The number of training, development and test datapoints in each of our datasets is shown in Table 7. 19

---

### Rank 4 — 4_2_5 (page 2, similarity 0.457)

**Source:** ODQA.pdf

at run-time. Note that M can be very large (e.g., 21 million passages in our experiments, de- scribed in Section 4.1) and k is usually small, such as 20–100. 3.1 Overview Our dense passage retriever (DPR) uses a dense encoder EP (·) which maps any text passage to a d- dimensional real-valued vectors and builds an index for all the M passages that we will use for retrieval. 3The ideal size and boundary of a text passage are func- tions of both the retriever and reader. We also experimented with natural paragraphs in our preliminary trials and found that using ﬁxed-length passages performs better in both retrieval and ﬁnal QA accuracy, as observed by Wang et al. (2019). 4Exceptions include (Seo et al., 2019) and (Roberts et al., 2020), which retrieves and generates the answers, respectively.

---

### Rank 5 — 4_9_3 (page 9, similarity 0.443)

**Source:** ODQA.pdf

hard negatives, Xiong et al. (2020a) use the retrieval model trained in the pre- vious iteration to discover new negatives and con- struct a different set of examples in each training iteration. Starting from our trained DPR model, they show that the retrieval performance can be further improved. Recent work (Izacard and Grave, 2020; Lewis et al., 2020b) have also shown that DPR can be combined with generation models such as BART (Lewis et al., 2020a) and T5 (Raf- fel et al., 2019), achieving good performance on open-domain QA and other knowledge-intensive tasks. 8 Conclusion In this work, we demonstrated that dense retrieval can outperform and potentially replace the tradi- tional sparse retrieval component in open-domain question answering. While a simple dual-encoder approach can be made to work surprisingly well, we showed that there are some critical ingredients to training a dense retriever successfully. Moreover, our empirical analysis and ablation studies indicate that more complex model frameworks or similarity functions do not necessarily provide additional val- ues. As a result of improved retrieval performance, we obtained new state-of-the-art results on

---

## Question 6

**Question:** How does RAG combine retrieval and generation?

**Original category:** answerable

### Rank 1 — 5_6_2 (page 6, similarity 0.600)

**Source:** RAG.pdf

that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results, over 452 pairs of generations from BART and RAG-Token. Evaluators indicated that BART was more factual than RAG in only 7.1% of cases, while RAG was more factual in 42.7% of cases, and both RAG and BART were factual in a further 17% of cases, clearly demonstrating the effectiveness of RAG on the task over a state-of-the-art generation model. Evaluators also ﬁnd RAG generations to be more speciﬁc by a large margin. Table 3 shows typical generations from each model. Jeopardy questions often contain two separate pieces of information, and RAG-Token may perform best because it can generate responses that combine content from several documents. Figure 2 shows an example. When generating “Sun”, the posterior is high for document 2 which mentions “The Sun Also Rises”. Similarly, document 1 dominates the posterior when “A Farewell to Arms” is generated. Intriguingly,

---

### Rank 2 — 5_6_4 (page 6, similarity 0.573)

**Source:** RAG.pdf

is a novel by this author of "A with "The Sun Also Rises" is a novel by this author of "A Farewell to Arms". This example shows how parametric and non-parametric memories work together—the non-parametric component helps to guide the generation, drawing out speciﬁc knowledge stored in the parametric memory. 4.4 Fact Veriﬁcation Table 2 shows our results on FEVER. For 3-way classiﬁcation, RAG scores are within 4.3% of state-of-the-art models, which are complex pipeline systems with domain-speciﬁc architectures and substantial engineering, trained using intermediate retrieval supervision, which RAG does not require. 6

---

### Rank 3 — 5_7_3 (page 7, similarity 0.555)

**Source:** RAG.pdf

gold article is present in the top 10 retrieved articles in 90% of cases. 4.5 Additional Results Generation Diversity Section 4.3 shows that RAG models are more factual and speciﬁc than BART for Jeopardy question generation. Following recent work on diversity-promoting decoding [33, 59, 39], we also investigate generation diversity by calculating the ratio of distinct ngrams to total ngrams generated by different models. Table 5 shows that RAG-Sequence’s generations are more diverse than RAG-Token’s, and both are signiﬁcantly more diverse than BART without needing any diversity-promoting decoding. Retrieval Ablations A key feature of RAG is learning to retrieve relevant information for the task. To assess the effectiveness of the retrieval mechanism, we run ablations where we freeze the retriever during training. As shown in Table 6, learned retrieval improves results for all tasks. We compare RAG’s dense retriever to a word overlap-based BM25 retriever [53]. Here, we replace RAG’s retriever with a ﬁxed BM25 system, and use BM25 retrieval scores as logits when calculating p(z|x). Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims

---

### Rank 4 — 5_8_2 (page 8, similarity 0.553)

**Source:** RAG.pdf

affect performance and runtime. Figure 3 (left) shows that retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence, but performance peaks for RAG-Token at 10 retrieved documents. Figure 3 (right) shows that retrieving more documents leads to higher Rouge-L for RAG-Token at the expense of Bleu-1, but the effect is less pronounced for RAG-Sequence. 10 20 30 40 50 K Retrieved Docs 39 40 41 42 43 44 NQ Exact Match RAG-Tok RAG-Seq 10 20 30 40 50 K Retrieved Docs 40 50 60 70 80 NQ Answer Recall @ K RAG-Tok RAG-Seq Fixed DPR BM25 10 20 30 40 50 K Retrieved Docs 48 50 52 54 56 Bleu-1 / Rouge-L score RAG-Tok R-L RAG-Tok B-1 RAG-Seq R-L RAG-Seq B-1 Figure 3: Left: NQ performance as more documents are retrieved. Center: Retrieval recall perfor- mance in NQ. Right: MS-MARCO Bleu-1 and Rouge-L as more documents are retrieved. 5 Related Work Single-Task Retrieval Prior work has shown that retrieval improves performance across a variety of NLP tasks when considered in isolation. Such tasks include open-domain question

---

### Rank 5 — 5_19_1 (page 19, similarity 0.535)

**Source:** RAG.pdf

728 dimensional vectors, consisting of 15.3B values. These can be easily be stored at 8-bit ﬂoating point precision to manage memory and disk footprints. H Retrieval Collapse In preliminary experiments, we observed that for some tasks such as story generation [11], the retrieval component would “collapse” and learn to retrieve the same documents regardless of the input. In these cases, once retrieval had collapsed, the generator would learn to ignore the documents, and the RAG model would perform equivalently to BART. The collapse could be due to a less-explicit requirement for factual knowledge in some tasks, or the longer target sequences, which could result in less informative gradients for the retriever. Perez et al. [46] also found spurious retrieval results when optimizing a retrieval component in order to improve performance on downstream tasks. I Number of instances per dataset The number of training, development and test datapoints in each of our datasets is shown in Table 7. 19

---

## Question 7

**Question:** How does DPR represent questions and passages?

**Original category:** answerable

### Rank 1 — 4_12_2 (page 12, similarity 0.491)

**Source:** ODQA.pdf

are sensitive to highly selective keywords and phrases, but cannot capture lexical variations or se- mantic relationships well. In contrast, DPR excels at semantic representation, but might lack sufﬁcient capacity to represent salient phrases which appear rarely. Table 7 illustrates this phenomenon with two examples. In the ﬁrst example, the top scor- ing passage from BM25 is irrelevant, even though keywords such as England and Ireland appear mul- tiple times. In comparison, DPR is able to return Top-1 Top-5 Top-20 Top-100 Gold 44.9 66.8 78.1 85.0 Dist. Sup. 43.9 65.3 77.1 84.4 Table 5: Retrieval accuracy on the development set of Natural Questions, trained on passages that match the gold context (Gold) or the top BM25 passage that con- tains the answer (Dist. Sup.). Sim Loss Retrieval Accuracy Top-1 Top-5 Top-20 Top-100 DP NLL 44.9 66.8 78.1 85.0 Triplet 41.6 65.0 77.2 84.5 L2 NLL 43.5 64.7 76.1 83.1 Triplet 42.2 66.0 78.1 84.9 Table 6: Retrieval Top-k accuracy on the development set of Natural Questions using different similarity and loss functions. the correct answer, presumably by matching “body of

---

### Rank 2 — 4_4_2 (page 4, similarity 0.450)

**Source:** ODQA.pdf

a set of trivia questions with answers that were originally scraped from the Web. WebQuestions (WQ) (Berant et al., 2013) consists of questions selected using Google Suggest API, where the answers are entities in Freebase. CuratedTREC (TREC) (Baudiˇs and ˇSediv`y, 2015) sources questions from TREC QA tracks 5However, Wang et al. (2019) also propose splitting docu- ments into overlapping passages, which we do not ﬁnd advan- tageous compared to the non-overlapping version. Dataset Train Dev Test Natural Questions 79,168 58,880 8,757 3,610 TriviaQA 78,785 60,413 8,837 11,313 WebQuestions 3,417 2,474 361 2,032 CuratedTREC 1,353 1,125 133 694 SQuAD 78,713 70,096 8,886 10,570 Table 1: Number of questions in each QA dataset. The two columns of Train denote the original training ex- amples in the dataset and the actual questions used for training DPR after ﬁltering. See text for more details. as well as various Web sources and is intended for open-domain QA from unstructured corpora. SQuAD v1.1 (Rajpurkar et al., 2016) is a popu- lar benchmark dataset for reading comprehension. Annotators were presented with a Wikipedia para- graph, and asked

---

### Rank 3 — 4_13_2 (page 13, similarity 0.436)

**Source:** ODQA.pdf

November 1980) is a Nor- wegian stage and screen actor. He appeared in the Norwe- gian ﬁlm ”Max Manus” and played Thor Heyerdahl in the Oscar-nominated 2012 ﬁlm ”Kon-Tiki”. Pl Hagen was born in Stavanger, Norway, the son of Roar Hagen, a Norwegian cartoonist who has long been associated with Norway´s largest daily, ”VG”. He lived in Jtten, a neighborhood in the city of Stavanger in south-western Norway. . . . Table 7: Examples of passages returned from BM25 and DPR. Correct answers are written in blue and the content words in the question are written in bold. from the top 5 positive and top 30 negative passages (from the set of top 100 passages retrieved from the same question). The question encoder’s initial state is taken from a DPR model previously trained on the NQ dataset. The reader’s initial state is a BERT-base model. In terms of the end-to-end QA results, our joint-training scheme does not provide better results compared to the usual retriever/reader training pipeline, resulting in the same 39.8 exact match score on NQ dev as in

---

### Rank 4 — 3_7_4 (page 7, similarity 0.396)

**Source:** NMT.pdf

a procedure, based on his status as a health care worker at a hospital. The RNNencdec-50 translated this sentence into: Un privil`ege d’admission est le droit d’un m´edecin de reconnaˆıtre un patient `a l’hˆopital ou un centre m´edical d’un diagnostic ou de prendre un diagnostic en fonction de son ´etat de sant´e. 7

---

### Rank 5 — 4_4_4 (page 4, similarity 0.380)

**Source:** ODQA.pdf

passages has the answer, the ques- tion will be discarded. For SQuAD and Natural Questions, since the original passages have been split and processed differently than our pool of candidate passages, we match and replace each gold passage with the corresponding passage in the candidate pool.7 We discard the questions when the matching is failed due to different Wikipedia versions or pre-processing. Table 1 shows the num- ber of questions in training/dev/test sets for all the datasets and the actual questions used for training the retriever. 5 Experiments: Passage Retrieval In this section, we evaluate the retrieval perfor- mance of our Dense Passage Retriever (DPR), along with analysis on how its output differs from 6We use the unﬁltered TriviaQA version and discard the noisy evidence documents mined from Bing. 7The improvement of using gold contexts over passages that contain answers is small. See Section 5.2 and Ap- pendix A.

---

## Question 8

**Question:** Why are sentence embeddings useful for semantic search?

**Original category:** answerable

### Rank 1 — 4_1_2 (page 1, similarity 0.586)

**Source:** ODQA.pdf

been released at https://github.com/facebookresearch/DPR. 2For instance, the exact match score on SQuAD v1.1 drops from above 80% to less than 40% (Yang et al., 2019a). Retrieval in open-domain QA is usually imple- mented using TF-IDF or BM25 (Robertson and Zaragoza, 2009), which matches keywords efﬁ- ciently with an inverted index and can be seen as representing the question and context in high- dimensional, sparse vectors (with weighting). Con- versely, the dense, latent semantic encoding is com- plementary to sparse representations by design. For example, synonyms or paraphrases that consist of completely different tokens may still be mapped to vectors close to each other. Consider the question “Who is the bad guy in lord of the rings?”, which can be answered from the context “Sala Baker is best known for portraying the villain Sauron in the Lord of the Rings trilogy.” A term-based system would have difﬁculty retrieving such a context, while a dense retrieval system would be able to better match “bad guy” with “villain” and fetch the cor- rect context. Dense encodings are also learnable by adjusting the embedding

---

### Rank 2 — 4_10_1 (page 10, similarity 0.557)

**Source:** ODQA.pdf

Indexing by latent semantic analysis. Jour- nal of the American society for information science, 41(6):391–407. Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language under- standing. In North American Association for Com- putational Linguistics (NAACL). David A Ferrucci. 2012. Introduction to “This is Wat- son”. IBM Journal of Research and Development, 56(3.4):1–1. Daniel Gillick, Sayali Kulkarni, Larry Lansing, Alessandro Presta, Jason Baldridge, Eugene Ie, and Diego Garcia-Olano. 2019. Learning dense repre- sentations for entity retrieval. In Computational Nat- ural Language Learning (CoNLL). Ruiqi Guo, Sanjiv Kumar, Krzysztof Choromanski, and David Simcha. 2016. Quantization based fast inner product search. In Artiﬁcial Intelligence and Statis- tics, pages 482–490. Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pa- supat, and Ming-Wei Chang. 2020. REALM: Retrieval-augmented language model pre-training. ArXiv, abs/2002.08909. Matthew Henderson, Rami Al-Rfou, Brian Strope, Yun- hsuan Sung, L´aszl´o Luk´acs, Ruiqi Guo, Sanjiv Ku- mar, Balint Miklos, and Ray Kurzweil. 2017. Efﬁ- cient natural language response suggestion for smart reply. ArXiv, abs/1705.00652. Po-Sen Huang, Xiaodong He, Jianfeng Gao, Li Deng, Alex Acero, and Larry

---

### Rank 3 — 2_2_0 (page 2, similarity 0.528)

**Source:** BERT.pdf

most similar sentence pair in a collection of 10,000 sentences is reduced from 65 hours with BERT to the computation of 10,000 sentence embeddings (~5 seconds with SBERT) and computing cosine- similarity (~0.01 seconds). By using optimized index structures, ﬁnding the most similar Quora question can be reduced from 50 hours to a few milliseconds (Johnson et al., 2017). We ﬁne-tune SBERT on NLI data, which cre- ates sentence embeddings that signiﬁcantly out- perform other state-of-the-art sentence embedding methods like InferSent (Conneau et al., 2017) and Universal Sentence Encoder (Cer et al., 2018). On seven Semantic Textual Similarity (STS) tasks, SBERT achieves an improvement of 11.7 points compared to InferSent and 5.5 points compared to Universal Sentence Encoder. On SentEval (Con- neau and Kiela, 2018), an evaluation toolkit for sentence embeddings, we achieve an improvement of 2.1 and 2.6 points, respectively. SBERT can be adapted to a speciﬁc task. It sets new state-of-the-art performance on a chal- lenging argument similarity dataset (Misra et al., 2016) and on a triplet dataset to distinguish sen- tences from different sections of a Wikipedia

---

### Rank 4 — 5_9_1 (page 9, similarity 0.522)

**Source:** RAG.pdf

pre-trained, neural language models [44, 26] similar to ours. Some work optimizes the retrieval module to aid in a speciﬁc, downstream task such as question answering, using search [46], reinforcement learning [6, 63, 62], or a latent variable approach [31, 20] as in our work. These successes leverage different retrieval-based architectures and optimization techniques to achieve strong performance on a single task, while we show that a single retrieval-based architecture can be ﬁne-tuned for strong performance on a variety of tasks. Memory-based Architectures Our document index can be seen as a large external memory for neural networks to attend to, analogous to memory networks [64, 55]. Concurrent work [14] learns to retrieve a trained embedding for each entity in the input, rather than to retrieve raw text as in our work. Other work improves the ability of dialog models to generate factual text by attending over fact embeddings [15, 13]. A key feature of our memory is that it is comprised of raw text rather distributed representations, which makes the memory both (i) human-readable, lending a form of interpretability to

---

### Rank 5 — 2_1_1 (page 1, similarity 0.522)

**Source:** BERT.pdf

sentence embeddings that can be com- pared using cosine-similarity. This reduces the effort for ﬁnding the most similar pair from 65 hours with BERT / RoBERTa to about 5 sec- onds with SBERT, while maintaining the ac- curacy from BERT. We evaluate SBERT and SRoBERTa on com- mon STS tasks and transfer learning tasks, where it outperforms other state-of-the-art sentence embeddings methods.1 1 Introduction In this publication, we present Sentence-BERT (SBERT), a modiﬁcation of the BERT network us- ing siamese and triplet networks that is able to derive semantically meaningful sentence embed- dings2. This enables BERT to be used for certain new tasks, which up-to-now were not applicable for BERT. These tasks include large-scale seman- 1Code available: https://github.com/UKPLab/ sentence-transformers 2With semantically meaningful we mean that semantically similar sentences are close in vector space. tic similarity comparison, clustering, and informa- tion retrieval via semantic search. BERT set new state-of-the-art performance on various sentence classiﬁcation and sentence-pair regression tasks. BERT uses a cross-encoder: Two sentences are passed to the transformer network and the target value is predicted. However, this setup is unsuitable

---

## Question 9

**Question:** What is the role of positional encoding?

**Original category:** answerable

### Rank 1 — 1_6_1 (page 6, similarity 0.541)

**Source:** Attention.pdf

of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies: PE(pos,2i) = sin(pos/100002i/dmodel) PE(pos,2i+1) = cos(pos/100002i/dmodel) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of PEpos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. 4 Why Self-Attention In this section we compare various

---

### Rank 2 — 1_5_1 (page 5, similarity 0.457)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 3 — 1_2_4 (page 2, similarity 0.453)

**Source:** Attention.pdf

encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next. 2

---

### Rank 4 — 1_5_2 (page 5, similarity 0.431)

**Source:** Attention.pdf

positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2. 3.3 Position-wise Feed-Forward Networks In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between. FFN(x) = max(0, xW1 + b1)W2 + b2 (2) While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048. 3.4 Embeddings and Softmax Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens

---

### Rank 5 — 1_3_1 (page 3, similarity 0.423)

**Source:** Attention.pdf

= 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i. 3.2 Attention An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum 3

---

## Question 10

**Question:** How does RAG use retrieved passages to generate answers?

**Original category:** answerable

### Rank 1 — 5_6_1 (page 6, similarity 0.609)

**Source:** RAG.pdf

21.4 40.8 44.2 to more effective marginalization over documents. Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%. 4.2 Abstractive Question Answering As shown in Table 2, RAG-Sequence outperforms BART on Open MS-MARCO NLG by 2.6 Bleu points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is impressive given that (i) those models access gold passages with speciﬁc information required to generate the reference answer , (ii) many questions are unanswerable without the gold passages, and (iii) not all questions are answerable from Wikipedia alone. Table 3 shows some generated answers from our models. Qualitatively, we ﬁnd that RAG models hallucinate less and generate factually correct text more often than BART. Later, we also show that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results,

---

### Rank 2 — 5_5_4 (page 5, similarity 0.605)

**Source:** RAG.pdf

the “closed-book” (parametric only) approaches and the performance of "open-book" retrieval-based approaches. Unlike REALM and T5+SSM, RAG enjoys strong results without expensive, specialized “salient span masking” pre-training [20]. It is worth noting that RAG’s retriever is initialized using DPR’s retriever, which uses retrieval supervision on Natural Questions and TriviaQA. RAG compares favourably to the DPR QA system, which uses a BERT-based “cross- encoder” to re-rank documents, along with an extractive reader. RAG demonstrates that neither a re-ranker nor extractive reader is necessary for state-of-the-art performance. There are several advantages to generating answers even when it is possible to extract them. Docu- ments with clues about the answer but do not contain the answer verbatim can still contribute towards a correct answer being generated, which is not possible with standard extractive approaches, leading 5

---

### Rank 3 — 5_4_4 (page 4, similarity 0.603)

**Source:** RAG.pdf

with our NQ RAG model. We use the same train/dev/test splits as prior work [31, 26] and report Exact Match (EM) scores. For TQA, to compare with T5 [52], we also evaluate on the TQA Wiki test set. 3.2 Abstractive Question Answering RAG models can go beyond simple extractive QA and answer questions with free-form, abstractive text generation. To test RAG’s natural language generation (NLG) in a knowledge-intensive setting, we use the MSMARCO NLG task v2.1 [43]. The task consists of questions, ten gold passages retrieved from a search engine for each question, and a full sentence answer annotated from the retrieved passages. We do not use the supplied passages, only the questions and answers, to treat 4

---

### Rank 4 — 5_6_2 (page 6, similarity 0.578)

**Source:** RAG.pdf

that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results, over 452 pairs of generations from BART and RAG-Token. Evaluators indicated that BART was more factual than RAG in only 7.1% of cases, while RAG was more factual in 42.7% of cases, and both RAG and BART were factual in a further 17% of cases, clearly demonstrating the effectiveness of RAG on the task over a state-of-the-art generation model. Evaluators also ﬁnd RAG generations to be more speciﬁc by a large margin. Table 3 shows typical generations from each model. Jeopardy questions often contain two separate pieces of information, and RAG-Token may perform best because it can generate responses that combine content from several documents. Figure 2 shows an example. When generating “Sun”, the posterior is high for document 2 which mentions “The Sun Also Rises”. Similarly, document 1 dominates the posterior when “A Farewell to Arms” is generated. Intriguingly,

---

### Rank 5 — 5_7_3 (page 7, similarity 0.570)

**Source:** RAG.pdf

gold article is present in the top 10 retrieved articles in 90% of cases. 4.5 Additional Results Generation Diversity Section 4.3 shows that RAG models are more factual and speciﬁc than BART for Jeopardy question generation. Following recent work on diversity-promoting decoding [33, 59, 39], we also investigate generation diversity by calculating the ratio of distinct ngrams to total ngrams generated by different models. Table 5 shows that RAG-Sequence’s generations are more diverse than RAG-Token’s, and both are signiﬁcantly more diverse than BART without needing any diversity-promoting decoding. Retrieval Ablations A key feature of RAG is learning to retrieve relevant information for the task. To assess the effectiveness of the retrieval mechanism, we run ablations where we freeze the retriever during training. As shown in Table 6, learned retrieval improves results for all tasks. We compare RAG’s dense retriever to a word overlap-based BM25 retriever [53]. Here, we replace RAG’s retriever with a ﬁxed BM25 system, and use BM25 retrieval scores as logits when calculating p(z|x). Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims

---

## Question 11

**Question:** What is additive attention?

**Original category:** answerable

### Rank 1 — 1_13_0 (page 13, similarity 0.529)

**Source:** Attention.pdf

Attention Visualizations It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS> <pad> <pad> <pad> <pad> <pad> <pad> It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult . <EOS> <pad> <pad> <pad> <pad> <pad> <pad> Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color. 13

---

### Rank 2 — 1_2_2 (page 2, similarity 0.439)

**Source:** Attention.pdf

2 Background The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2. Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22]. End-to-end memory networks are based

---

### Rank 3 — 1_14_0 (page 14, similarity 0.417)

**Source:** Attention.pdf

The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . <EOS> <pad> The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . <EOS> <pad> The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . <EOS> <pad> The Law will never be perfect , but its application should be just - this is what we are missing , in my opinion . <EOS> <pad> Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word. 14

---

### Rank 4 — 1_7_1 (page 7, similarity 0.416)

**Source:** Attention.pdf

the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d2). Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model. As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences. 5 Training This section describes the training regime for our models. 5.1 Training Data and Batching We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source- target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting

---

### Rank 5 — 1_4_1 (page 4, similarity 0.415)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

## Question 12

**Question:** How is dot-product attention computationally efficient?

**Original category:** answerable

### Rank 1 — 1_4_2 (page 4, similarity 0.716)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 2 — 1_4_1 (page 4, similarity 0.707)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

### Rank 3 — 1_4_0 (page 4, similarity 0.660)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

### Rank 4 — 1_1_2 (page 1, similarity 0.610)

**Source:** Attention.pdf

of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research. †Work performed while at Google Brain. ‡Work performed while at Google Research. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. arXiv:1706.03762v7 [cs.CL] 2 Aug 2023

---

### Rank 5 — 1_6_2 (page 6, similarity 0.530)

**Source:** Attention.pdf

nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. 4 Why Self-Attention In this section we compare various aspects of self-attention layers to the recurrent and convolu- tional layers commonly used for mapping one variable-length sequence of symbol representations (x1, ..., xn) to another sequence of equal length (z1, ..., zn), with xi, zi ∈Rd, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata. One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required. The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these

---

## Question 13

**Question:** What is the purpose of the encoder-decoder architecture in neural machine translation?

**Original category:** answerable

### Rank 1 — 3_1_2 (page 1, similarity 0.673)

**Source:** NMT.pdf

neural machine translation models belong to a family of encoder– decoders (Sutskever et al., 2014; Cho et al., 2014a), with an encoder and a decoder for each lan- guage, or involve a language-speciﬁc encoder applied to each sentence whose outputs are then com- pared (Hermann and Blunsom, 2014). An encoder neural network reads and encodes a source sen- tence into a ﬁxed-length vector. A decoder then outputs a translation from the encoded vector. The whole encoder–decoder system, which consists of the encoder and the decoder for a language pair, is jointly trained to maximize the probability of a correct translation given a source sentence. A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a ﬁxed-length vector. This may make it difﬁcult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus. Cho et al. (2014b) showed that indeed the performance of a basic encoder–decoder deteriorates rapidly as the length of an

---

### Rank 2 — 3_10_2 (page 10, similarity 0.580)

**Source:** NMT.pdf

in Natural Language Processing (EMNLP 2014). to appear. Cho, K., van Merri¨enboer, B., Bahdanau, D., and Bengio, Y. (2014b). On the properties of neural machine translation: Encoder–Decoder approaches. In Eighth Workshop on Syntax, Semantics and Structure in Statistical Translation. to appear. Devlin, J., Zbib, R., Huang, Z., Lamar, T., Schwartz, R., and Makhoul, J. (2014). Fast and robust neural network joint models for statistical machine translation. In Association for Computational Linguistics. Forcada, M. L. and ˜Neco, R. P. (1997). Recursive hetero-associative memories for translation. In J. Mira, R. Moreno-D´ıaz, and J. Cabestany, editors, Biological and Artiﬁcial Computation: From Neuroscience to Technology, volume 1240 of Lecture Notes in Computer Science, pages 453–462. Springer Berlin Heidelberg. Goodfellow, I., Warde-Farley, D., Mirza, M., Courville, A., and Bengio, Y. (2013). Maxout net- works. In Proceedings of The 30th International Conference on Machine Learning, pages 1319– 1327. Graves, A. (2012). Sequence transduction with recurrent neural networks. In Proceedings of the 29th International Conference on Machine Learning (ICML 2012). Graves, A. (2013). Generating sequences with recurrent neural networks. arXiv:1308.0850 [cs.NE]. Graves, A., Jaitly, N., and Mohamed,

---

### Rank 3 — 3_6_2 (page 6, similarity 0.577)

**Source:** NMT.pdf

length of the sentences increases. On the other hand, both RNNsearch-30 and RNNsearch-50 are more robust to the length of the sentences. RNNsearch- 50, especially, shows no performance deterioration even with sentences of length 50 or more. This superiority of the proposed model over the basic encoder–decoder is further conﬁrmed by the fact that the RNNsearch-30 even outperforms RNNencdec-50 (see Table 1). 6

---

### Rank 4 — 3_2_3 (page 2, similarity 0.573)

**Source:** NMT.pdf

units achieves close to the state-of-the-art performance of the conventional phrase-based machine translation system on an English-to-French translation task.1 Adding neural components to existing translation systems, for instance, to score the phrase pairs in the phrase table (Cho et al., 2014a) or to re-rank candidate translations (Sutskever et al., 2014), has allowed to surpass the previous state-of-the-art performance level. 2.1 RNN ENCODER–DECODER Here, we describe brieﬂy the underlying framework, called RNN Encoder–Decoder, proposed by Cho et al. (2014a) and Sutskever et al. (2014) upon which we build a novel architecture that learns to align and translate simultaneously. In the Encoder–Decoder framework, an encoder reads the input sentence, a sequence of vectors x = (x1, · · · , xTx), into a vector c.2 The most common approach is to use an RNN such that ht = f (xt, ht−1) (1) and c = q ({h1, · · · , hTx}) , where ht ∈Rn is a hidden state at time t, and c is a vector generated from the sequence of the hidden states. f and q are some nonlinear

---

### Rank 5 — 3_9_2 (page 9, similarity 0.560)

**Source:** NMT.pdf

translation, called an encoder–decoder approach, en- codes a whole input sentence into a ﬁxed-length vector from which a translation will be decoded. We conjectured that the use of a ﬁxed-length context vector is problematic for translating long sen- tences, based on a recent empirical study reported by Cho et al. (2014b) and Pouget-Abadie et al. (2014). In this paper, we proposed a novel architecture that addresses this issue. We extended the basic encoder–decoder by letting a model (soft-)search for a set of input words, or their annotations com- puted by an encoder, when generating each target word. This frees the model from having to encode a whole source sentence into a ﬁxed-length vector, and also lets the model focus only on information relevant to the generation of the next target word. This has a major positive impact on the ability of the neural machine translation system to yield good results on longer sentences. Unlike with the traditional machine translation systems, all of the pieces of the translation system, including the alignment mechanism, are jointly trained towards a better log-probability of

---

## Question 14

**Question:** What is the role of the softmax function in scaled dot-product attention?

**Original category:** answerable

### Rank 1 — 1_4_0 (page 4, similarity 0.741)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

### Rank 2 — 1_4_1 (page 4, similarity 0.716)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

### Rank 3 — 1_4_2 (page 4, similarity 0.591)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 4 — 1_5_2 (page 5, similarity 0.516)

**Source:** Attention.pdf

positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2. 3.3 Position-wise Feed-Forward Networks In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between. FFN(x) = max(0, xW1 + b1)W2 + b2 (2) While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is dmodel = 512, and the inner-layer has dimensionality dff = 2048. 3.4 Embeddings and Softmax Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens

---

### Rank 5 — 1_1_2 (page 1, similarity 0.510)

**Source:** Attention.pdf

of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research. †Work performed while at Google Brain. ‡Work performed while at Google Research. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. arXiv:1706.03762v7 [cs.CL] 2 Aug 2023

---

## Question 15

**Question:** How are query, key, and value vectors used in attention?

**Original category:** answerable

### Rank 1 — 1_4_0 (page 4, similarity 0.583)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

### Rank 2 — 1_4_2 (page 4, similarity 0.570)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 3 — 1_4_1 (page 4, similarity 0.535)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

### Rank 4 — 1_5_1 (page 5, similarity 0.532)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 5 — 1_3_1 (page 3, similarity 0.512)

**Source:** Attention.pdf

= 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i. 3.2 Attention An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum 3

---

## Question 16

**Question:** In scaled dot-product attention, are query-key dot products multiplied by the square root of the key dimension?

**Original category:** repairable

### Rank 1 — 1_4_2 (page 4, similarity 0.641)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 2 — 1_4_1 (page 4, similarity 0.554)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

### Rank 3 — 1_4_0 (page 4, similarity 0.549)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

### Rank 4 — 1_1_2 (page 1, similarity 0.389)

**Source:** Attention.pdf

of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research. †Work performed while at Google Brain. ‡Work performed while at Google Research. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. arXiv:1706.03762v7 [cs.CL] 2 Aug 2023

---

### Rank 5 — 4_3_0 (page 3, similarity 0.363)

**Source:** ODQA.pdf

At run-time, DPR applies a different encoder EQ(·) that maps the input question to a d-dimensional vector, and retrieves k passages of which vectors are the closest to the question vector. We deﬁne the similarity between the question and the passage using the dot product of their vectors: sim(q, p) = EQ(q)⊺EP (p). (1) Although more expressive model forms for measur- ing the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the represen- tations of the collection of passages can be pre- computed. Most decomposable similarity functions are some transformations of Euclidean distance (L2). For instance, cosine is equivalent to inner product for unit vectors and the Mahalanobis dis- tance is equivalent to L2 distance in a transformed space. Inner product search has been widely used and studied, as well as its connection to cosine similarity and L2 distance (Mussmann and Ermon, 2016; Ram and Gray, 2012). As our ablation study ﬁnds other similarity functions perform compara- bly (Section 5.2;

---

## Question 17

**Question:** Does the Transformer rely on recurrence to process sequences?

**Original category:** repairable

### Rank 1 — 1_2_1 (page 2, similarity 0.451)

**Source:** Attention.pdf

through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains. Attention mechanisms have become an integral part of compelling sequence modeling and transduc- tion models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs. 2 Background The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for

---

### Rank 2 — 1_2_3 (page 2, similarity 0.405)

**Source:** Attention.pdf

in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22]. End-to-end memory networks are based on a recurrent attention mechanism instead of sequence- aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34]. To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence- aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9]. 3 Model Architecture Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a

---

### Rank 3 — 1_5_1 (page 5, similarity 0.404)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 4 — 1_2_4 (page 2, similarity 0.378)

**Source:** Attention.pdf

encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next. 2

---

### Rank 5 — 1_1_0 (page 1, similarity 0.337)

**Source:** Attention.pdf

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works. Attention Is All You Need Ashish Vaswani∗ Google Brain avaswani@google.com Noam Shazeer∗ Google Brain noam@google.com Niki Parmar∗ Google Research nikip@google.com Jakob Uszkoreit∗ Google Research usz@google.com Llion Jones∗ Google Research llion@google.com Aidan N. Gomez∗† University of Toronto aidan@cs.toronto.edu Łukasz Kaiser∗ Google Brain lukaszkaiser@google.com Illia Polosukhin∗‡ illia.polosukhin@gmail.com Abstract The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English- to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT

---

## Question 18

**Question:** Does multi-head attention use only one learned projection of queries, keys, and values?

**Original category:** repairable

### Rank 1 — 1_4_2 (page 4, similarity 0.616)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 2 — 1_5_0 (page 5, similarity 0.575)

**Source:** Attention.pdf

output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2. Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this. MultiHead(Q, K, V ) = Concat(head1, ..., headh)W O where headi = Attention(QW Q i , KW K i , V W V i ) Where the projections are parameter matrices W Q i ∈Rdmodel×dk, W K i ∈Rdmodel×dk, W V i ∈Rdmodel×dv and W O ∈Rhdv×dmodel. In this work we employ h = 8 parallel attention layers, or heads. For each of these we use dk = dv = dmodel/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality. 3.2.3 Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the

---

### Rank 3 — 1_3_1 (page 3, similarity 0.525)

**Source:** Attention.pdf

= 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i. 3.2 Attention An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum 3

---

### Rank 4 — 1_5_1 (page 5, similarity 0.518)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 5 — 1_4_0 (page 4, similarity 0.500)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

## Question 19

**Question:** Does DPR encode questions and passages using exactly the same encoder parameters?

**Original category:** repairable

### Rank 1 — 4_13_2 (page 13, similarity 0.554)

**Source:** ODQA.pdf

November 1980) is a Nor- wegian stage and screen actor. He appeared in the Norwe- gian ﬁlm ”Max Manus” and played Thor Heyerdahl in the Oscar-nominated 2012 ﬁlm ”Kon-Tiki”. Pl Hagen was born in Stavanger, Norway, the son of Roar Hagen, a Norwegian cartoonist who has long been associated with Norway´s largest daily, ”VG”. He lived in Jtten, a neighborhood in the city of Stavanger in south-western Norway. . . . Table 7: Examples of passages returned from BM25 and DPR. Correct answers are written in blue and the content words in the question are written in bold. from the top 5 positive and top 30 negative passages (from the set of top 100 passages retrieved from the same question). The question encoder’s initial state is taken from a DPR model previously trained on the NQ dataset. The reader’s initial state is a BERT-base model. In terms of the end-to-end QA results, our joint-training scheme does not provide better results compared to the usual retriever/reader training pipeline, resulting in the same 39.8 exact match score on NQ dev as in

---

### Rank 2 — 4_12_3 (page 12, similarity 0.492)

**Source:** ODQA.pdf

65.0 77.2 84.5 L2 NLL 43.5 64.7 76.1 83.1 Triplet 42.2 66.0 78.1 84.9 Table 6: Retrieval Top-k accuracy on the development set of Natural Questions using different similarity and loss functions. the correct answer, presumably by matching “body of water” with semantic neighbors such as sea and channel, even though no lexical overlap exists. The second example is one where BM25 does better. The salient phrase “Thoros of Myr” is critical, and DPR is unable to capture it. D Joint Training of Retriever and Reader We ﬁx the passage encoder in our joint-training scheme while allowing only the question encoder to receive backpropagation signal from the com- bined (retriever + reader) loss function. This allows us to leverage the HNSW-based FAISS index for efﬁcient low-latency retrieving, without reindexing the passages during model updates. Our loss func- tion largely follows ORQA’s approach, which uses log probabilities of positive passages selected from the retriever model, and correct spans and passages selected from the reader model. Since the passage encoder is ﬁxed, we could use larger amount of retrieved passages when calculating

---

### Rank 3 — 4_1_4 (page 1, similarity 0.474)

**Source:** ODQA.pdf

contain the masked sentence, for additional pretraining. The question encoder and the reader model are then ﬁne- tuned using pairs of questions and answers jointly. Although ORQA successfully demonstrates that dense retrieval can outperform BM25, setting new state-of-the-art results on multiple open-domain arXiv:2004.04906v3 [cs.CL] 30 Sep 2020

---

### Rank 4 — 4_3_0 (page 3, similarity 0.464)

**Source:** ODQA.pdf

At run-time, DPR applies a different encoder EQ(·) that maps the input question to a d-dimensional vector, and retrieves k passages of which vectors are the closest to the question vector. We deﬁne the similarity between the question and the passage using the dot product of their vectors: sim(q, p) = EQ(q)⊺EP (p). (1) Although more expressive model forms for measur- ing the similarity between a question and a passage do exist, such as networks consisting of multiple layers of cross attentions, the similarity function needs to be decomposable so that the represen- tations of the collection of passages can be pre- computed. Most decomposable similarity functions are some transformations of Euclidean distance (L2). For instance, cosine is equivalent to inner product for unit vectors and the Mahalanobis dis- tance is equivalent to L2 distance in a transformed space. Inner product search has been widely used and studied, as well as its connection to cosine similarity and L2 distance (Mussmann and Ermon, 2016; Ram and Gray, 2012). As our ablation study ﬁnds other similarity functions perform compara- bly (Section 5.2;

---

### Rank 5 — 4_9_2 (page 9, similarity 0.463)

**Source:** ODQA.pdf

phrases as vectors and directly retrieve the answers to the input ques- tions efﬁciently. Using additional pretraining with the objective that matches surrogates of questions and relevant passages, Lee et al. (2019) jointly train the question encoder and reader. Their approach outperforms the BM25 plus reader paradigm on multiple open-domain QA datasets in QA accuracy, and is further extended by REALM (Guu et al., 2020), which includes tuning the passage encoder asynchronously by re-indexing the passages dur- ing training. The pretraining objective has also recently been improved by Xiong et al. (2020b). In contrast, our model provides a simple and yet effective solution that shows stronger empirical per- formance, without relying on additional pretraining or complex joint training schemes. DPR has also been used as an important mod- ule in very recent work. For instance, extending the idea of leveraging hard negatives, Xiong et al. (2020a) use the retrieval model trained in the pre- vious iteration to discover new negatives and con- struct a different set of examples in each training iteration. Starting from our trained DPR model, they show

---

## Question 20

**Question:** Does Sentence-BERT require comparing every pair of sentences with a cross-encoder at search time?

**Original category:** repairable

### Rank 1 — 2_1_2 (page 1, similarity 0.632)

**Source:** BERT.pdf

retrieval via semantic search. BERT set new state-of-the-art performance on various sentence classiﬁcation and sentence-pair regression tasks. BERT uses a cross-encoder: Two sentences are passed to the transformer network and the target value is predicted. However, this setup is unsuitable for various pair regression tasks due to too many possible combinations. Finding in a collection of n = 10 000 sentences the pair with the highest similarity requires with BERT n·(n−1)/2 = 49 995 000 inference computations. On a modern V100 GPU, this requires about 65 hours. Similar, ﬁnding which of the over 40 mil- lion existent questions of Quora is the most similar for a new question could be modeled as a pair-wise comparison with BERT, however, answering a sin- gle query would require over 50 hours. A common method to address clustering and se- mantic search is to map each sentence to a vec- tor space such that semantically similar sentences are close. Researchers have started to input indi- vidual sentences into BERT and to derive ﬁxed- size sentence embeddings. The most commonly used approach is to average

---

### Rank 2 — 2_5_4 (page 5, similarity 0.630)

**Source:** BERT.pdf

10-fold cross-validation setup gives a performance that is nearly on-par with BERT. However, in the cross-topic evaluation, we ob- serve a performance drop of SBERT by about 7 points Spearman correlation. To be considered similar, arguments should address the same claims and provide the same reasoning. BERT is able to use attention to compare directly both sentences (e.g. word-by-word comparison), while SBERT must map individual sentences from an unseen topic to a vector space such that arguments with similar claims and reasons are close. This is a much more challenging task, which appears to re- quire more than just two topics for training to work on-par with BERT. 4.4 Wikipedia Sections Distinction Dor et al. (2018) use Wikipedia to create a the- matically ﬁne-grained train, dev and test set for sentence embeddings methods. Wikipedia arti- cles are separated into distinct sections focusing on certain aspects. Dor et al. assume that sen-

---

### Rank 3 — 2_2_3 (page 2, similarity 0.606)

**Source:** BERT.pdf

BERT network structure is that no independent sentence embed- dings are computed, which makes it difﬁcult to de- rive sentence embeddings from BERT. To bypass this limitations, researchers passed single sen- tences through BERT and then derive a ﬁxed sized vector by either averaging the outputs (similar to average word embeddings) or by using the output of the special CLS token (for example: May et al. (2019); Zhang et al. (2019); Qiao et al. (2019)). These two options are also provided by the popu- lar bert-as-a-service-repository3. Up to our knowl- edge, there is so far no evaluation if these methods lead to useful sentence embeddings. Sentence embeddings are a well studied area with dozens of proposed methods. Skip-Thought (Kiros et al., 2015) trains an encoder-decoder ar- chitecture to predict the surrounding sentences. InferSent (Conneau et al., 2017) uses labeled data of the Stanford Natural Language Inference dataset (Bowman et al., 2015) and the Multi- Genre NLI dataset (Williams et al., 2018) to train a siamese BiLSTM network with max-pooling over the output. Conneau et al. showed, that InferSent consistently outperforms

---

### Rank 4 — 2_7_1 (page 7, similarity 0.589)

**Source:** BERT.pdf

well sentiment information: We observe large improvements for all sentiment tasks (MR, CR, and SST) from SentEval in comparison to InferSent and Universal Sentence Encoder. The only dataset where SBERT is signiﬁcantly worse than Universal Sentence Encoder is the TREC dataset. Universal Sentence Encoder was pre-trained on question-answering data, which ap- pears to be beneﬁcial for the question-type classi- ﬁcation task of the TREC dataset. Average BERT embeddings or using the CLS- token output from a BERT network achieved bad results for various STS tasks (Table 1), worse than average GloVe embeddings. However, for Sent- Eval, average BERT embeddings and the BERT CLS-token output achieves decent results (Ta- ble 5), outperforming average GloVe embeddings. The reason for this are the different setups. For the STS tasks, we used cosine-similarity to es- timate the similarities between sentence embed- dings. Cosine-similarity treats all dimensions equally. In contrast, SentEval ﬁts a logistic regres- sion classiﬁer to the sentence embeddings. This allows that certain dimensions can have higher or lower impact on the classiﬁcation result. We conclude that average BERT embeddings / CLS-token output

---

### Rank 5 — 2_8_4 (page 8, similarity 0.588)

**Source:** BERT.pdf

state-of-the-art sen- tence embeddings methods. Replacing BERT with RoBERTa did not yield a signiﬁcant improvement in our experiments. SBERT is computationally efﬁcient. On a GPU, it is about 9% faster than InferSent and about 55% faster than Universal Sentence Encoder. SBERT can be used for tasks which are computationally not feasible to be modeled with BERT. For exam- ple, clustering of 10,000 sentences with hierarchi- cal clustering requires with BERT about 65 hours, as around 50 Million sentence combinations must be computed. With SBERT, we were able to re- duce the effort to about 5 seconds.

---

## Question 21

**Question:** Does RAG generate answers without conditioning on retrieved documents?

**Original category:** repairable

### Rank 1 — 5_6_1 (page 6, similarity 0.650)

**Source:** RAG.pdf

21.4 40.8 44.2 to more effective marginalization over documents. Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%. 4.2 Abstractive Question Answering As shown in Table 2, RAG-Sequence outperforms BART on Open MS-MARCO NLG by 2.6 Bleu points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is impressive given that (i) those models access gold passages with speciﬁc information required to generate the reference answer , (ii) many questions are unanswerable without the gold passages, and (iii) not all questions are answerable from Wikipedia alone. Table 3 shows some generated answers from our models. Qualitatively, we ﬁnd that RAG models hallucinate less and generate factually correct text more often than BART. Later, we also show that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results,

---

### Rank 2 — 5_5_4 (page 5, similarity 0.643)

**Source:** RAG.pdf

the “closed-book” (parametric only) approaches and the performance of "open-book" retrieval-based approaches. Unlike REALM and T5+SSM, RAG enjoys strong results without expensive, specialized “salient span masking” pre-training [20]. It is worth noting that RAG’s retriever is initialized using DPR’s retriever, which uses retrieval supervision on Natural Questions and TriviaQA. RAG compares favourably to the DPR QA system, which uses a BERT-based “cross- encoder” to re-rank documents, along with an extractive reader. RAG demonstrates that neither a re-ranker nor extractive reader is necessary for state-of-the-art performance. There are several advantages to generating answers even when it is possible to extract them. Docu- ments with clues about the answer but do not contain the answer verbatim can still contribute towards a correct answer being generated, which is not possible with standard extractive approaches, leading 5

---

### Rank 3 — 5_6_2 (page 6, similarity 0.593)

**Source:** RAG.pdf

that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results, over 452 pairs of generations from BART and RAG-Token. Evaluators indicated that BART was more factual than RAG in only 7.1% of cases, while RAG was more factual in 42.7% of cases, and both RAG and BART were factual in a further 17% of cases, clearly demonstrating the effectiveness of RAG on the task over a state-of-the-art generation model. Evaluators also ﬁnd RAG generations to be more speciﬁc by a large margin. Table 3 shows typical generations from each model. Jeopardy questions often contain two separate pieces of information, and RAG-Token may perform best because it can generate responses that combine content from several documents. Figure 2 shows an example. When generating “Sun”, the posterior is high for document 2 which mentions “The Sun Also Rises”. Similarly, document 1 dominates the posterior when “A Farewell to Arms” is generated. Intriguingly,

---

### Rank 4 — 5_4_4 (page 4, similarity 0.583)

**Source:** RAG.pdf

with our NQ RAG model. We use the same train/dev/test splits as prior work [31, 26] and report Exact Match (EM) scores. For TQA, to compare with T5 [52], we also evaluate on the TQA Wiki test set. 3.2 Abstractive Question Answering RAG models can go beyond simple extractive QA and answer questions with free-form, abstractive text generation. To test RAG’s natural language generation (NLG) in a knowledge-intensive setting, we use the MSMARCO NLG task v2.1 [43]. The task consists of questions, ten gold passages retrieved from a search engine for each question, and a full sentence answer annotated from the retrieved passages. We do not use the supplied passages, only the questions and answers, to treat 4

---

### Rank 5 — 5_8_2 (page 8, similarity 0.578)

**Source:** RAG.pdf

affect performance and runtime. Figure 3 (left) shows that retrieving more documents at test time monotonically improves Open-domain QA results for RAG-Sequence, but performance peaks for RAG-Token at 10 retrieved documents. Figure 3 (right) shows that retrieving more documents leads to higher Rouge-L for RAG-Token at the expense of Bleu-1, but the effect is less pronounced for RAG-Sequence. 10 20 30 40 50 K Retrieved Docs 39 40 41 42 43 44 NQ Exact Match RAG-Tok RAG-Seq 10 20 30 40 50 K Retrieved Docs 40 50 60 70 80 NQ Answer Recall @ K RAG-Tok RAG-Seq Fixed DPR BM25 10 20 30 40 50 K Retrieved Docs 48 50 52 54 56 Bleu-1 / Rouge-L score RAG-Tok R-L RAG-Tok B-1 RAG-Seq R-L RAG-Seq B-1 Figure 3: Left: NQ performance as more documents are retrieved. Center: Retrieval recall perfor- mance in NQ. Right: MS-MARCO Bleu-1 and Rouge-L as more documents are retrieved. 5 Related Work Single-Task Retrieval Prior work has shown that retrieval improves performance across a variety of NLP tasks when considered in isolation. Such tasks include open-domain question

---

## Question 22

**Question:** Does positional encoding provide the Transformer with no information about token positions?

**Original category:** repairable

### Rank 1 — 1_6_1 (page 6, similarity 0.436)

**Source:** Attention.pdf

of the encoder and decoder stacks. The positional encodings have the same dimension dmodel as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies: PE(pos,2i) = sin(pos/100002i/dmodel) PE(pos,2i+1) = cos(pos/100002i/dmodel) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PEpos+k can be represented as a linear function of PEpos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training. 4 Why Self-Attention In this section we compare various

---

### Rank 2 — 1_5_1 (page 5, similarity 0.402)

**Source:** Attention.pdf

Applications of Attention in our Model The Transformer uses multi-head attention in three different ways: • In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9]. • The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder. • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all

---

### Rank 3 — 1_3_0 (page 3, similarity 0.386)

**Source:** Attention.pdf

Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 3.1 Encoder and Decoder Stacks Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output

---

### Rank 4 — 1_2_4 (page 2, similarity 0.368)

**Source:** Attention.pdf

encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next. 2

---

### Rank 5 — 1_9_0 (page 9, similarity 0.364)

**Source:** Attention.pdf

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities. N dmodel dff h dk dv Pdrop ϵls train PPL BLEU params steps (dev) (dev) ×106 base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65 (A) 1 512 512 5.29 24.9 4 128 128 5.00 25.5 16 32 32 4.91 25.8 32 16 16 5.01 25.4 (B) 16 5.16 25.1 58 32 5.01 25.4 60 (C) 2 6.11 23.7 36 4 5.19 25.3 50 8 4.88 25.5 80 256 32 32 5.75 24.5 28 1024 128 128 4.66 26.0 168 1024 5.12 25.4 53 4096 4.75 26.2 90 (D) 0.0 5.77 24.6 0.2 4.95 25.5 0.0 4.67 25.3 0.2 5.47 25.7 (E) positional embedding instead of sinusoids 4.92 25.7 big 6 1024 4096 16 0.3 300K 4.33 26.4 213 development set, newstest2013. We used beam search as described in the previous section,

---

## Question 23

**Question:** Does additive attention compute compatibility using a dot product alone?

**Original category:** repairable

### Rank 1 — 1_4_1 (page 4, similarity 0.654)

**Source:** Attention.pdf

as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1 √dk . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code. While for small values of dk the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of dk [3]. We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values

---

### Rank 2 — 1_4_2 (page 4, similarity 0.580)

**Source:** Attention.pdf

counteract this effect, we scale the dot products by 1 √dk . 3.2.2 Multi-Head Attention Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional 4To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk. 4

---

### Rank 3 — 1_4_0 (page 4, similarity 0.576)

**Source:** Attention.pdf

Scaled Dot-Product Attention Multi-Head Attention Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values. In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as: Attention(Q, K, V ) = softmax(QKT √dk )V (1) The two most commonly used attention functions are additive attention [2], and dot-product (multi- plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of 1

---

### Rank 4 — 1_1_2 (page 1, similarity 0.558)

**Source:** Attention.pdf

of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research. †Work performed while at Google Brain. ‡Work performed while at Google Research. 31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA. arXiv:1706.03762v7 [cs.CL] 2 Aug 2023

---

### Rank 5 — 1_9_1 (page 9, similarity 0.493)

**Source:** Attention.pdf

0.2 4.95 25.5 0.0 4.67 25.3 0.2 5.47 25.7 (E) positional embedding instead of sinusoids 4.92 25.7 big 6 1024 4096 16 0.3 300K 4.33 26.4 213 development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3. In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads. In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results

---

## Question 24

**Question:** What exact GPU temperature did the authors observe during Transformer training?

**Original category:** insufficient_evidence

### Rank 1 — 1_8_3 (page 8, similarity 0.481)

**Source:** Attention.pdf

to input length + 50, but terminate early when possible [38]. Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU 5. 6.2 Model Variations To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the 5We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively. 8

---

### Rank 2 — 1_8_0 (page 8, similarity 0.373)

**Source:** Attention.pdf

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost. Model BLEU Training Cost (FLOPs) EN-DE EN-FR EN-DE EN-FR ByteNet [18] 23.75 Deep-Att + PosUnk [39] 39.2 1.0 · 1020 GNMT + RL [38] 24.6 39.92 2.3 · 1019 1.4 · 1020 ConvS2S [9] 25.16 40.46 9.6 · 1018 1.5 · 1020 MoE [32] 26.03 40.56 2.0 · 1019 1.2 · 1020 Deep-Att + PosUnk Ensemble [39] 40.4 8.0 · 1020 GNMT + RL Ensemble [38] 26.30 41.16 1.8 · 1020 1.1 · 1021 ConvS2S Ensemble [9] 26.36 41.29 7.7 · 1019 1.2 · 1021 Transformer (base model) 27.3 38.1 3.3 · 1018 Transformer (big) 28.4 41.8 2.3 · 1019 Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop

---

### Rank 3 — 1_8_1 (page 8, similarity 0.356)

**Source:** Attention.pdf

is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 Results 6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models. On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of

---

### Rank 4 — 4_7_5 (page 7, similarity 0.336)

**Source:** ODQA.pdf

allowed, we ﬁne-tune the reader trained on Natural Questions to the target dataset. All experiments were done on eight 32GB GPUs. 6.2 Results Table 4 summarizes our ﬁnal end-to-end QA re- sults, measured by exact match with the reference answer after minor normalization as in (Chen et al., 2017; Lee et al., 2019). From the table, we can

---

### Rank 5 — 1_9_0 (page 9, similarity 0.315)

**Source:** Attention.pdf

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities. N dmodel dff h dk dv Pdrop ϵls train PPL BLEU params steps (dev) (dev) ×106 base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65 (A) 1 512 512 5.29 24.9 4 128 128 5.00 25.5 16 32 32 4.91 25.8 32 16 16 5.01 25.4 (B) 16 5.16 25.1 58 32 5.01 25.4 60 (C) 2 6.11 23.7 36 4 5.19 25.3 50 8 4.88 25.5 80 256 32 32 5.75 24.5 28 1024 128 128 4.66 26.0 168 1024 5.12 25.4 53 4096 4.75 26.2 90 (D) 0.0 5.77 24.6 0.2 4.95 25.5 0.0 4.67 25.3 0.2 5.47 25.7 (E) positional embedding instead of sinusoids 4.92 25.7 big 6 1024 4096 16 0.3 300K 4.33 26.4 213 development set, newstest2013. We used beam search as described in the previous section,

---

## Question 25

**Question:** What was the electricity cost of training the original RAG model?

**Original category:** insufficient_evidence

### Rank 1 — 5_8_0 (page 8, similarity 0.333)

**Source:** RAG.pdf

Table 4: Human assessments for the Jeopardy Question Generation Task. Factuality Speciﬁcity BART better 7.1% 16.8% RAG better 42.7% 37.4% Both good 11.7% 11.8% Both poor 17.7% 6.9% No majority 20.8% 20.1% Table 5: Ratio of distinct to total tri-grams for generation tasks. MSMARCO Jeopardy QGen Gold 89.6% 90.0% BART 70.7% 32.4% RAG-Token 77.8% 46.8% RAG-Seq. 83.5% 53.8% Table 6: Ablations on the dev set. As FEVER is a classiﬁcation task, both RAG models are equivalent. Model NQ TQA WQ CT Jeopardy-QGen MSMarco FVR-3 FVR-2 Exact Match B-1 QB-1 R-L B-1 Label Accuracy RAG-Token-BM25 29.7 41.5 32.1 33.1 17.5 22.3 55.5 48.4 75.1 91.6 RAG-Sequence-BM25 31.8 44.1 36.6 33.8 11.1 19.5 56.5 46.9 RAG-Token-Frozen 37.8 50.1 37.1 51.1 16.7 21.7 55.9 49.4 72.9 89.4 RAG-Sequence-Frozen 41.2 52.1 41.8 52.6 11.8 19.6 56.7 47.3 RAG-Token 43.5 54.8 46.5 51.9 17.9 22.6 56.2 49.4 74.5 90.6 RAG-Sequence 44.0 55.8 44.9 53.4 15.3 21.5 57.2 47.5 between these dates and use a template “Who is {position}?” (e.g. “Who is the President of Peru?”) to query our NQ RAG model with each index. RAG

---

### Rank 2 — 5_17_2 (page 17, similarity 0.331)

**Source:** RAG.pdf

is sufﬁciently fast on CPU, so we store document index vectors on CPU, requiring ∼100 GB of CPU memory for all of Wikipedia. After submission, We have ported our code to HuggingFace Transformers [66]3, which achieves equivalent performance to the previous version but is a cleaner and easier to use implementation. This version is also open-sourced. We also compress the document index using FAISS’s compression tools, reducing the CPU memory requirement to 36GB. Scripts to run experiments with RAG can be found at https://github.com/huggingface/transformers/ blob/master/examples/rag/README.md and an interactive demo of a RAG model can be found at https://huggingface.co/rag/ 2https://github.com/pytorch/fairseq 3https://github.com/huggingface/transformers 17

---

### Rank 3 — 5_4_0 (page 4, similarity 0.304)

**Source:** RAG.pdf

minimize the negative marginal log-likelihood of each target, P j −log p(yj|xj) using stochastic gradient descent with Adam [28]. Updating the document encoder BERTd during training is costly as it requires the document index to be periodically updated as REALM does during pre-training [20]. We do not ﬁnd this step necessary for strong performance, and keep the document encoder (and index) ﬁxed, only ﬁne-tuning the query encoder BERTq and the BART generator. 2.5 Decoding At test time, RAG-Sequence and RAG-Token require different ways to approximate arg maxy p(y|x). RAG-Token The RAG-Token model can be seen as a standard, autoregressive seq2seq genera- tor with transition probability: p′ θ(yi|x, y1:i−1) = P z∈top-k(p(·|x)) pη(zi|x)pθ(yi|x, zi, y1:i−1) To decode, we can plug p′ θ(yi|x, y1:i−1) into a standard beam decoder. RAG-Sequence For RAG-Sequence, the likelihood p(y|x) does not break into a conventional per- token likelihood, hence we cannot solve it with a single beam search. Instead, we run beam search for each document z, scoring each hypothesis using pθ(yi|x, z, y1:i−1). This yields a set of hypotheses Y , some of which may

---

### Rank 4 — 1_8_0 (page 8, similarity 0.287)

**Source:** Attention.pdf

Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost. Model BLEU Training Cost (FLOPs) EN-DE EN-FR EN-DE EN-FR ByteNet [18] 23.75 Deep-Att + PosUnk [39] 39.2 1.0 · 1020 GNMT + RL [38] 24.6 39.92 2.3 · 1019 1.4 · 1020 ConvS2S [9] 25.16 40.46 9.6 · 1018 1.5 · 1020 MoE [32] 26.03 40.56 2.0 · 1019 1.2 · 1020 Deep-Att + PosUnk Ensemble [39] 40.4 8.0 · 1020 GNMT + RL Ensemble [38] 26.30 41.16 1.8 · 1020 1.1 · 1021 ConvS2S Ensemble [9] 26.36 41.29 7.7 · 1019 1.2 · 1021 Transformer (base model) 27.3 38.1 3.3 · 1018 Transformer (big) 28.4 41.8 2.3 · 1019 Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop

---

### Rank 5 — 5_6_2 (page 6, similarity 0.284)

**Source:** RAG.pdf

that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results, over 452 pairs of generations from BART and RAG-Token. Evaluators indicated that BART was more factual than RAG in only 7.1% of cases, while RAG was more factual in 42.7% of cases, and both RAG and BART were factual in a further 17% of cases, clearly demonstrating the effectiveness of RAG on the task over a state-of-the-art generation model. Evaluators also ﬁnd RAG generations to be more speciﬁc by a large margin. Table 3 shows typical generations from each model. Jeopardy questions often contain two separate pieces of information, and RAG-Token may perform best because it can generate responses that combine content from several documents. Figure 2 shows an example. When generating “Sun”, the posterior is high for document 2 which mentions “The Sun Also Rises”. Similarly, document 1 dominates the posterior when “A Farewell to Arms” is generated. Intriguingly,

---

## Question 26

**Question:** What exact salary did each author receive while conducting the research?

**Original category:** insufficient_evidence

### Rank 1 — 5_7_0 (page 7, similarity 0.246)

**Source:** RAG.pdf

Document 1: his works are considered classics of American literature ... His wartime experiences formed the basis for his novel ”A Farewell to Arms” (1929) ... Document 2: ... artists of the 1920s ”Lost Generation” expatriate community. His debut novel, ”The Sun Also Rises”, was published in 1926. BOS ” The Sun Also R ises ” is a novel by this authorof ” A Fare well to Arms” Doc 1 Doc 2 Doc 3 Doc 4 Doc 5 Figure 2: RAG-Token document posterior p(zi|x, yi, y−i) for each generated token for input “Hem- ingway" for Jeopardy generation with 5 retrieved documents. The posterior for document 1 is high when generating “A Farewell to Arms" and for document 2 when generating “The Sun Also Rises". Table 3: Examples from generation tasks. RAG models generate more speciﬁc and factually accurate responses. ‘?’ indicates factually incorrect responses, * indicates partially correct responses. Task Input Model Generation MS- MARCO deﬁne middle ear BART ?The middle ear is the part of the ear between the middle ear and the nose. RAG-T The middle ear is

---

### Rank 2 — 5_7_2 (page 7, similarity 0.220)

**Source:** RAG.pdf

Dante is divided into 3 parts: the Inferno, the Purgatorio & the Purgatorio RAG-T Dante’s "Inferno" is the ﬁrst part of this epic poem RAG-S This 14th century work is divided into 3 sections: "Inferno", "Purgatorio" & "Paradiso" For 2-way classiﬁcation, we compare against Thorne and Vlachos [57], who train RoBERTa [35] to classify the claim as true or false given the gold evidence sentence. RAG achieves an accuracy within 2.7% of this model, despite being supplied with only the claim and retrieving its own evidence. We also analyze whether documents retrieved by RAG correspond to documents annotated as gold evidence in FEVER. We calculate the overlap in article titles between the top k documents retrieved by RAG and gold evidence annotations. We ﬁnd that the top retrieved document is from a gold article in 71% of cases, and a gold article is present in the top 10 retrieved articles in 90% of cases. 4.5 Additional Results Generation Diversity Section 4.3 shows that RAG models are more factual and speciﬁc than BART for Jeopardy question generation. Following recent work on

---

### Rank 3 — 4_8_4 (page 8, similarity 0.194)

**Source:** ODQA.pdf

longer passages compared to DPR (288 word pieces compared to our 100 tokens) and the computational complexity is super- linear in passage length. We also note that we found k = 50 to be optimal for NQ, and k = 10 leads to only marginal loss in exact match accu- racy (40.8 vs. 41.5 EM on NQ), which should be roughly comparable to ORQA’s 5-passage setup. 7 Related Work Passage retrieval has been an important compo- nent for open-domain QA (Voorhees, 1999). It not only effectively reduces the search space for answer extraction, but also identiﬁes the support context for users to verify the answer. Strong sparse vector space models like TF-IDF or BM25 have

---

### Rank 4 — 5_10_0 (page 10, similarity 0.182)

**Source:** RAG.pdf

Broader Impact This work offers several positive societal beneﬁts over previous work: the fact that it is more strongly grounded in real factual knowledge (in this case Wikipedia) makes it “hallucinate” less with generations that are more factual, and offers more control and interpretability. RAG could be employed in a wide variety of scenarios with direct beneﬁt to society, for example by endowing it with a medical index and asking it open-domain questions on that topic, or by helping people be more effective at their jobs. With these advantages also come potential downsides: Wikipedia, or any potential external knowledge source, will probably never be entirely factual and completely devoid of bias. Since RAG can be employed as a language model, similar concerns as for GPT-2 [50] are valid here, although arguably to a lesser extent, including that it might be used to generate abuse, faked or misleading content in the news or on social media; to impersonate others; or to automate the production of spam/phishing content [54]. Advanced language models may also lead to the automation of various jobs in

---

### Rank 5 — 5_6_1 (page 6, similarity 0.171)

**Source:** RAG.pdf

21.4 40.8 44.2 to more effective marginalization over documents. Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%. 4.2 Abstractive Question Answering As shown in Table 2, RAG-Sequence outperforms BART on Open MS-MARCO NLG by 2.6 Bleu points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is impressive given that (i) those models access gold passages with speciﬁc information required to generate the reference answer , (ii) many questions are unanswerable without the gold passages, and (iii) not all questions are answerable from Wikipedia alone. Table 3 shows some generated answers from our models. Qualitatively, we ﬁnd that RAG models hallucinate less and generate factually correct text more often than BART. Later, we also show that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results,

---

## Question 27

**Question:** What was the authors' private opinion of every competing retrieval system?

**Original category:** insufficient_evidence

### Rank 1 — 4_4_4 (page 4, similarity 0.458)

**Source:** ODQA.pdf

passages has the answer, the ques- tion will be discarded. For SQuAD and Natural Questions, since the original passages have been split and processed differently than our pool of candidate passages, we match and replace each gold passage with the corresponding passage in the candidate pool.7 We discard the questions when the matching is failed due to different Wikipedia versions or pre-processing. Table 1 shows the num- ber of questions in training/dev/test sets for all the datasets and the actual questions used for training the retriever. 5 Experiments: Passage Retrieval In this section, we evaluate the retrieval perfor- mance of our Dense Passage Retriever (DPR), along with analysis on how its output differs from 6We use the unﬁltered TriviaQA version and discard the noisy evidence documents mined from Bing. 7The improvement of using gold contexts over passages that contain answers is small. See Section 5.2 and Ap- pendix A.

---

### Rank 2 — 5_5_4 (page 5, similarity 0.442)

**Source:** RAG.pdf

the “closed-book” (parametric only) approaches and the performance of "open-book" retrieval-based approaches. Unlike REALM and T5+SSM, RAG enjoys strong results without expensive, specialized “salient span masking” pre-training [20]. It is worth noting that RAG’s retriever is initialized using DPR’s retriever, which uses retrieval supervision on Natural Questions and TriviaQA. RAG compares favourably to the DPR QA system, which uses a BERT-based “cross- encoder” to re-rank documents, along with an extractive reader. RAG demonstrates that neither a re-ranker nor extractive reader is necessary for state-of-the-art performance. There are several advantages to generating answers even when it is possible to extract them. Docu- ments with clues about the answer but do not contain the answer verbatim can still contribute towards a correct answer being generated, which is not possible with standard extractive approaches, leading 5

---

### Rank 3 — 4_10_3 (page 10, similarity 0.435)

**Source:** ODQA.pdf

TriviaQA: A large scale dis- tantly supervised challenge dataset for reading com- prehension. In Association for Computational Lin- guistics (ACL), pages 1601–1611. Omar Khattab and Matei Zaharia. 2020. ColBERT: Efﬁcient and effective passage search via contextu- alized late interaction over BERT. In ACM SIGIR Conference on Research and Development in Infor- mation Retrieval (SIGIR), pages 39–48. Brian Kulis. 2013. Metric learning: A survey. Foun- dations and Trends in Machine Learning, 5(4):287– 364. Tom Kwiatkowski, Jennimaria Palomaki, Olivia Red- ﬁeld, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Matthew Kelcey, Jacob Devlin, Kenton Lee, Kristina N. Toutanova, Llion Jones, Ming-Wei Chang, Andrew Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natu- ral questions: a benchmark for question answering research. Transactions of the Association of Compu- tational Linguistics (TACL). Kenton Lee, Ming-Wei Chang, and Kristina Toutanova. 2019. Latent retrieval for weakly supervised open domain question answering. In Association for Com- putational Linguistics (ACL), pages 6086–6096. Mike Lewis, Yinhan Liu, Naman Goyal, Mar- jan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Veselin Stoyanov, and Luke Zettlemoyer. 2020a. BART: Denoising sequence-to-sequence pre- training

---

### Rank 4 — 5_7_3 (page 7, similarity 0.418)

**Source:** RAG.pdf

gold article is present in the top 10 retrieved articles in 90% of cases. 4.5 Additional Results Generation Diversity Section 4.3 shows that RAG models are more factual and speciﬁc than BART for Jeopardy question generation. Following recent work on diversity-promoting decoding [33, 59, 39], we also investigate generation diversity by calculating the ratio of distinct ngrams to total ngrams generated by different models. Table 5 shows that RAG-Sequence’s generations are more diverse than RAG-Token’s, and both are signiﬁcantly more diverse than BART without needing any diversity-promoting decoding. Retrieval Ablations A key feature of RAG is learning to retrieve relevant information for the task. To assess the effectiveness of the retrieval mechanism, we run ablations where we freeze the retriever during training. As shown in Table 6, learned retrieval improves results for all tasks. We compare RAG’s dense retriever to a word overlap-based BM25 retriever [53]. Here, we replace RAG’s retriever with a ﬁxed BM25 system, and use BM25 retrieval scores as logits when calculating p(z|x). Table 6 shows the results. For FEVER, BM25 performs best, perhaps since FEVER claims

---

### Rank 5 — 5_6_1 (page 6, similarity 0.407)

**Source:** RAG.pdf

21.4 40.8 44.2 to more effective marginalization over documents. Furthermore, RAG can generate correct answers even when the correct answer is not in any retrieved document, achieving 11.8% accuracy in such cases for NQ, where an extractive model would score 0%. 4.2 Abstractive Question Answering As shown in Table 2, RAG-Sequence outperforms BART on Open MS-MARCO NLG by 2.6 Bleu points and 2.6 Rouge-L points. RAG approaches state-of-the-art model performance, which is impressive given that (i) those models access gold passages with speciﬁc information required to generate the reference answer , (ii) many questions are unanswerable without the gold passages, and (iii) not all questions are answerable from Wikipedia alone. Table 3 shows some generated answers from our models. Qualitatively, we ﬁnd that RAG models hallucinate less and generate factually correct text more often than BART. Later, we also show that RAG generations are more diverse than BART generations (see §4.5). 4.3 Jeopardy Question Generation Table 2 shows that RAG-Token performs better than RAG-Sequence on Jeopardy question generation, with both models outperforming BART on Q-BLEU-1. 4 shows human evaluation results,

---

## Question 28

**Question:** What was the precise network latency of every experiment in the DPR paper?

**Original category:** insufficient_evidence

### Rank 1 — 4_8_3 (page 8, similarity 0.394)

**Source:** ODQA.pdf

tions where the retriever and reader are jointly trained, following Lee et al. (2019). This approach obtains a score of 39.8 EM, which suggests that our strategy of training a strong retriever and reader in isolation can leverage effectively available supervi- sion, while outperforming a comparable joint train- ing approach with a simpler design (Appendix D). One thing worth noticing is that our reader does consider more passages compared to ORQA, al- though it is not completely clear how much more time it takes for inference. While DPR processes up to 100 passages for each question, the reader is able to ﬁt all of them into one batch on a sin- gle 32GB GPU, thus the latency remains almost identical to the single passage case (around 20ms). The exact impact on throughput is harder to mea- sure: ORQA uses 2-3x longer passages compared to DPR (288 word pieces compared to our 100 tokens) and the computational complexity is super- linear in passage length. We also note that we found k = 50 to be optimal for NQ, and k =

---

### Rank 2 — 3_14_0 (page 14, similarity 0.277)

**Source:** NMT.pdf

Published as a conference paper at ICLR 2015 Model Updates (×105) Epochs Hours GPU Train NLL Dev. NLL RNNenc-30 8.46 6.4 109 TITAN BLACK 28.1 53.0 RNNenc-50 6.00 4.5 108 Quadro K-6000 44.0 43.6 RNNsearch-30 4.71 3.6 113 TITAN BLACK 26.7 47.2 RNNsearch-50 2.88 2.2 111 Quadro K-6000 40.7 38.1 RNNsearch-50⋆ 6.67 5.0 252 Quadro K-6000 36.7 35.2 Table 2: Learning statistics and relevant information. Each update corresponds to updating the parameters once using a single minibatch. One epoch is one pass through the training set. NLL is the average conditional log-probabilities of the sentences in either the training set or the development set. Note that the lengths of the sentences differ. where αij = exp (eij) PTx k=1 exp (eik) eij =v⊤ a tanh (Wasi−1 + Uahj) , and hj is the j-th annotation in the source sentence (see Eq. (7)). va ∈Rn′, Wa ∈Rn′×n and Ua ∈Rn′×2n are weight matrices. Note that the model becomes RNN Encoder–Decoder (Cho et al., 2014a), if we ﬁx ci to −→h Tx. With the decoder state si−1, the context ci and the

---

### Rank 3 — 4_5_4 (page 5, similarity 0.271)

**Source:** ODQA.pdf

DPR with BM25 in both single- and multi-dataset settings. We conjecture that the lower performance on SQuAD is due to two reasons. First, the annota- tors wrote questions after seeing the passage. As a result, there is a high lexical overlap between passages and questions, which gives BM25 a clear advantage. Second, the data was collected from only 500+ Wikipedia articles and thus the distribu- tion of training examples is extremely biased, as argued previously by Lee et al. (2019). 5.2 Ablation Study on Model Training To understand further how different model training options affect the results, we conduct several addi- tional experiments and discuss our ﬁndings below.

---

### Rank 4 — 1_9_1 (page 9, similarity 0.258)

**Source:** Attention.pdf

0.2 4.95 25.5 0.0 4.67 25.3 0.2 5.47 25.7 (E) positional embedding instead of sinusoids 4.92 25.7 big 6 1024 4096 16 0.3 300K 4.33 26.4 213 development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3. In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads. In Table 3 rows (B), we observe that reducing the attention key size dk hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results

---

### Rank 5 — 4_8_1 (page 8, similarity 0.253)

**Source:** ODQA.pdf

model but pretrained on Wikipedia and CC-News, respectively. Single and Multi denote that our Dense Passage Retriever (DPR) is trained using individual or combined training datasets (all except SQuAD). For WQ and TREC in the Multi setting, we ﬁne-tune the reader trained on NQ. see that higher retriever accuracy typically leads to better ﬁnal QA results: in all cases except SQuAD, answers extracted from the passages retrieved by DPR are more likely to be correct, compared to those from BM25. For large datasets like NQ and TriviaQA, models trained using multiple datasets (Multi) perform comparably to those trained using the individual training set (Single). Conversely, on smaller datasets like WQ and TREC, the multi- dataset setting has a clear advantage. Overall, our DPR-based models outperform the previous state- of-the-art results on four out of the ﬁve datasets, with 1% to 12% absolute differences in exact match accuracy. It is interesting to contrast our results to those of ORQA (Lee et al., 2019) and also the concurrently developed approach, REALM (Guu et al., 2020). While both methods include addi- tional pretraining

---

## Question 29

**Question:** What exact hardware temperature was recorded during Sentence-BERT experiments?

**Original category:** insufficient_evidence

### Rank 1 — 2_4_0 (page 4, similarity 0.397)

**Source:** BERT.pdf

Model STS12 STS13 STS14 STS15 STS16 STSb SICK-R Avg. Avg. GloVe embeddings 55.14 70.66 59.73 68.25 63.66 58.02 53.76 61.32 Avg. BERT embeddings 38.78 57.98 57.98 63.15 61.06 46.35 58.40 54.81 BERT CLS-vector 20.16 30.01 20.09 36.88 38.08 16.50 42.63 29.19 InferSent - Glove 52.86 66.75 62.15 72.77 66.87 68.03 65.65 65.01 Universal Sentence Encoder 64.49 67.80 64.61 76.83 73.18 74.92 76.69 71.22 SBERT-NLI-base 70.97 76.53 73.19 79.09 74.30 77.03 72.91 74.89 SBERT-NLI-large 72.27 78.46 74.90 80.99 76.25 79.23 73.75 76.55 SRoBERTa-NLI-base 71.54 72.49 70.80 78.74 73.69 77.77 74.46 74.21 SRoBERTa-NLI-large 74.53 77.00 73.18 81.85 76.82 79.10 74.29 76.68 Table 1: Spearman rank correlation ρ between the cosine similarity of sentence representations and the gold labels for various Textual Similarity (STS) tasks. Performance is reported by convention as ρ × 100. STS12-STS16: SemEval 2012-2016, STSb: STSbenchmark, SICK-R: SICK relatedness dataset. (Williams et al., 2018) dataset. The SNLI is a col- lection of 570,000 sentence pairs annotated with the labels contradiction, eintailment, and neu- tral. MultiNLI contains 430,000 sentence pairs and covers a range of genres of spoken and written text.

---

### Rank 2 — 2_8_4 (page 8, similarity 0.382)

**Source:** BERT.pdf

state-of-the-art sen- tence embeddings methods. Replacing BERT with RoBERTa did not yield a signiﬁcant improvement in our experiments. SBERT is computationally efﬁcient. On a GPU, it is about 9% faster than InferSent and about 55% faster than Universal Sentence Encoder. SBERT can be used for tasks which are computationally not feasible to be modeled with BERT. For exam- ple, clustering of 10,000 sentences with hierarchi- cal clustering requires with BERT about 65 hours, as around 50 Million sentence combinations must be computed. With SBERT, we were able to re- duce the effort to about 5 seconds.

---

### Rank 3 — 2_7_0 (page 7, similarity 0.374)

**Source:** BERT.pdf

Model MR CR SUBJ MPQA SST TREC MRPC Avg. Avg. GloVe embeddings 77.25 78.30 91.17 87.85 80.18 83.0 72.87 81.52 Avg. fast-text embeddings 77.96 79.23 91.68 87.81 82.15 83.6 74.49 82.42 Avg. BERT embeddings 78.66 86.25 94.37 88.66 84.40 92.8 69.45 84.94 BERT CLS-vector 78.68 84.85 94.21 88.23 84.13 91.4 71.13 84.66 InferSent - GloVe 81.57 86.54 92.50 90.38 84.18 88.2 75.77 85.59 Universal Sentence Encoder 80.09 85.19 93.98 86.70 86.38 93.2 70.14 85.10 SBERT-NLI-base 83.64 89.43 94.39 89.86 88.96 89.6 76.00 87.41 SBERT-NLI-large 84.88 90.07 94.52 90.33 90.66 87.4 75.94 87.69 Table 5: Evaluation of SBERT sentence embeddings using the SentEval toolkit. SentEval evaluates sentence embeddings on different sentence classiﬁcation tasks by training a logistic regression classiﬁer using the sentence embeddings as features. Scores are based on a 10-fold cross-validation. It appears that the sentence embeddings from SBERT capture well sentiment information: We observe large improvements for all sentiment tasks (MR, CR, and SST) from SentEval in comparison to InferSent and Universal Sentence Encoder. The only dataset where SBERT is signiﬁcantly worse than Universal Sentence Encoder is the TREC

---

### Rank 4 — 2_5_0 (page 5, similarity 0.365)

**Source:** BERT.pdf

sion method for the output. Model Spearman Not trained for STS Avg. GloVe embeddings 58.02 Avg. BERT embeddings 46.35 InferSent - GloVe 68.03 Universal Sentence Encoder 74.92 SBERT-NLI-base 77.03 SBERT-NLI-large 79.23 Trained on STS benchmark dataset BERT-STSb-base 84.30 ± 0.76 SBERT-STSb-base 84.67 ± 0.19 SRoBERTa-STSb-base 84.92 ± 0.34 BERT-STSb-large 85.64 ± 0.81 SBERT-STSb-large 84.45 ± 0.43 SRoBERTa-STSb-large 85.02 ± 0.76 Trained on NLI data + STS benchmark data BERT-NLI-STSb-base 88.33 ± 0.19 SBERT-NLI-STSb-base 85.35 ± 0.17 SRoBERTa-NLI-STSb-base 84.79 ± 0.38 BERT-NLI-STSb-large 88.77 ± 0.46 SBERT-NLI-STSb-large 86.10 ± 0.13 SRoBERTa-NLI-STSb-large 86.15 ± 0.35 Table 2: Evaluation on the STS benchmark test set. BERT systems were trained with 10 random seeds and 4 epochs. SBERT was ﬁne-tuned on the STSb dataset, SBERT-NLI was pretrained on the NLI datasets, then ﬁne-tuned on the STSb dataset. We use the training set to ﬁne-tune SBERT us- ing the regression objective function. At predic- tion time, we compute the cosine-similarity be- tween the sentence embeddings. All systems are trained with 10 random seeds to counter variances (Reimers and Gurevych, 2018). The results are depicted in Table

---

### Rank 5 — 2_2_3 (page 2, similarity 0.364)

**Source:** BERT.pdf

BERT network structure is that no independent sentence embed- dings are computed, which makes it difﬁcult to de- rive sentence embeddings from BERT. To bypass this limitations, researchers passed single sen- tences through BERT and then derive a ﬁxed sized vector by either averaging the outputs (similar to average word embeddings) or by using the output of the special CLS token (for example: May et al. (2019); Zhang et al. (2019); Qiao et al. (2019)). These two options are also provided by the popu- lar bert-as-a-service-repository3. Up to our knowl- edge, there is so far no evaluation if these methods lead to useful sentence embeddings. Sentence embeddings are a well studied area with dozens of proposed methods. Skip-Thought (Kiros et al., 2015) trains an encoder-decoder ar- chitecture to predict the surrounding sentences. InferSent (Conneau et al., 2017) uses labeled data of the Stanford Natural Language Inference dataset (Bowman et al., 2015) and the Multi- Genre NLI dataset (Williams et al., 2018) to train a siamese BiLSTM network with max-pooling over the output. Conneau et al. showed, that InferSent consistently outperforms

---

## Question 30

**Question:** What private discussions led the authors to select their final model architectures?

**Original category:** insufficient_evidence

### Rank 1 — 5_14_3 (page 14, similarity 0.279)

**Source:** RAG.pdf

social impacts of language models. ArXiv, abs/1908.09203, 2019. [55] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory net- works. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates, Inc., 2015. URL http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf. 14

---

### Rank 2 — 1_3_0 (page 3, similarity 0.259)

**Source:** Attention.pdf

Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 3.1 Encoder and Decoder Stacks Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- wise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output

---

### Rank 3 — 1_8_1 (page 8, similarity 0.229)

**Source:** Attention.pdf

is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 Results 6.1 Machine Translation On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models. On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of

---

### Rank 4 — 1_9_0 (page 9, similarity 0.228)

**Source:** Attention.pdf

Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities. N dmodel dff h dk dv Pdrop ϵls train PPL BLEU params steps (dev) (dev) ×106 base 6 512 2048 8 64 64 0.1 0.1 100K 4.92 25.8 65 (A) 1 512 512 5.29 24.9 4 128 128 5.00 25.5 16 32 32 4.91 25.8 32 16 16 5.01 25.4 (B) 16 5.16 25.1 58 32 5.01 25.4 60 (C) 2 6.11 23.7 36 4 5.19 25.3 50 8 4.88 25.5 80 256 32 32 5.75 24.5 28 1024 128 128 4.66 26.0 168 1024 5.12 25.4 53 4096 4.75 26.2 90 (D) 0.0 5.77 24.6 0.2 4.95 25.5 0.0 4.67 25.3 0.2 5.47 25.7 (E) positional embedding instead of sinusoids 4.92 25.7 big 6 1024 4096 16 0.3 300K 4.33 26.4 213 development set, newstest2013. We used beam search as described in the previous section,

---

### Rank 5 — 5_11_0 (page 11, similarity 0.228)

**Source:** RAG.pdf

[7] Christopher Clark and Matt Gardner. Simple and Effective Multi-Paragraph Reading Compre- hension. arXiv:1710.10723 [cs], October 2017. URL http://arxiv.org/abs/1710.10723. arXiv: 1710.10723. [8] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Con- ference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://www.aclweb.org/anthology/N19-1423. [9] Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. Wiz- ard of wikipedia: Knowledge-powered conversational agents. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=r1l73iRqKm. [10] Matthew Dunn, Levent Sagun, Mike Higgins, V. Ugur Guney, Volkan Cirik, and Kyunghyun Cho. SearchQA: A New Q&A Dataset Augmented with Context from a Search Engine. arXiv:1704.05179 [cs], April 2017. URL http://arxiv.org/abs/1704.05179. arXiv: 1704.05179. [11] Angela Fan, Mike Lewis, and Yann Dauphin. Hierarchical neural story generation. In Proceed- ings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 889–898, Melbourne, Australia,

---

