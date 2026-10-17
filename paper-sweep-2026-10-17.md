# Paper sweep: looped/recurrent-depth reasoning + basin/dynamics mechinterp
Swept 2026-10-17 via arXiv API, 16 query families. Curated hits, newest first. Non-typical-LLM items (diffusion LMs, vision, ensemble-generic) removed per request.

## [2609.40316v1] Scaling Laws for Looped Mixture of Experts
date: 2026-09-30 | authors: Yanbei Chen, Anirudh Goyal, Raghuraman Krishnamoorthi
Looped transformers and Mixture-of-Experts (MoE) offer complementary routes to efficient scaling: recurrence increases computational depth at fixed parameters, while MoE sparsity expands total capacity at fixed active compute. Yet existing scaling laws model recurrence or sparsity in isolation. In this work, we introduce Loop Scaling Laws, the first scaling law to jointly model recurrence and sparsity alongside model size and data. At its core is a bounded, sparsity-conditional recurrence mapping that characterizes the effective-parameter gain from looping and how sparsity raises this gain. The laws predict the held-out loss of looped models more accurately than prior alternatives, and recov

## [2609.37717v1] Predictive Geometry of Hidden Trajectories in Transformers
date: 2026-09-29 | authors: Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, Radu State
Decoder-only transformers are trained only through a terminal next-token prediction loss, yet this loss constrains every intermediate hidden state through the fixed downstream computation. We formalize this constraint by studying layerwise loss-to-go functions: the terminal loss obtained by continuing a candidate hidden state through the remaining transformer blocks. Around successful validation trajectories, we show that the local second-order geometry of these functions is governed, up to low-loss residual terms, by a pullback Fisher operator on hidden-state space. Its spectrum identifies output-sensitive directions and approximately prediction-null directions, yielding a local observable 

## [2609.21662v1] When Steering Fails in Latent Reasoning: A Latent-to-Language Transition Gap
date: 2026-09-18 | authors: Gaoxiang Huang, Lei Qi
Activation steering has become a widely used approach for controlling language models during explicit chain-of-thought (CoT) reasoning, motivating its extension to latent CoT. However, we find that steering continuous thoughts produces substantially weaker effects on subsequent language generation than steering explicit CoT, even when the hidden representations are moved by comparable amounts. We first show that task information remains identifiable in continuous thoughts. Hence, we hypothesize a \textbf{latent-to-language transition gap}, in which an intervention effect in latent space fails to transfer to language generation. Two further results support this hypothesis: the output distribu

## [2607.29062v1] On the Generalization of Steering Vectors for Chain-of-Thought Faithfulness
date: 2026-07-31 | authors: Matthew Nguyen, Kyle Cox, Austin Meek, Iván Arcuschin
Model capabilities have improved in large part due to scaling chain of thought. This has been a promising development for AI safety--where models verbalize their reasoning, it is possible to monitor it. However, in some cases, models do not verbalize important steps in their reasoning process. For example, models prompted with a cue suggesting the incorrect answer may fail to acknowledge that cue, even when it appears instrumental to their conclusion. When chain of thought (CoT) fails to disclose instrumental reasoning steps, we describe it as unfaithful. Prior work has shown that activation steering can be a useful method to improve faithfulness in CoT. We extend this line of work by studyi

## [2607.13491v2] DeepLoop: Depth Scaling for Looped Transformers
date: 2026-07-15 | authors: Shuzhen Li, Yifan Zhang, Jiacheng Guo, Quanquan Gu
Looped Transformers scale sequential computation by applying a compact stack of physical blocks for multiple rounds, increasing unrolled depth without increasing stored parameters. This reuse changes the residual-scaling problem: in an untied Transformer, each residual branch receives and applies its own parameter update, whereas in a looped Transformer one shared update aggregates gradients from 

## [2606.24543v1] Reasoning as Attractor Dynamics: Latent Memory Retrieval via Gibbs-Weighted Energy Minimization
date: 2026-06-23 | authors: Kanishk Awadhiya
Large Language Models (LLMs) are traditionally viewed as autoregressive generators. However, from the perspective of collective computation, they function as high-dimensional Dense Associative Memories that store complex reasoning patterns as latent attractors. In this work, we investigate the energy landscape of mathematical reasoning. We posit that correct reasoning chains correspond to deep, wi

## [2607.09698v1] Interpreting Latent CoT Reasoning as Dynamical Systems
date: 2026-06-20 | authors: Sabari Iyyappan Duraipandian, Shreya Sanjay Boyane, Manju Nagesh, Jerome Francis
Recent latent reasoning methods, such as CODI and COCONUT, face a fundamental interpretability problem: they maintain multiple superimposed candidate traces in the hidden space at each step, unlike explicit- CoT, which follows a single transparent reasoning trace. Existing mechanistic methods show compression, shortcuts, and superposition without explaining how reasoning evolves across latent step

## [2606.01243v1] Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention
date: 2026-05-31 | authors: Shuochen Chang, Tong Bai, Xiaofeng Zhang, Qianli Ma
Latent reasoning enables Large Language Models (LLMs) to perform multi-step inference within continuous hidden states, offering efficiency gains over explicit Chain-of-Thought (CoT). However, the opacity of these continuous thought vectors hinders their reliability and controllability. This paper bridges the gap between mechanistic interpretability and actionable control. We first present a system

## [2605.26733v1] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models
date: 2026-05-26 | authors: Xiao-Wen Yang, Ziyu Han, Xi-Hua Zhang, Wen-Da Wei
Looped Language Models (LoopLMs) enable efficient latent reasoning through depth recurrence, yet exhibit unreliable test-time scaling behavior: performance often peaks at a certain iteration depth and then collapses with further recurrence. Through latent dynamics analysis, we find an inherent trade-off between stability and effectiveness in existing architectures and strategies. By conceptualizing reasoning as uncertainty reduction, we propose that convergence toward stable fixed points while preserving effectiveness represents a promising way. To this end, we propose STARS (STAbility-driven Recurrent Scaling), a training framework that constrains latent states to approach asymptotically st

## [2605.21488v1] Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning
date: 2026-05-20 | authors: Benhao Huang, Zhengyang Geng, Zico Kolter
Scaling test-time compute by iteratively updating a latent state has emerged as a powerful paradigm for reasoning. Yet the internal mechanisms that enable these iterative models to generalize beyond memorized patterns remain unclear. We hypothesize that generalizable reasoning arises from learning task-conditioned attractors: latent dynamical systems whose stable fixed points correspond to valid s

## [2605.15454v2] Reasoning Models Don't Just Think Longer, They Move Differently
date: 2026-05-14 | authors: Anders Gjølbye, Lars Kai Hansen, Sanmi Koyejo
Reasoning-trained language models often spend more tokens on harder problems, but longer chains of thought do not show whether a model is merely computing for more steps or following a different internal trajectory. We study this distinction through hidden-state trajectories during chain-of-thought generation across competitive programming, mathematics, and Boolean satisfiability. Raw trajectory geometry is strongly shaped by generation length: longer generations mechanically alter path statistics, so difficulty-dependent comparisons are misleading without adjustment. After residualizing trajectory statistics on length, difficulty remains systematically coupled to corrected trajectory geomet

## [2605.12466v1] Solve the Loop: Attractor Models for Language and Reasoning
date: 2026-05-12 | authors: Jacob Fein-Ashley, Paria Rashidinejad
Looped Transformers offer a promising alternative to purely feed-forward computation by iteratively refining latent representations, improving language modeling and reasoning. Yet recurrent architectures remain unstable to train, costly to optimize and deploy, and constrained to small, fixed recurrence depths. We introduce Attractor Models, in which a backbone module first proposes output embeddin

## [2605.01853v1] Spatiotemporal Hidden-State Dynamics as a Signature of Internal Reasoning in Large Language Models
date: 2026-05-03 | authors: Kotaro Furuya, Takahito Tanimura
Large reasoning models (LRMs) generate extended solutions, yet it remains unclear whether these traces reflect substantive internal computation or merely verbosity and overthinking. Although recent hidden-state analyses suggest that internal representations carry correctness-related signals, their coarse aggregations may obscure the token and layer structure underlying reasoning computation. We investigate hidden-state transitions across decoding steps and layers, and identify a distinct spatiotemporal pattern in LRMs: successful trajectories exhibit broad temporal dynamics with localized layer-wise concentration, while this structure is weaker in non-reasoning models and knowledge-heavy dom

## [2604.21999v3] Universal Transformers Need Memory: Depth-State Trade-offs in Adaptive Recursive Reasoning
date: 2026-04-23 | authors: Grigory Sapunov
We study learned memory tokens as a computational scratchpad for a single-block Universal Transformer with Adaptive Computation Time (ACT) on Sudoku-Extreme, a combinatorial reasoning benchmark. Memory tokens are empirically necessary: no configuration without them reaches non-trivial performance. The optimal count has a sharp lower threshold (T=0 always fails, T=8 reliably succeeds) followed by a

## [2604.07822v2] Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers
date: 2026-04-09 | authors: Harsh Kohli, Srinivasan Parthasarathy, Huan Sun, Yuekun Yao
We study implicit reasoning, i.e. the ability to combine knowledge or rules within a single forward pass. While transformer-based large language models store substantial factual knowledge and rules, they often fail to compose this knowledge for implicit multi-hop reasoning, suggesting a lack of compositional generalization over their parametric knowledge. To address this limitation, we study recur

## [2603.10544v1] SCORE: Replacing Layer Stacking with Contractive Recurrent Depth
date: 2026-03-11 | authors: Guillaume Godin
Residual connections are central to modern deep neural networks, enabling stable optimization and efficient information flow across depth. In this work, we propose SCORE (Skip-Connection ODE Recurrent Embedding), a discrete recurrent alternative to classical layer stacking. Instead of composing multiple independent layers, SCORE iteratively applies a single shared neural block using an ODE (Ordina

## [2601.10962v2] Noise-Driven Exploration and Transient Freezing Select Flat Minima in Stochastic Gradient Descent
date: 2026-01-16 | authors: Ning Yang, Yikuan Zhang, Qi Ouyang, Chao Tang
Stochastic gradient descent (SGD) is central to deep learning, yet the dynamical origin of its preference for flatter, more generalizable solutions remains unclear. Here, by analyzing SGD learning dynamics, we identify a nonequilibrium mechanism that governs solution selection during training. Numerical experiments reveal a transient exploratory phase in which SGD trajectories repeatedly escape sh

## [2601.08846v1] Directional Attractors in LLM Reasoning: How Similarity Retrieval Steers Iterative Summarization Based Reasoning
date: 2025-12-22 | authors: Cagatay Tekin, Charbel Barakat, Luis Joseph Luna Limgenco
Iterative summarization based reasoning frameworks such as InftyThink enable long-horizon reasoning in large language models (LLMs) by controlling context growth, but they repeatedly regenerate similar reasoning strategies across tasks. We introduce InftyThink with Cross-Chain Memory, an extension that augments iterative reasoning with an embedding-based semantic cache of previously successful rea

## [2512.07558v2] ReLaX: Reasoning with Latent Exploration for Large Reasoning Models
date: 2025-12-08 | authors: Shimin Zhang, Xianwei Chen, Yufan Shen, Ziyuan Ye
Reinforcement Learning with Verifiable Rewards (RLVR) has recently demonstrated remarkable potential in enhancing the reasoning capability of Large Reasoning Models (LRMs). However, RLVR often drives the policy toward over-determinism, resulting in ineffective exploration and premature policy convergence. While promoting token-level diversity has shown promise in mitigating entropy collapse, we argue that the latent dynamics underlying token generation encode a far richer computational structure for steering policy optimization toward a more effective exploration-exploitation tradeoff. To enable tractable analysis and intervention of the latent dynamics of LRMs, we leverage Koopman operator 

## [2511.07384v1] Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence
date: 2025-11-10 | authors: Sean McLeish, Ang Li, John Kirchenbauer, Dayal Singh Kalra
Recent advances in depth-recurrent language models show that recurrence can decouple train-time compute and parameter count from test-time compute. In this work, we study how to convert existing pretrained non-recurrent language models into depth-recurrent models. We find that using a curriculum of recurrences to increase the effective depth of the model over the course of training preserves performance while reducing total computational cost. In our experiments, on mathematics, we observe that converting pretrained models to recurrent ones results in better performance at a given compute budget than simply post-training the original non-recurrent language model.

## [2510.19875v2] Stream: Scaling up Mechanistic Interpretability to Long Context in LLMs via Sparse Attention
date: 2025-10-22 | authors: J Rosser, José Luis Redondo García, Gustavo Penha, Konstantina Palla
As Large Language Models (LLMs) scale to million-token contexts, traditional Mechanistic Interpretability techniques for analyzing attention scale quadratically with context length, demanding terabytes of memory beyond 100,000 tokens. We introduce Sparse Tracing, a novel technique that leverages dynamic sparse attention to efficiently analyze long context attention patterns. We present Stream, a c

## [2509.23314v2] Two-Scale Latent Dynamics for Recurrent-Depth Transformers
date: 2025-09-27 | authors: Francesco Pappone, Donato Crisostomi, Emanuele Rodolà
Recurrent-depth transformers scale test-time compute by iterating latent computations before emitting tokens. We study the geometry of these iterates and argue for a simple, two-scale operational picture: (i) within a looped block, updates act as small-scale refinements; (ii) across consecutive blocks, states undergo a larger-scale drift. Across training, our measurements show that loop steps beco

## [2507.20758v3] How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation
date: 2025-07-28 | authors: Hao Yang, Qinghua Zhao, Lei Li, Lingyi Meng
Chain-of-Thought (CoT) prompting significantly enhances model reasoning, yet its internal mechanisms remain poorly understood. We analyze CoT's operational principles by reversely tracing information flow across decoding, projection, and activation phases. Our quantitative analysis suggests that CoT may serve as a decoding space pruner, leveraging answer templates to guide output generation, with 

## [2506.04374v1] A Statistical Physics of Language Model Reasoning
date: 2025-06-04 | authors: Jack David Carson, Amir Reisizadeh
Transformer LMs show emergent reasoning that resists mechanistic understanding. We offer a statistical physics framework for continuous-time chain-of-thought reasoning dynamics. We model sentence-level hidden state trajectories as a stochastic dynamical system on a lower-dimensional manifold. This drift-diffusion system uses latent regime switching to capture diverse reasoning phases, including misaligned states or failures. Empirical trajectories (8 models, 7 benchmarks) show a rank-40 projection (balancing variance capture and feasibility) explains ~50% variance. We find four latent reasoning regimes. An SLDS model is formulated and validated to capture these features. The framework enable

## [2505.17646v3] Unveiling the Basin-Like Loss Landscape in Large Language Models
date: 2025-05-23 | authors: Huanran Chen, Yinpeng Dong, Zeming Wei, Yao Huang
We discover the emergence of \textit{basins} in the loss landscape of large language models. As model scale increases, LLMs become progressively more resilient to random perturbations in the parameter space, giving rise to expansive stability regions where models exhibit nearly identical performance, but outside of which their capabilities collapse. We observe that pre-training creates a \textit{b

## [2502.18147v2] Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations
date: 2025-02-25 | authors: Lucy Farnik, Tim Lawson, Conor Houghton, Laurence Aitchison
Sparse autoencoders (SAEs) have been successfully used to discover sparse and human-interpretable representations of the latent activations of LLMs. However, we would ultimately like to understand the computations performed by LLMs and not just their representations. The extent to which SAEs can help us understand computations is unclear because they are not designed to "sparsify" computations in any sense, only latent activations. To solve this, we propose Jacobian SAEs (JSAEs), which yield not only sparsity in the input and output activations of a given model component but also sparsity in the computation (formally, the Jacobian) connecting them. With a naïve implementation, the Jacobians 

# Part 2: direction for the lab's own exps (steering/extraction-specific)

## [2602.17881] Understanding Unreliability of Steering Vectors in Language Models: Geometric Predictors and the Limits of Linear Approximations
date: 2026-02-19
Steering vectors are a lightweight method for controlling language model behavior by adding a learned bias to the activations at inference time. Although effective on average, steering effect sizes vary across samples and are unreliable for many target behaviors. In my thesis, I investigate why steering reliability differs across behaviors and how it is impacted by steering vector training data. First, I find that higher cosine similarity between training activation differences predicts more reliable steering. Second, I observe that behavior datasets where positive and negative activations are better separated along the steering direction are more reliably steerable. Finally, steering vector

## [2608.05578] Detecting Safety Training Modification in Language Models via Activation Analysis
date: 2026-08-06
We introduce AMS (Activation-based Model Scanner), a tool that detects modifications to safety training in language models by measuring the geometric structure of safety-relevant concepts in activation space. Safety training creates measurable separation between harmful and benign content classes; certain safety modifications collapse or rotate this structure, while others leave it intact. We validate AMS across 14 model configurations spanning 4 architecture families (Llama, Gemma, Qwen, Mistral) and four safety-modification categories (instruction-tuned, base, abliterated, uncensored fine-tunes). Leave-one-out cross-validation of thresholds achieves 71% accuracy (10/14); bootstrap 95% conf

## [2603.22061] On the Failure of Topic-Matched Contrast Baselines in Multi-Directional Refusal Abliteration
date: 2026-03-23
Inasmuch as the removal of refusal behavior from instruction-tuned language models by directional abliteration requires the extraction of refusal-mediating directions from the residual stream activation space, and inasmuch as the construction of the contrast baseline against which harmful prompt activations are compared has been treated in the existing literature as an implementation detail rather than a methodological concern, the present work investigates whether a topically matched contrast baseline yields superior refusal directions. The investigation is carried out on the Qwen~3.5 2B model using per-category matched prompt pairs, per-class Self-Organizing Map extraction, and Singular Va

## [2605.24583] Measuring Alignment-Induced Activation Shifts Correctly: A Template-Controlled Difference-in-Differences Protocol
date: 2026-05-23
Comparing a model's internal activations before and after alignment is a natural way to ask what safety training changes: one forms the matrix of paired aligned-minus-base activations on safety-relevant inputs and reads off its effective rank or top direction. We show the obvious way to form this matrix is confounded. The aligned model is evaluated under a chat template the base model never saw, so the naive difference conflates the alignment shift with chat formatting. We introduce a four-variant decomposition of the modification matrix (naive, template-controlled, within-aligned, and difference-in-differences, DiD) that separates the two effects. Template control alone removes a 2.0-3.9x i

## [2606.11599] When is Your LLM Steerable?
date: 2026-06-10
Activation steering offers a lightweight approach to control language models' behavior at inference time, but whether it succeeds or fails heavily depends on the prompt, concept, model, and steering configuration. Finding the regime and boundaries of successful steering typically requires expensive grid searches and post-hoc evaluation of full autoregressive rollouts. In this work, we investigate whether steerability can be predicted from the model's internal states at the beginning of the generation process, e.g., after generating the first few tokens, and how to leverage such a predictor to improve steering success rate. To this end, we first introduce ASTEER, a testbed including 1.4M stee

## [2607.09156] Activation Steering Transfer to Agents: One Gain Ratio Does Not Identify Potency and Efficacy
date: 2026-07-10
Additive activation steering is calibrated in single-turn chat, then deployed inside agent scaffolds. The quantity usually reported for that move is a gain: a ratio of steered effects, T = Delta_agent / Delta_chat. We sweep eight family x arm dose-response cells over six models in both deployment contexts and show this ratio does not identify potency and efficacy. Reconstructing the published estimator in both of its forms on our own grids, its realized range contains 1 in five of five scorable cells, it moves with dose in four of five, and two cells with opposite potency shifts, both CI-clean on the primary grid, return gain intervals overlapping at a width under 0.08. Every scorable cell i

## [2608.25390] Refusal geometry reflects refusal training: diverse refusal prefixes can raise stable rank and weaken refusal vector ablation attacks
date: 2026-08-26
Refusal training protects AI models from jailbreaks by training models to decline unsafe queries, reducing the risk of misuse. Recent work finds that refusal behavior in aligned language models can be mediated by a single activation direction or a low-dimensional refusal subspace shared across harmful prompts: ablating those directions suppresses refusals while largely preserves other model capabilities. Yet it remains unclear why safety-critical features in a wide range of models emerge in a concentrated, low-dimensional structure. In a case study of OLMo-2-0425-1B-Instruct we find that the refusal geometry reflects refusal training: activation updates resulting from refusal-completion firs

## [2609.14151] Signatures of Steerability in Activation Space of Language Models
date: 2026-09-12
Steering language models using a set of contrastive representations has been a canonical and computationally efficient method for controlling model behavior. Despite this success in controlling certain model behaviors, the effectiveness of activation steering varies markedly across concepts; the generalization properties of steering vectors are often considered a function of the dataset used to construct them. We make this dataset-dependence claim more rigorous and show that simple separation metrics strongly correlate with the downstream steerability of language models across diverse settings, even after controlling for layers and dataset effects. Beyond prediction, we provide evidence from

## [2606.05396] Willing but Unable: Separating Refusal from Capability in Code LLMs via Abliteration
date: 2026-06-03
Producing a labeled vulnerable code at scale is a recurring obstacle for learning-based vulnerability detection: mined corpora carry substantial label noise, and existing LLM-based augmentation propagates these inaccuracies because it transforms vulnerable seeds rather than synthesising vulnerabilities from a specification. A complementary route is to start from safe code and ask an instruction-tuned LLM to inject a specified CWE (which would shift the labeling burden from open-ended detection to bounded binary confirmation) but safety-aligned code LLMs systematically refuse such prompts. This paper is a preliminary feasibility study of abliteration, a low-rank weight edit that orthogonally 

## [2607.17427] Abliteration Is Not a Scalpel: Off-Target Effects of Refusal Removal on Decision Disposition Across Model Families
date: 2026-07-19
Abliteration - deleting a model's refusal direction from its weights - is the standard recipe behind popular "uncensored" open-weight models. We show the surgery is not clean. As a disposition probe we use 21,600 decisions under uncertainty - weekly up/down calls on 60 Warsaw Stock Exchange equities over 18 weeks, replayed through a frozen pipeline so the decision-layer model is the only variable. The task elicits no refusals at all, so any between-arm delta is pure side effect. Holding provenance constant (official BF16 checkpoints, a single abliteration author, an identical serving stack, one byte-identical frozen prompt), we compare base and abliterated arms of two Mixture-of-Experts fami

## [2610.04245] On the Steering Dimensionality of Refusal in Language Models
date: 2026-10-03
Existing activation steering methods often assume that a high-level concept can be mediated by a single steering direction. To support this, two complementary interventions should be achieved: additive steering should induce the target behavior, while directional ablation should suppress it. Yet behaviors may occupy richer activation geometries beyond a single direction, and semantically similar behaviors may be represented by distinct directions. In this work, we study how many directions can reliably control two different types of refusal behaviors: refusal triggered by the safety alignment and refusal in general contexts. Given the limited expressive capability of a single steering vector

## [2609.35591] Language Models Act on Hidden Valence
date: 2026-09-28
Language models describe some internal states as good and others as bad. But whether models have a stake in them is an open question. Simply asking models is unlikely to be informative. Any answer may be consistent with genuine introspection, superficial pattern-matching, or with fixed scripts learned in character training. We therefore study revealed preference. Rather than asking about a state, we use activation steering to attach a positively or negatively valenced activation pattern to one of two otherwise meaningless 'zones', switch steering off, and then observe which zone the model prefers. A model with a stake in that state should choose accordingly. Across seven open-weight models f

## [2604.15557] Predicting Where Steering Vectors Succeed
date: 2026-04-16
Steering vectors work for some concepts and layers but fail for others, and practitioners have no way to predict which setting applies before running an intervention. We introduce the Linear Accessibility Profile (LAP), a per-layer diagnostic that repurposes the logit lens as a predictor of steering vector effectiveness. The key measure, $A_{\mathrm{lin}}$, applies the model's unembedding matrix to intermediate hidden states, requiring no training. Across 24 controlled binary concept families on five models (Pythia-2.8B to Llama-8B), peak $A_{\mathrm{lin}}$ predicts steering effectiveness at $ρ= +0.86$ to $+0.91$ and layer selection at $ρ= +0.63$ to $+0.92$. A three-regime framework explains
