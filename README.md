# 150-days-of-ML
This repository documents my ongoing journey in Machine Learning & AI. Rather than showcasing only polished projects, it focuses on the daily process of learning, building, debugging, reading, researching, reflecting, and shipping projects across classical machine learning, deep learning, MLOps, and applied AI. Walk with me.

## Day 001
Over the past four hours, I went from being uncertain about how to frame [HyperSense](https://github.com/Phawazz/HyperSense) in a methodologically rigorous way to developing a clear roadmap from its current state to a submission-ready abstract. The project is now structured into three sequential phases: Transportability, Regional Adaptation, and Clinical Enrichment. With the overall direction now established, Stage 1 (Transportability) begins tomorrow.

## Day 002
I spent the past 3+ hours reviewing similar literature on studies where machine learning approaches were used to build predictive models for hypertension risk stratification. The most interesting paper I found was [Septian et al.'s study](https://link.springer.com/article/10.1007/s10916-025-02253-5), which demonstrated rigorous methodology and made use of a huge, robust nationally representative Indonesian dataset. While the study consistently reported discrimination metrics such as AUC, sensitivity, and specificity, it did not evaluate calibration, which is equally important for assessing model performance. Another key takeaway that I plan to incorporate into the HyperSense manuscript is benchmarking our model's AUC against the pooled c-statistic of **0.75** reported in a [systematic review](https://doi.org/10.1371/journal.pone.0266334) of machine learning models for hypertension risk prediction.

Another paper that stood out to me was [this study](https://www.sciencepublishinggroup.com/article/10.11648/j.ajmcm.20170202.12), published in 2016. It used decision tree algorithms (C4.5 and ID3) to predict hypertension using an almost unbelievable dataset of just 32 samples, and also introduced me to older approaches such as fuzzy logic systems. Reading it made me appreciate just how far AI/ML has come over the past decade, and it also made me excited about what the next 10 years might look like.

This quick review of the literature was necessary before beginning Stage 1 of **HyperSense** because I wanted to ensure that I wasn't making avoidable methodological mistakes before moving forward.

## Day 003
I started the full technical phase of HyperSense Phase 2 today (night actually). I also set up the dedicated transportability notebook, [01_transportability](https://github.com/Phawazz/HyperSense/blob/main/HyperSense%20v1.2/01_transportability.ipynb), loaded the REMAH dataset locally, and began systematically re-examining & inspecting its variables and coding. I think I've made a solid start on the Ghana → Nigeria transportability analysis. Work continues on it tomorrow (random, but I have less than 7 days to complete all the technical work for this abstract). 

## Day 004
I completed the first full technical stage of HyperSense Phase 2 today by carrying out the external transportability evaluation of the Ghana-derived v1.1 model using the REMAH Nigerian cohort. I harmonized the relevant REMAH variables to match the model's original six predictors, generated the measurement-only hypertension outcome, applied the saved v1.1 model without retraining or refitting, and evaluated its performance both across the full Nigerian cohort and within the model's original age range. I also compared the resulting discrimination, classification, and calibration metrics against the original Ghana performance. This gives me a proper baseline for assessing what changes, if any, are achieved when Nigerian data are incorporated in the next stage.

## Day 005

While deep into the regional adaptation phase of my ongoing project, I came across this [intro to deep learning in healthcare](https://www.linkedin.com/learning/deep-learning-fundamentals-for-healthcare) on LinkedIn. As part of my plans to delve *'deeper'* into deep learning and MLOps before year end, I couldn't resist the urge to skim through the entire course. While much of the content was familiar, the brief course was particularly helpful in clearly organising concepts I've encountered before but couldn't always explain or place appropriately.

I was exposed to DL architectures such as multilayer perceptrons (MLPs), CNNs, RNNs, Transformers, autoencoders, GANs, and others. Apart from the L1 and L2 regularisation techniques I've been familiar with in classical ML, I learnt about **early stopping** (which honestly sounds a bit like *pruning* a random forest or decision tree to me) and **dropout**. In addition to gradient descent, SGD, and Adam as optimisation algorithms I was already familiar with, I also learnt about RMSProp (root mean square propagation). Lastly, while *MSE* and *MAE* are popular *loss functions* for regression tasks, *binary and categorical cross-entropy* for classification in neural networks are concepts I'm much less familiar with. Can't wait to delve deeper later. Back to my classical work for now!

## Day 006
Day 006 turned out to be a very low-output day due to some resons. However, I managed to start work on the second phase of my ongoing project. I began the regional adaptation stage by pooling the original Ghanaian training data with harmonized Nigerian data while keeping the feature set fixed and carefully checking the resulting source and outcome composition before model development.

## Day 007

I continued the regional adaptation stage of HyperSense Phase 2 today by developing the pooled Ghana–Nigeria baseline model. After fixing the six-feature modelling set and confirming the training/test outcome composition, I trained an XGBoost classifier on the pooled training data and evaluated it on the held-out test set. I assessed overall discrimination and classification performance, followed by separate Ghanaian and Nigerian evaluations to examine regional heterogeneity. The results showed a substantial difference in sensitivity between the two populations, which raised an important methodological issue that I need to investigate deeply before proceeding any further.

I also spent some time understanding **model calibration** more rigorously rather than treating it as just another evaluation metric. In particular, I now have a much clearer grasp of how predicted probabilities can be transformed into **log-odds(logits)** and regressed against the observed outcome to obtain the calibration intercept and slope. A *perfectly calibrated model* has an **intercept of 0 and a slope of 1**; I also learnt the biostatistical terminology of *calibration-in-the-large* for the intercept and *Cox calibration regression* for the calibration slope. This was probably the most useful conceptual takeaway from today's modelling work.

## Day 008

Today I completed the **regional adaptation stage of HyperSense**, developing and evaluating the pooled Ghana–Nigeria XGBoost model. I tuned the model using **10-fold cross-validation** on the training data, then derived the final classification threshold from **out-of-fold predictions**, giving a threshold of **0.1689** and a Youden's index of **0.4881**. I particularly enjoyed converting `J = sensitivity + specificity − 1` to `J = TPR − FPR` with some basic biostat knowledge.

After freezing the threshold, I refitted the model on the full training set and evaluated it once on the held-out test set. The final model achieved an overall **AUC of 0.803**, with **0.768 in Ghana** and **0.784 in Nigeria**, while also revealing important differences in regional performance. I also developed a better understanding of the **Brier score** as essentially the MSE of predicted probabilities, where lower values indicate better probabilistic accuracy, further reinforcing the distinction between **discrimination and calibration**.

Finally, I consolidated the results across Ghana v1.1, REMAH transportability, the Stage 2 baseline, and the final tuned model. **Stage 2 is done!**

## Day 009
Today I went back to HyperSense v1.1 to address a methodological gap I had previously overlooked in the deployment threshold selection. The original threshold had been derived by evaluating candidate thresholds directly on the held-out test set, meaning the test set was inadvertently involved in a modelling decision. More importantly, the final threshold was ultimately chosen somewhat intuitively based on which point seemed to give sufficiently high sensitivity for a screening tool, rather than through a reproducible criterion that properly considered both sensitivity and specificity.

I corrected this by keeping the XGBoost model completely frozen and generating 10-fold out-of-fold predictions on the training set only using the already-selected hyperparameters and DHS survey weights. I then evaluated a prespecified threshold grid using Youden's index (J = sensitivity + specificity − 1) and selected the threshold that maximised it, giving a new deployment threshold of 0.54. I subsequently applied this threshold once to the held-out test set, obtaining 77.90% sensitivity, 61.60% specificity, 20.92% PPV, and 95.53% NPV. I also updated the threshold artifact, deployment logic documentation, README, and downstream threshold-dependent reporting while leaving the frozen model and its ~0.79 AUC unchanged. The notebook is available for review by anone [here](https://github.com/Phawazz/HyperSense/blob/main/notebooks/07_bmi_model_ready.ipynb).

## Day 010
Today I went back through HyperSense v1.2 to update and cross-check the downstream metrics affected by the audited v1.1 threshold re-work, ensuring the corrected values were consistently reflected across the relevant notebooks and comparisons. With that cleanup done, the remaining work for the project is stage 3: Clinical Enrichment, using Nigerian data only. Tomorrow, we build.

## Day 011
Spent some time restructuring the objectives for Stage 3 to ensure methodological consistency throughout the project. The need for reproducibility also meant being more deliberate with each decision and taking extra care to ensure the methodology could be consistently applied and replicated.

## Day 012
Travelled to Lagos for the MedVerge conference, convened by Dr Abiodun Aderemi, founder and CEO of Helpmum, Dobic Health, and SmartMRS. It was a great opportunity to gain some exposure to the current state of healthtech in Nigeria and the companies and people shaping the industry. The session that stood out to me the most was the interview with Dr Abiodun. It was particularly interesting to hear his story and understand his approach to building and solving problems.

## Day 013
Another equally busy and tasking day. I couldn't get anything substantial done.

## Day 014
Spent most of the day working through the methodological decisions for Stage 3, particularly around the `HTN/HTN_DR` outcome construction, substantial missingness in variables such as the laboratory measurements, potential leakage from hypertension-management variables, and deciding which additional clinical predictors to include. A lot of the work has been necessary to make sure the analysis is methodologically sound and reproducible, but it has also meant that the actual technical work has been delayed more than I initially expected. At this point, most of the groundwork and decision-making is finally out of the way, so the real implementation can begin.

## Day 015
Started implementing Stage 3 by loading the REMAH dataset, mapping the variables needed for the outcome and final predictor sets, and auditing their coding, missingness, and ranges. I was able to confirm that the five clinic BP measurements have very little percentage of missingness, identified a few invalid categorical codes and unusual continuous values to investigate, and began constructing the Stage 3 hypertension outcome from the five-reading BP average and antihypertensive treatment status.

## Day 016
---

## Day 017
It’s relieving to finally come to an end of this project. Funny how what actually took most of the time was the data cleaning, feature engineering, and methodological decisions rather than the code itself. From constructing the hypertension outcome from the five BP readings and treatment status, to defining the final predictor sets, handling missingness, comparing the parsimonious and clinically enriched XGBoost models, and always ensuring the notebooks are well-organized for quick review by anyone or even future me. In the end, the enriched model showed better discrimination and detection, while the simpler model was slightly better calibrated. It’s been quite the process, but I’ve learned a lot from working through the details rather than just getting the model to run. Next is to work on the abstract and full manuscript. & some other cool technical work.