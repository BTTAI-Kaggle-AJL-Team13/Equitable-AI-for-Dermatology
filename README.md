# Equitable AI for Dermatology

## **🎯 Project Highlights**

* [ADD] Built a \[insert model type\] using \[techniques used\] to solve \[Kaggle competition task\]
* [ADD] Achieved an F1 score of \[insert score\] and a ranking of \[insert ranking out of participating teams\] on the final Kaggle Leaderboard
* [ADD] Used \[explainability tool\] to interpret model decisions
* [ADD] Implemented \[data preprocessing method\] to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

## **🏗️ Project Overview**

The Kaggle competition, in partnership with the Break Through Tech AI Program and the Algorithmic Justice League, addresses bias in dermatology AI tools, which often underperform for individuals with darker skin tones due to a lack of diverse training data. Participants are tasked with developing inclusive machine learning models to ensure equitable and accurate dermatological diagnoses.

This challenge has a significant real-world impact. Misdiagnoses and delayed treatments disproportionately affect underserved communities, worsening health disparities. By creating fairer AI models, participants can contribute to improving healthcare outcomes and promoting algorithmic justice in medical technology.

## **📊 Data Exploration**

### Dataset
The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set. We used a subset in order to create a more manageable and hopefully satisfying classification problem, while trying to maintain some of the representation issues surfaced by the full set.

### Data Preprocessing
Our team performed various steps to ensure the data was clean, uniformly sized, and ready for model input.
- Imported essential libraries, including Keras, PIL, NumPy, and scikit-learn
- Extracted image dimensions and RGB pixel values from JPEG files in the training directory
- Resized images to a consistent size and normalized pixel values to the range [0,1] to standardize input for the model
- Filtered out samples labeled as "Wrongly Labelled"
- One-hot encoded categorical variables

### Visualizations
The Fitzpatrick Scale graph visualizes the different skin types present in the dataset, which is important because it helps assess the diversity of skin tones represented.

<img src="https://github.com/user-attachments/assets/2272bd20-0025-40d7-acd2-68f2375c79d6" alt="Fitzpatrick Scale" width="275" height="200">

## **🧠 Model Development**

* [ADD] Model(s) used (e.g., CNN with transfer learning, regression models)
* [ADD] Feature selection and Hyperparameter tuning strategies
* [ADD] Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)

## **📈 Results & Key Findings**

* Placed 19/74 on Kaggle Competition Leaderboard
* [ADD] How your model performed overall
* [ADD] How your model performed across different skin tones (AJL)
* [ADD] Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

## **👥 Team Members**
| Ashley Bao | @ashleybao | Co-led team,  |

| Aimee Hong | @aimeehong1 | Co-led team,  |

| Varsha Athreya | @varsha487 | contribution |

| Anissa Patel | @anissakp | Researched pre-processing and model selection, built baseline CNN model |
