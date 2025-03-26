# Equitable AI for Dermatology

## **🎯 Project Highlights**

* Developed a convolutional neural network (CNN) model and utilized MobileNet for transfer learning to improve the classification of skin conditions.
* Used data augmentation on underrepresented samples and selective unfreezing on MobileNet layers to improve accuracy
* Achieved an F1-score of approximately 0.59 and ranked 19th out of 74 participating teams on the final Kaggle Leaderboard.


🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

## **🏗️ Project Overview**

The Kaggle competition, in partnership with the Break Through Tech AI Program and the Algorithmic Justice League, addresses bias in dermatology AI tools that often underperform for individuals with darker skin tones due to a lack of diverse and comprehensive training data. Participants are tasked with developing inclusive machine learning models to ensure equitable and accurate dermatological diagnosis using limited medical data and image classification.

This challenge has a significant real-world impact since underperforming AI dermatology tools can lead to diagnostic errors and delayed treatments that disproportionately affect underserved communities and worsen health disparities. By creating more equitable AI models, participants can contribute to improving healthcare outcomes and promoting algorithmic justice in medical technology.

## **📊 Data Exploration**

### Dataset
The dataset is a subset of the FitzPatrick17k dataset, a labeled collection of about 17,000 images depicting a variety of serious (e.g., melanoma) and cosmetic (e.g., acne) dermatological conditions with a range of skin tones scored on the FitzPatrick skin tone scale (FST). About 4500 images are in this set, representing 21 skin conditions out of the 100+ in the full FitzPatrick set. This competition subset was utilized in order to create a more manageable and satisfying classification problem, while trying to maintain some of the representation issues surfaced by the full set.

The dataset included the following variables: 
md5hash (string) is the unique identifier
* fitpatric_scale (int) is a nominal int in range [-1, 0) and [1, 6] indicating self-described Fitzpatrick Skin Type (FST). A larger FST indicates a darker skin tone.
* fitzpatrick_centaur (int) nominal int in the range [-1, 0) and [1, 6] indicating FST assigned by Centaur Labs, a medical data annotation firm. A larger FST value indicates a darker skin tone
* label (string) is our dependent variable, indicating the medical diagnosis with 21 possible conditions
* nine_partition_label (string) is a categorical variable indicating one of nine diagnostic categories
  * Inflammatory
  * Malignant-epidermal
  * Malignant-melanoma
  * Benign-epidermal
  * Benign-dermal
  * Malignant-cutaneous-lymphoma
  * Malignant-dermal
* three_partition_label (string) is a categorical variable indicating one of three diagnostic categories
  * Malignant
  * Non-neoplastic
  * Benign
* qc (int) a nominal int for quality control check by a Board-certified dermatologist.
  * 1: Diagnostic - the image shows a good example of the skin condition
  * 2: Characteristic - the image shows something that could be the skin condition, but isn't diagnostic
  * 3: Wrongly labelled - the image shows something that is definitely not the labeled condition
  * 4: Other
  * 5: Potentially - not clearly diagnostic, but not necessarily mislabeled, further testing would be required
* ddi_scale (int) A column used to reconcile this dataset with another dataset (may not be relevant)

Missing values: this dataset included only one variable with missing values, qc, which is an important indicator for understanding whether the image is adequately diagnostic of the skin condition. Of our 2860 observations, we had values for qc in only 90 observations, and 30 observations in the test dataset. Of those 90 observations, 4 were wrongly labelled with images that clearly did not pertain to the labelled skin condition. These observations were removed from our model training, but it is unclear how many of the 2770 observations with missing qc values were also wrongly labelled. 

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

![image](https://github.com/user-attachments/assets/0f71b96f-cc20-4b61-a744-c301f2e47959)


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
