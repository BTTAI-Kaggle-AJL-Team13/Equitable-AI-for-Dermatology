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

Missing values: this dataset included only one variable with missing values, qc, which is an important indicator for understanding whether the image is adequately diagnostic of the skin condition. Of our 2860 observations, we had values for qc in only 90 observations, and 30 observations in the test dataset. Of those 90 observations, 4 were wrongly labelled with images that clearly did not pertain to the labelled skin condition. These observations were removed from our model training, but it is unclear how many of the 2770 observations with missing qc values were also wrongly labelled. Furthermore, the test dataset excluded the nine_partition_label and three_partition_label, which are often unavailable until some sort of medical examination.

### Data Preprocessing
Our team performed various steps to ensure the data was clean, uniformly sized, and ready for model input.
- Imported essential libraries, including Keras, PIL, NumPy, and scikit-learn
- Extracted image dimensions and RGB pixel values from JPEG files in the training directory
- Resized images to a consistent size and normalized pixel values to the range [0,1] to standardize input for the model
- Filtered out samples labeled as "Wrongly Labelled"
- One-hot encoded categorical variables

### Visualizations
Distribution of skin types: The Fitzpatrick Scale graph shows the distribution of skin types present in the dataset using Fitzpatrick Skin Types, which range from 1 for the lightest skin tones to 7 for the darkest skin tones. This allows us to assess the diversity of skin tones represented, which is skewed toward lighter skin tones. People with an FST of 4 or larger are a significant minority in the population, with 698 of the 2860 observations, or just less than 25% of the population. Since darker skin tones are such a small percentage of the dataset, our model will have less training data and will likely perform worse in future cases for people with dark skin tones. 

<img src="https://github.com/user-attachments/assets/2272bd20-0025-40d7-acd2-68f2375c79d6" alt="Fitzpatrick Scale" width="275" height="200">

Distribution of skin conditions: Of the 21 skin conditions represented in our dataset, squamous cell carcinoma and basal cell carcinoma are the most common, comprising about 25.7% of the data. Acne, the most common type of skin condition, is only the 7th most represented skin condition in the data, making up about 4.5% of the observations. For many of the less common skin conditions in the dataset, there are only about 50 observations, or less than 2% of the dataset, for the model to learn from. 

![image](https://github.com/user-attachments/assets/0f71b96f-cc20-4b61-a744-c301f2e47959)


## **🧠 Model Development**

* [ADD] Model(s) used (e.g., CNN with transfer learning, regression models)
* [ADD] Feature selection and Hyperparameter tuning strategies
* [ADD] Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)
* Applied an 80% train and 20% test split.
* Trained a baseline CNN on pictures and classifications in the training set. Achieved an F1-score of ~0.2 in Kaggle.
* Fine-tuned a pre-trained CNN model (Resnet and MobileNet) by implementing transfer learning. Augmented data from underrepresented skin conditions and increased the number of layers in the model.
* Achieved an accuracy of ~70% during training on the validation data and an improved F1-score of ~0.59 in Kaggle. 


## **📈 Results & Key Findings**

* Placed 19/74 on Kaggle Competition Leaderboard
* [ADD] How your model performed overall
* [ADD] How your model performed across different skin tones (AJL)
* [ADD] Insights from evaluating model fairness (AJL)

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

## **👥 Team Members**
| Ashley Bao | @ashleybao | Co-led team, preprocessed data, built baseline CNN model |

| Aimee Hong | @aimeehong1 | Co-led team, conducted exploratory data analysis, one hot encoded categorical variables, and built a baseline CNN model |

| Varsha Athreya | @varsha487 | Preprocessed data, fine-tuned pre-trained CNN model using transfer learning, calculated evaluation metrics |

| Anissa Patel | @anissakp | Researched pre-processing and model selection, built baseline CNN model |
