!pip install sentence-transformers
!pip install prettytable
!pip install torch torchvision torchaudio
!pip install transformers

import numpy as np
import pandas as pd

import transformers
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification, AdamW

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from prettytable import PrettyTable

from sentence_transformers import SentenceTransformer, util

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson

from sklearn.metrics import mean_squared_error, mean_absolute_error


### Uploading the document with the tweets data (Google Collab Env) ###

from google.colab import files
uploaded = files.upload()
df = pd.read_excel('FileName.xlsx')


### ZTPI Items ###

scale_items = [
    "I believe that getting together with one's friends to party is one of life's important pleasures.",
    "Familiar childhood sights, sounds, smells often bring back a flood of wonderful memories.",
    "Fate determines much in my life.",
    "I often think of what I should have done differently in my life.",
    "My decisions are mostly influenced by people and things around me.",
    "I believe that a person's day should be planned ahead each morning.",
    "It gives me pleasure to think about my past.",
    "I do things impulsively.",
    "If things don't get done on time, I don't worry about it.",
    "When I want to achieve something, I set goals and consider specific means for reaching those goals.",
    "On balance, there is much more good to recall than bad in my past.",
    "When listening to my favorite music, I often lose all track of time.",
    "Meeting tomorrow's deadlines and doing other necessary work comes before tonight's play.",
    "Since whatever will be will be, it doesn't really matter what I do.",
    'I enjoy stories about how things used to be in the "good old times."',
    "Painful past experiences keep being replayed in my mind.",
    "I try to live my life as fully as possible, one day at a time.",
    "It upsets me to be late for appointments.",
    "Ideally, I would live each day as if it were my last.",
    "Happy memories of good times spring readily to mind.",
    "I meet my obligations to friends and authorities on time.",
    "I've taken my share of abuse and rejection in the past.",
    "I make decisions on the spur of the moment.",
    "I take each day as it is rather than try to plan it out.",
    "The past has too many unpleasant memories that I prefer not to think about.",
    "It is important to put excitement in my life.",
    "I've made mistakes in the past that I wish I could undo.",
    "I feel that it's more important to enjoy what you're doing than to get work done on time.",
    "I get nostalgic about my childhood.",
    "Before making a decision, I weigh the costs against the benefits.",
    "Taking risks keeps my life from becoming boring.",
    "It is more important for me to enjoy life's journey than to focus only on the destination.",
    "Things rarely work out as I expected.",
    "It's hard for me to forget unpleasant images of my youth.",
    "It takes joy out of the process and flow of my activities, if I have to think about goals, outcomes, and products.",
    "Even when I am enjoying the present, I am drawn back to comparisons with similar past experiences.",
    "You can't really plan for the future because things change so much.",
    "My life path is controlled by forces I cannot influence.",
    "It doesn't make sense to worry about the future, since there is nothing that I can do about it anyway.",
    "I complete projects on time by making steady progress.",
    "I find myself tuning out when family members talk about the way things used to be.",
    "I take risks to put excitement in my life.",
    "I make lists of things to do.",
    "I often follow my heart more than my head.",
    "I am able to resist temptations when I know that there is work to be done.",
    "I find myself getting swept up in the excitement of the moment.",
    "Life today is too complicated; I would prefer the simpler life of the past.",
    "I prefer friends who are spontaneous rather than predictable.",
    "I like family rituals and traditions that are regularly repeated.",
    "I think about the bad things that have happened to me in the past.",
    "I keep working at difficult, uninteresting tasks if they will help me get ahead.",
    "Spending what I earn on pleasures today is better than saving for tomorrow's security.",
    "Often luck pays off better than hard work.",
    "I think about the good things that I have missed out on in my life.",
    "I like my close relationships to be passionate.",
    "There will always be time to catch up on my work.",

]


### ZTPI Scoring Key ###

past_negative_items = [
    49,
    15,
    33,
    3,
    53,
    26,
    21,
    35,
    32,
    4
]

present_hedonistic_items = [
    41,
    30,
    25,
    22,
    7,
    16,
    47,
    31,
    43,
    54,
    45,
    0,
    18,
    27,
    13
]

future_items = [
    12,
    39,
    44,
    9,
    50,
    17,
    5,
    20,
    42,
    29,
    8,
    55,
    23
]

past_positive_items = [
    6,
    28,
    19,
    10,
    14,
    1,
    48,
    40,
    24
]

present_fatalistic_items = [
    37,
    38,
    13,
    36,
    52,
    2,
    34,
    46,
    51
]


factors = [
    "Past-Negative",
    "Present-Hedonistic",
    "Future",
    "Past-Positive",
    "Present-Fatalistic"
]

factor_loadings = [
    [0.07, 0.42, -0.02, 0.14, -0.1],
    [-0.08, 0.18, 0.06, 0.62, 0.02],
    [0.24, 0.19, 0.09, 0.14, 0.44],
    [0.66, -0.01, -0.07, 0.05, 0.15],
    [0.41,0.00,0.02,0.23,0.18],
    [0.08,-0.16,0.46,0.1,0.02],
    [-0.25,0.14,0.01,0.68,-0.02],
    [0.03,0.51,-0.27,-0.1,0.05],
    [-0.09,0.21,-0.33,-0.08,0.12],
    [-0.16,0.13,0.56,-0.03,-0.09],
    [-0.41,0.06,0.03,0.63,-0.12],
    [0.09,0.32,-0.04,0.13,0.22],
    [-0.08,-0.17,0.63,0.04,0.1],
    [0.1,0.04,-0.15,-0.07,0.64],
    [0.18,0.09,0.09,0.63,0.06],
    [0.69,0.16,-0.01,-0.18,0.06],
    [-0.2,0.5,0.19,0.11,-0.06],
    [0.11,0.04,0.48,-0.06,-0.04],
    [0.05,0.38,0.12,0.1,0.07],
    [-0.24,0.24,0.11,0.64,-0.03],
    [-0.12,0.04,0.46,0.17,-0.04],
    [0.49,0.24,0.07,-0.2,-0.04],
    [0.07,0.51,-0.25,-0.12,0.13],
    [0.06,0.28,-0.49,-0.11,0.2],
    [0.55,-0.02,0.02,-0.52,0.21],
    [0.05,0.56,0.05,0.18,-0.14],
    [0.55,0.03,-0.18,0.05,0.02],
    [0.00,0.36,-0.3,0.06,0.33],
    [0.04,0.06,-0.02,0.64,0.21],
    [0.08,0.03,0.37,0.16,-0.29],
    [-0.00,0.7,-0.02,-0.00,0.03],
    [-0.13,0.45,-0.08,0.08,0.15],
    [0.43,0.04,-0.17,-0.08,0.29],
    [0.67,-0.01,0.05,-0.25,0.07],
    [0.2,0.16,-0.2,-0.09,0.42],
    [0.47,0.08,0.06,0.24,0.21],
    [0.14,0.17,-0.12,-0.04,0.59],
    [0.17,-0.02,0.06,0.02,0.73],
    [0.04,-0.02,-0.01,-0.1,0.68],
    [-0.17,-0.02,0.61,-0.01,0.04],
    [-0.00,0.00,-0.00,-0.45,0.25],
    [0.00,0.71,-0.01,-0.04,0.08],
    [-0.05,0.07,0.45,0.07,-0.05],
    [0.18,0.45,-0.1,0.07,0.12],
    [-0.16,-0.09,0.61,-0.06,-0.06],
    [0.16,0.44,-0.22,0.23,0.1],
    [0.2,-0.09,-0.00,0.09,0.42],
    [-0.04,0.45,-0.16,-0.1,0.18],
    [0.1,-0.06,0.11,0.47,-0.03],
    [0.76,0.06,0.06,-0.08,0.05],
    [0.09,-0.07,0.51,0.01,-0.08],
    [-0.05,0.28,-0.18,-0.04,0.34],
    [0.08,0.14,-0.11,0.02,0.45],
    [0.63,-0.07,-0.13,0.01,0.21],
    [0.2,0.44,-0.00,0.07,-0.02],
    [-0.11,0.29,-0.36,0.09,0.1],

]


### Tweets & ZTPI Items Embeddings + Similarity Scores (CCR Values) Calculation ###

model = SentenceTransformer('all-mpnet-base-v2')

def calculate_similarity_scores(InTweets,InScaleItems, InModel):
    scale_items_embeddings = InModel.encode(InScaleItems, convert_to_tensor=True).cuda()
    tweets_list_embeddings = InModel.encode(InTweets, convert_to_tensor=True).cuda()

    similarity_scores_list = []

    for tweet_embedding in tweets_list_embeddings:
        similarity_scores = util.pytorch_cos_sim(tweet_embedding.unsqueeze(0), scale_items_embeddings).tolist()
        similarity_scores_list.append(similarity_scores)

    similarity_scores_array = np.array(similarity_scores_list)
    similarity_scores_array = similarity_scores_array.squeeze(axis=1)
    return similarity_scores_array

similarity_scores = calculate_similarity_scores(Tweets,scale_items, model)
print(similarity_scores)
print(similarity_scores.shape)


factors_mapping = {
    "past_negative_items": past_negative_items,
    "present_hedonistic_items": present_hedonistic_items,
    "future_items": future_items,
    "past_positive_items": past_positive_items,
    "present_fatalistic_items": present_fatalistic_items
}
data = []
for i, tweet_similarity_scores in enumerate(similarity_scores):
    for category, category_items in factors_mapping.items():
        category_similarity_scores = [tweet_similarity_scores[index] for index in category_items]
        mean_similarity = np.mean(category_similarity_scores)
        data.append([f"Tweet {i+1}", category, mean_similarity, df.loc[i,'text']])

df_factors_mean = pd.DataFrame(data, columns=["Tweet Num", "Category", "Mean Similarity Score", "Tweet Text"])

df_factors_mean['Retweets'] = np.repeat(Retweets.values, 5)

df_factors_mean.reset_index(drop=True, inplace=True)

print(df_factors_mean)


### Pivot the DataFrame to have the categories as columns ###

df_factors_mean_pivoted = df_factors_mean.pivot_table(index=['Tweet Num', 'Tweet Text', 'Retweets'],
                                                      columns='Category',
                                                      values='Mean Similarity Score',
                                                      aggfunc='first').reset_index()


df_factors_mean_pivoted.columns.name = None
df_factors_mean_pivoted = df_factors_mean_pivoted.rename_axis(None, axis=1)

print(df_factors_mean_pivoted)


### Drop duplicates ###

df_no_dups = df_factors_mean_pivoted.copy()

# Drop duplicates based on Tweet Text column
df_no_dups.drop_duplicates(subset=['Tweet Text'], inplace=True)
print(df_no_dups)


### The Retweets Count ###

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['Retweets'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Retweets for Unique Tweets')
plt.xlabel('Number of Retweets')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


### Drop duplicates, include only the 99 percentile based on the number of retweets - no ouliers ###

df_no_dups_no_outliers = df_factors_mean_pivoted.copy()

# Drop duplicates based on Tweet Text column
df_no_dups_no_outliers.drop_duplicates(subset=['Tweet Text'], inplace=True)

# Keep only rows where the number of retweets is within the 99th percentile
percentile_value = df_no_dups_no_outliers['Retweets'].quantile(0.99)
df_no_dups_no_outliers = df_no_dups_no_outliers[df_no_dups_no_outliers['Retweets'] <= percentile_value]

df_no_dups_no_outliers.reset_index(drop=True, inplace=True)

print(df_no_dups_no_outliers)


### The Retweets Count Range ###

retweets_range = np.max(df['Retweets']) - np.min(df['Retweets'])
max = np.max(df['Retweets'])
min = np.min(df['Retweets'])

# Calculate variance
retweets_variance = np.var(df['Retweets'])

print(max)
print(min)
print("Range of Retweets:", retweets_range)
print("Variance of Retweets:", retweets_variance)

### Zero Inflated Negative Binomial modelling of count data with excess zeros ###

formula = 'Retweets ~ future_items + past_negative_items + past_positive_items + present_fatalistic_items + present_hedonistic_items'
model2 = sm.ZeroInflatedNegativeBinomialP(endog=df_no_dups['Retweets'],
                                          exog=sm.add_constant(df_no_dups[['future_items', 'past_negative_items', 'past_positive_items', 'present_fatalistic_items', 'present_hedonistic_items']]),
                                          inflation='logit')
result = model2.fit()

print(result.summary())

result = model2.fit()

predictions = result.predict()

actual_data = df_no_dups['Retweets']

mse = mean_squared_error(actual_data, predictions)
mae = mean_absolute_error(actual_data, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')