# -*- coding: utf-8 -*-
"""
Complete Analysis Script for Big Brothers Big Sisters Match Data
Covers:
1. Cadence analysis (days between contact notes)
2. Keyword analysis by Match Length
3. Keyword comparison between early vs. late stages of a match
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_excel(r"cleanedTrain_output.xlsx")

# Standardize column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Parse dates
df['completion_date'] = pd.to_datetime(df['completion_date'])
df['match_activation_date'] = pd.to_datetime(df['match_activation_date'])
df['match_closure_meeting_date'] = pd.to_datetime(df['match_closure_meeting_date'])

# -----------------------
# 1. CADENCE ANALYSIS
# -----------------------

# Calculate days between calls per match
df_sorted = df.sort_values(by=['match_id_18char', 'completion_date']).copy()
df_sorted['days_between'] = df_sorted.groupby('match_id_18char')['completion_date'].diff().dt.days

# Compute average cadence per match
cadence_df = df_sorted.groupby('match_id_18char').agg({
    'days_between': 'mean',
    'match_length': 'first'
}).dropna()

# Remove zero or negative values to avoid issues with log
cadence_df = cadence_df[(cadence_df['days_between'] > 0) & (cadence_df['match_length'] > 0)]

# Apply log transformation
cadence_df['log_days_between'] = np.log(cadence_df['days_between'])
cadence_df['log_match_length'] = np.log(cadence_df['match_length'])

# Plotting log-transformed cadence vs. match length
sns.scatterplot(data=cadence_df, x='log_days_between', y='log_match_length')
plt.title("Log(Average Days Between Notes) vs. Log(Match Length)")
plt.xlabel("Log(Average Days Between Notes)")
plt.ylabel("Log(Match Length)")
plt.grid(True)
plt.show()

# Correlation (log-transformed)
print("Correlation between log-transformed cadence and match length:")
print(cadence_df[['log_days_between', 'log_match_length']].corr())

# -----------------------
# 2. KEYWORD ANALYSIS BY MATCH LENGTH
# -----------------------

notes_df = df[['match_length', 'match_support_contact_notes']].dropna()
median_length = notes_df['match_length'].median()
notes_df['match_type'] = np.where(notes_df['match_length'] >= median_length, 'long', 'short')

vectorizer = CountVectorizer(stop_words='english', max_features=150)
X = vectorizer.fit_transform(notes_df['match_support_contact_notes'])
word_counts = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_counts['match_type'] = notes_df['match_type'].values

word_summary = word_counts.groupby('match_type').mean().T
word_summary['difference'] = word_summary['long'] - word_summary['short']
word_summary_sorted = word_summary.sort_values('difference', ascending=False)

print("\\nTop 10 Keywords in Long Matches:")
print(word_summary_sorted.head(10))

print("\\nTop 10 Keywords in Short Matches:")
print(word_summary_sorted.tail(10))

#------------------------
# Generate top ten key words
#---------------------------
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get top 10 keywords used more in long matches
top10_long = word_summary_sorted[word_summary_sorted['difference'] > 0].head(10)

# Get top 10 keywords used more in short matches
top10_short = word_summary_sorted[word_summary_sorted['difference'] < 0].tail(10)

# Combine the two sets
top20_keywords = pd.concat([top10_long, top10_short])

# Create frequency (size) and color (group) mappings
word_freq = top20_keywords['difference'].abs().to_dict()
word_signs = top20_keywords['difference'].apply(lambda x: 'long' if x > 0 else 'short').to_dict()

# Define color function
def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return 'blue' if word_signs.get(word) == 'long' else 'red'

# Generate word cloud
wc = WordCloud(width=800, height=400, background_color='white',
               color_func=color_func, prefer_horizontal=1.0)

wc.generate_from_frequencies(word_freq)

# Display word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Top 10 Keywords: Long (blue) vs Short (red)", fontsize=16)
plt.tight_layout()
plt.show()


# -----------------------
# 3. EARLY VS LATE STAGE KEYWORD ANALYSIS
# -----------------------

# Keep necessary columns
timeline_df = df[['match_id_18char', 'completion_date', 'match_activation_date', 'match_closure_meeting_date', 'match_support_contact_notes']]
timeline_df = timeline_df.dropna()

# Calculate days into the match
timeline_df['days_into_match'] = (timeline_df['completion_date'] - timeline_df['match_activation_date']).dt.days

# Label as early or late using a 90-day cutoff
timeline_df['stage'] = pd.cut(
    timeline_df['days_into_match'], 
    bins=[-float('inf'), 90, float('inf')], 
    labels=['early', 'late']
)

# Keyword analysis for early vs. late
vectorizer_stage = CountVectorizer(stop_words='english', max_features=150)
X_stage = vectorizer_stage.fit_transform(timeline_df['match_support_contact_notes'])
stage_counts = pd.DataFrame(X_stage.toarray(), columns=vectorizer_stage.get_feature_names_out())
stage_counts['stage'] = timeline_df['stage'].values

stage_summary = stage_counts.groupby('stage').mean().T
stage_summary['difference'] = stage_summary['early'] - stage_summary['late']
stage_summary_sorted = stage_summary.sort_values('difference', ascending=False)

print("\\nTop 10 Keywords in Early Stage:")
print(stage_summary_sorted.head(10))

print("\\nTop 10 Keywords in Late Stage:")
print(stage_summary_sorted.tail(10))

#visualization for early and late 
import matplotlib.pyplot as plt

# Select top 10 keywords more common in early and late stages
top_early = stage_summary_sorted.head(10)
top_late = stage_summary_sorted.tail(10)

# Combine for plotting
combined = pd.concat([top_early, top_late])

# Set color based on direction of difference
colors = combined['difference'].apply(lambda x: 'green' if x > 0 else 'purple')

# Plot diverging bar chart
plt.figure(figsize=(10, 6))
combined['difference'].plot(kind='barh', color=colors)
plt.axvline(0, color='gray', linestyle='--')
plt.title('Keyword Usage Difference: Early vs Late Stage Matches')
plt.xlabel('Usage Difference (Early - Late)')
plt.ylabel('Keyword')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
