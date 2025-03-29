import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
BBBS_Novice_df = pd.read_excel("Novice_2.xlsx")
BBBS_Novice_df["Closure Initiator"] = BBBS_Novice_df["Closure Initiator"].astype("string")
BBBS_Novice_df["Closure Reason"] = BBBS_Novice_df["Closure Reason"].astype("string")
print(BBBS_Novice_df["Closure Reason"].dtype)

#Match Length Distribution
plt.hist(BBBS_Novice_df["Match Length"], bins = 40)
plt.title("Distribution of Match Length")
plt.xlabel("Match Length in Months")
plt.ylabel("Frequency")

plt.show()

#Closure Reason Distribution
Closure_Reason_Counts = BBBS_Novice_df["Closure Reason"].value_counts().head(15)

plt.figure(figsize=(10, 10))
plt.bar(Closure_Reason_Counts.index, Closure_Reason_Counts.values)
plt.xlabel('Closure Reason')
plt.xticks(rotation = 45, ha="right", fontsize = 8)
plt.subplots_adjust(bottom=0.3)
plt.ylabel('Frequency')
plt.title('Distribution of Closure Reasons')
plt.show()

#Match Length by program type
plt.figure(figsize=(12, 6))
sns.boxplot(data=BBBS_Novice_df, x='Program Type', y='Match Length')
plt.title('Match Length Distribution by Program Type')
plt.xlabel('Program Type')
plt.ylabel('Match Duration (Months)')
plt.xticks(rotation=0, fontsize = 8)
plt.show()

plt.figure(figsize=(10,8))
mean_match_len = sns.barplot(data=BBBS_Novice_df, x='Program Type', y='Match Length', errorbar=None)
plt.title('Average Match Length by Program Type', fontsize=14, pad=20)
plt.xlabel('Program Type', fontsize=12)
plt.ylabel('Average Match Length (Months)', fontsize=12)
plt.xticks(ha='center')
for i in mean_match_len.patches:
    mean_match_len.annotate(f"{i.get_height():.1f}",
                (i.get_x() + i.get_width() / 2., i.get_height()),
                ha='center', va='center',
                xytext=(0, 10),
                textcoords='offset points')
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=BBBS_Novice_df, x= "Closure Reason", y='Match Length')
plt.title('Match Length Distribution by Program Type')
plt.xlabel('Program Type')
plt.ylabel('Match Duration (Months)')
plt.xticks(rotation=45, ha="right", fontsize = 8)
plt.show()

contingency_table = pd.crosstab(BBBS_Novice_df['Program Type'], BBBS_Novice_df['Closure Reason'])

# Plot stacked bar chart
contingency_table.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Closure Reasons by Program Type')
plt.xlabel('Program Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Closure Reason', bbox_to_anchor=(1.05, 1))
plt.show()

#Closures by Volunteer
Volunteer_Closures = BBBS_Novice_df[BBBS_Novice_df["Closure Initiator"].str.contains("Volunteer")]
Volunteer_contingency_table = pd.crosstab(Volunteer_Closures['Program Type'], Volunteer_Closures['Closure Reason'])
Volunteer_contingency_table.plot(kind='bar', stacked=False, figsize=(16, 6))
plt.title('Closures initiated by Volunteer by Program Type')
plt.xlabel('Program Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Closure Reason', bbox_to_anchor=(1.05, 1))
plt.show()

#Closures by child and/or family
Family_Closures = BBBS_Novice_df[BBBS_Novice_df["Closure Initiator"].str.contains("Child", "Child/Family")]
Family_contingency_table = pd.crosstab(Family_Closures['Program Type'], Family_Closures['Closure Reason'])
Family_contingency_table.plot(kind='bar', stacked=False, figsize=(16, 6))
plt.title('Closures initiated by Child and/or Family by Program Type')
plt.xlabel('Program Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Closure Reason', bbox_to_anchor=(1.05, 1))
plt.show()

#Closures by agency
Agency_Closures = BBBS_Novice_df[BBBS_Novice_df["Closure Initiator"].str.contains("Agency")]
Agency_contingency_table = pd.crosstab(Agency_Closures['Program Type'], Agency_Closures['Closure Reason'])
Agency_contingency_table.plot(kind='bar', stacked=False, figsize=(16, 6))
plt.title('Closures initiated by Agency by Program Type')
plt.xlabel('Program Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Closure Reason', bbox_to_anchor=(1.05, 1))
plt.show()



