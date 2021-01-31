# Ian Wallace
# CAP 5610 - Spring 2021
# Homework 1ana

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Import datasets
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')
combine = [train_df, test_df]


# Define figure and axes
fig, ax = plt.subplots()

# Hide the axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')

# Question 5
print("Question 5")
print("==========")
print("Null values in training data")
print(train_df.columns[train_df.isnull().any()])
print("Null values in test data")
print(test_df.columns[test_df.isnull().any()])
print()

print("Question 7")
print("==========")
print("Descriptive statistics of test data")
train_df_describe = train_df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].describe().round(4)
print(train_df_describe)

des_labels = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

table = ax.table(cellText=train_df_describe.values, colLabels=train_df_describe.columns, rowLabels=des_labels, loc='center')
table.set_fontsize(12)
table.scale(1,2)

plt.savefig('q7.svg')
plt.savefig('q7.png')
#plt.show()
print()

# Question 8
print("Question 8")
print("==========")
cats = ['Survived', 'Pclass', 'Sex', 'Embarked']
train_df['Survived'] = pd.Categorical(train_df.Survived)
train_df['Pclass'] = pd.Categorical(train_df.Pclass)
train_df['Sex'] = pd.Categorical(train_df.Sex)
train_df['Embarked'] = pd.Categorical(train_df.Embarked)

# Get descriptive statistics for each column
print("Survived")
print(train_df.Survived.describe())

print("\nPclass")
print(train_df.Pclass.describe())

print("\nSex")
print(train_df.Sex.describe())

print("\nEmbarked")
print(train_df.Embarked.describe())

# Define figure and axes
fig, ax = plt.subplots()

# Hide the axes
fig.patch.set_visible(False)
ax.axis('off')
ax.axis('tight')


# Create table
des_labels = ['count', 'unique', 'top', 'freq']
train_df_describe = train_df[['Survived', 'Pclass', 'Sex', 'Embarked']].describe().round(4)
table = ax.table(cellText=train_df_describe.values, colLabels=train_df_describe.columns, rowLabels=des_labels, loc='center')
table.set_fontsize(12)
table.scale(1,2)

plt.savefig('q8.svg')
plt.savefig('q8.png')
print()

# Question 9
print("Question 9")
print("==========")

pclass_survived = list(zip(train_df['Pclass'], train_df['Survived']))

# Caculate the average survival value for each class
def avg_survived_value_pclass(pclass, data):
    total_p = 0
    num_survived = 0
    for pair in data:    
        if (pair[0] == pclass):
            total_p += 1

            if (pair[1] == 1):
                num_survived += 1

    return num_survived/total_p

print("Average Survived Values (by Pclass)")
print("Class 1:", avg_survived_value_pclass(1, pclass_survived))
print("Class 2:", avg_survived_value_pclass(2, pclass_survived))
print("Class 3:", avg_survived_value_pclass(3, pclass_survived))

print("Yes we should include this in the model.")
print()

# Question 10
print("Question 10")
print("===========")

sex_survived = list(zip(train_df['Sex'], train_df['Survived']))

# Caculate the average survival value for each sex
def avg_survived_value_sex(sex, data):
    total_s = 0
    num_survived = 0
    for pair in data:    
        if (pair[0] == sex):
            total_s += 1

            if (pair[1] == 1):
                num_survived += 1

    return num_survived/total_s

# Print results
print("Average Survived Values (by Sex)")
print("Male:", avg_survived_value_sex('male', sex_survived))
print("Female:", avg_survived_value_sex('female', sex_survived))

print("Women are more likely to survive.")
print()

# Question 11
print("Question 11")
print("===========")

# Create graph of age vs number survived/died
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

survived0 = train_df.loc[ (train_df.Survived == 0) ]
survived1 = train_df.loc[ (train_df.Survived == 1) ]

survived0['Age'].plot(ax=ax1, kind='hist')
survived1['Age'].plot(kind='hist', ax=ax2)
ax1.set_title('Survived = 0')
ax2.set_title('Survived = 1')

ax1.set(xlabel='Age', ylabel='Number Died')
ax2.set(xlabel='Age', ylabel='Number Survived')

plt.xlabel('Age')
plt.savefig('q11.svg')
#plt.show()
print()

# Qusetion 12
print("Question 12")
print("===========")

fig, axs = plt.subplots(3, 2, sharey=True, sharex=True)

fig.tight_layout()

# Filter Data
p1s0 = train_df.loc[ (train_df.Survived == 0) & (train_df.Pclass == 1) ]
p2s0 = train_df.loc[ (train_df.Survived == 0) & (train_df.Pclass == 2) ]
p3s0 = train_df.loc[ (train_df.Survived == 0) & (train_df.Pclass == 3) ]
p1s1 = train_df.loc[ (train_df.Survived == 1) & (train_df.Pclass == 1) ]
p2s1 = train_df.loc[ (train_df.Survived == 1) & (train_df.Pclass == 2) ]
p3s1 = train_df.loc[ (train_df.Survived == 1) & (train_df.Pclass == 3) ]

# Create histograms
p1s0['Age'].plot(ax=axs[0, 0], kind='hist')
p2s0['Age'].plot(ax=axs[1, 0], kind='hist')
p3s0['Age'].plot(ax=axs[2, 0], kind='hist')
p1s1['Age'].plot(ax=axs[0, 1], kind='hist')
p2s1['Age'].plot(ax=axs[1, 1], kind='hist')
p3s1['Age'].plot(ax=axs[2, 1], kind='hist')

# Set suplot titles
axs[0, 0].set_title('Pclass = 1 | Survived = 0')
axs[1, 0].set_title('Pclass = 2 | Survived = 0')
axs[2, 0].set_title('Pclass = 3 | Survived = 0')
axs[0, 1].set_title('Pclass = 1 | Survived = 1')
axs[1, 1].set_title('Pclass = 2 | Survived = 1')
axs[2, 1].set_title('Pclass = 3 | Survived = 1')

plt.savefig('q12.svg')
#plt.show()
print()

print("Question 13")
print("===========")

# Set up subplots
fig, axs = plt.subplots(3, 2)

# Filter Data
g1 = train_df.loc[ (train_df.Survived == 0) & (train_df.Embarked == 'S') ]
g2 = train_df.loc[ (train_df.Survived == 0) & (train_df.Embarked == 'C') ]
g3 = train_df.loc[ (train_df.Survived == 0) & (train_df.Embarked == 'Q') ]
g4 = train_df.loc[ (train_df.Survived == 1) & (train_df.Embarked == 'S') ]
g5 = train_df.loc[ (train_df.Survived == 1) & (train_df.Embarked == 'C') ]
g6 = train_df.loc[ (train_df.Survived == 1) & (train_df.Embarked == 'Q') ]

# Seperate male and female averages
g1_male_avg_fare = g1.loc[ g1.Sex == 'male'].Fare.mean()
g1_female_avg_fare = g1.loc[ g1.Sex == 'female'].Fare.mean()

g2_male_avg_fare = g2.loc[ g2.Sex == 'male'].Fare.mean()
g2_female_avg_fare = g2.loc[ g2.Sex == 'female'].Fare.mean()

g3_male_avg_fare = g3.loc[ g3.Sex == 'male'].Fare.mean()
g3_female_avg_fare = g3.loc[ g3.Sex == 'female'].Fare.mean()

g4_male_avg_fare = g4.loc[ g4.Sex == 'male'].Fare.mean()
g4_female_avg_fare = g4.loc[ g4.Sex == 'female'].Fare.mean()

g5_male_avg_fare = g5.loc[ g5.Sex == 'male'].Fare.mean()
g5_female_avg_fare = g5.loc[ g5.Sex == 'female'].Fare.mean()

g6_male_avg_fare = g6.loc[ g6.Sex == 'male'].Fare.mean()
g6_female_avg_fare = g6.loc[ g6.Sex == 'female'].Fare.mean()

# Set up data to be plotted
x = ['male', 'female']
y1 = [g1_male_avg_fare, g1_female_avg_fare]
y2 = [g2_male_avg_fare, g2_female_avg_fare]
y3 = [g3_male_avg_fare, g3_female_avg_fare]
y4 = [g4_male_avg_fare, g4_female_avg_fare]
y5 = [g5_male_avg_fare, g5_female_avg_fare]
y6 = [g6_male_avg_fare, g6_female_avg_fare]

# Generate plots
plt.subplot(3, 2, 1)
plt.bar(x, y1)
plt.yticks([0, 20, 40, 60, 80])
plt.title('Embarked = S | Survived = 0')
plt.ylabel('Fare')

plt.subplot(3, 2, 3)
plt.bar(x, y2)
plt.yticks([0, 20, 40, 60, 80])
plt.title('Embarked = C | Survived = 0')
plt.ylabel('Fare')

plt.subplot(3, 2, 5)
plt.bar(x, y3)
plt.yticks([0, 20, 40, 60, 80])
plt.title('Embarked = Q | Survived = 0')
plt.xlabel('Sex')
plt.ylabel('Fare')

plt.subplot(3, 2, 2)
plt.bar(x, y4)
plt.yticks([0, 20, 40, 60, 80])
plt.title('Embarked = S | Survived = 1')

plt.subplot(3, 2, 4)
plt.bar(x, y5)
plt.yticks([0, 20, 40, 60, 80])
plt.title('Embarked = C | Survived = 1')

plt.subplot(3, 2, 6)
plt.bar(x, y6)
plt.yticks([0, 20, 40, 60, 80])
plt.title('Embarked = Q | Survived = 1')
plt.xlabel('Sex')

# Save figure and show plot
plt.savefig('q13.svg')
plt.tight_layout()

#plt.show()
print()

print("Question 14")
print("===========")

# Calculate the number of duplicate values in ticket column
num_duplicates = train_df.pivot_table(index=['Ticket'], aggfunc='size')
num_duplicates = len(num_duplicates)

# Calculate total number of values in Ticket
total_tickets = len(train_df['Ticket'])

# Print Results
print("Number of duplicate:", num_duplicates)
print("Number of tickets:", total_tickets)
print("Duplicate rate:", num_duplicates/total_tickets)
print()

# Question 15
print("Question 15")
print("===========")

# Count null values in the Cabin column
train_cabin_null = train_df['Cabin'].isnull().sum()
test_cabin_null = test_df['Cabin'].isnull().sum()

# Calculate total number of values in both train and test sets
total_values = len(train_df['Cabin'].index) + len(test_df['Cabin'].index)

# Print results
print("Null values in training set (Cabin):", train_cabin_null)
print("Null values in test set (Cabin):", test_cabin_null)
print("Total null values in Cabin column:", train_cabin_null + test_cabin_null)
print("Total values in Cabin column:", total_values)

print( (((train_cabin_null + test_cabin_null)/total_values) * 100).round(2), "percent of the cabin values are null." )
print()

# Question 16
print("Question 16")
print("===========")

# Replace male and female with 0 and 1 respectivly
train_df['Sex'].replace('male', 0, inplace=True)
train_df['Sex'].replace('female', 1, inplace=True)

# Rename the Sex column to Gender
train_df.rename(columns={'Sex':'Gender'}, inplace=True)

# Show results
print(train_df.Gender)
print()

# Question 17
print("Question 17")
print("===========")

# Calculate the mean and std of Age
age_mean = train_df.Age.mean()
age_std = train_df.Age.std()

# Get random age within 1 std of mean
def get_random_age(mean, std):

    lower = int(mean - std)
    upper = int(mean + std)

    return random.randint(lower, upper)

# Show null values
print("Null values in age (before):", train_df['Age'].isnull().sum())

# Replace all null values in age column with random age values within 1 std of mean
train_df['Age'] = train_df['Age'].fillna( get_random_age(age_mean, age_std) )

# Show that null values have been filled in
print("Null values in age (after):", train_df['Age'].isnull().sum())
print()

# Question 18
print("Question 18")
print("===========")

# Show null values
print("Null values in embarked (before):", train_df['Embarked'].isnull().sum())

# Get the most common Embarked value
mode = train_df['Embarked'].mode()
print("The most common embarked value is:")
print(mode)

# Replace all null values in embarked column with S, it is the most common destination
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# Show that null values have been filled in
print("Null values in embarked (after):", train_df['Embarked'].isnull().sum())
print()

# Question 19
print("Question 19")
print("===========")

# Show null values
print("Null values in fare (before):", test_df['Fare'].isnull().sum())

# Calculate the most common fare value
test_mode = float(test_df['Fare'].mode())
print("Mode of fare column in test dataset:", test_mode)

# Replace all null values in fare column with the mode of the column
test_df['Fare'] = test_df['Fare'].fillna(test_mode)

# Show that null values have been filled in
print("Null values in fare (after):", test_df['Fare'].isnull().sum())
print()

# Quesiton 20
print("Question 20")
print("===========")

# Ordinal 0
train_df['Fare'].loc[ (train_df['Fare'] > -0.001) & (train_df['Fare'] <= 7.91) ] = 0
# Ordinal 1
train_df['Fare'].loc[ (train_df['Fare'] > 7.91) & (train_df['Fare'] <= 14.454) ] = 1
# Ordinal 2
train_df['Fare'].loc[ (train_df['Fare'] > 14.454) & (train_df['Fare'] <= 31.0) ] = 2
# Ordinal 3
train_df['Fare'].loc[ (train_df['Fare'] > 31.0)] = 3

# Convert the Fare column to Category instead of float
train_df['Fare'] = train_df['Fare'].astype('int')
train_df['Fare'] = train_df['Fare'].astype('category')

# Print descriptive satistics
print(train_df['Fare'].describe())
print()