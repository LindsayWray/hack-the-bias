# # Import Pandas Library, used for data manipulation
# # Import matplotlib, used to plot our data
# # Import nump for mathemtical operations
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


pd.set_option("display.max_colwidth", 10000)


# read data
data = pd.read_csv("survey_results_public_mega_inc.csv")

# print(data['Gender'][0:30])

# removing NaN from Gender and Salary question answers
data = data[~data.Gender.isin([float('nan')])]
data = data[~data.ConvertedSalary.isin([float('nan')])]
data = data[~data.Age.isin([float('nan')])]
data = data[~data.FormalEducation.isin([float('nan')])]

def age_to_int(age):
	if age == "Under 18 years old":
		return 18
	if age == "18 - 24 years old":
		return 21
	if age == "25 - 34 years old":
		return 30
	if age == "35 - 44 years old":
		return 40
	if age == "45 - 54 years old":
		return 50
	if age == "55 - 64 years old":
		return 60
	if age == "65 years or older":
		return 65
	else:
		print(age)
		return (0)

def edu_to_int(edu):
	if edu == "I never completed any formal education":
		return 0
	if edu == "Primary/elementary school":
		return 1
	if edu == "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)":
		return 2
	if edu == "Some college/university study without earning a degree":
		return 3
	if edu == "Bachelor’s degree (BA, BS, B.Eng., etc.)":
		return 4
	if edu == "Master’s degree (MA, MS, M.Eng., MBA, etc.)":
		return 5
	if edu == "Associate degree":
		return 6
	if edu == "Professional degree (JD, MD, etc.)":
		return 7
	if edu == "Other doctoral degree (Ph.D, Ed.D., etc.)":
		return 7
	else:
		print(edu)
		return (0)

# group non male and female into other, and set dtype to string
data.Gender = data.Gender.apply(lambda x: x if x in ['Male', 'Female'] else 'other')
data.Gender = data.Gender.astype("string")

# floats to integers
data.ConvertedSalary = data.ConvertedSalary.astype(int)
# print(data.ConvertedSalary)

# gender to integers
data.Gender = data.Gender.apply(lambda x: 0 if x == "Male" else 1)
data['constant'] = 1

data.Age = data.Age.apply(age_to_int)

data.FormalEducation = data.FormalEducation.apply(edu_to_int)

x = data[['Gender', 'Age', 'FormalEducation']]
y = data['ConvertedSalary']

model = LinearRegression()
model.fit(x, y)

jane = [1, 32, 4]
john = [0, 32, 4]
salary_pred = model.predict([jane, john])

print("Jane (Age 32; BSc)", salary_pred[0].round(2), "; John (Age 32; BSc)", salary_pred[1].round(2))

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x1 = x["Age"]
x2 = x["FormalEducation"]
ax.scatter(x1, x2, y, c='r', marker='o')

ax.set_xlabel('Age')
ax.set_ylabel('FormalEducation')
ax.set_zlabel('Salary')

plt.show()