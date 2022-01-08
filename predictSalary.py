import pandas as pd
from sklearn import linear_model

df = pd.read_csv('hiring.csv')
df.experience = df.experience.fillna(0)
df.test_score = df.test_score.fillna(df.test_score.median())

reg = linear_model.LinearRegression()
reg.fit(df[['experience', 'test_score', 'interview_score']],df.salary)
#2 yr experience, 9 test score, 6 interview score
predict1 = reg.predict([[2, 9, 6]])
print(predict1)
#12 yr experience, 10 test score, 10 interview score
predict2 = reg.predict([[12, 10, 10]])
print(predict2)