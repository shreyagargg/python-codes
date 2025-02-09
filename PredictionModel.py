import seaborn as sb
import matplotlib.pyplot as mpl
import pandas as p
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
import missingno as ms
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix


d = p.read_csv('C:/Users/shrey/Downloads/archive (1).zip')
df = d.head(7)
# print(df.transpose())
s = d.transpose()
# print(s)
# print(dir(sb))
# d.info()  #category / datatype
# p.set_option('display.float_format', lambda x: '%.3f'%x)
# print(d.describe().transpose())
x = d.drop(['education'], axis=1)
# x = x.transpose()
# print(x)
# print(x.describe().transpose())
# print(x.shape)
# print(x.isnull())
# print(x.isnull().sum()/ len(x)*100)
#  how to visualise relative missingness of col
# ms.bar(x)
# ms.heatmap(x)
# mpl.show()
x['glucose'] = x['glucose'].fillna(x['glucose'].mean())
x['totChol'] = x['totChol'].fillna(x['totChol'].mean())
x['BPMeds'] = x['BPMeds'].fillna(x['BPMeds'].mean())


''' totChol           1.180
sysBP             0.000
diaBP             0.000
BMI               0.448
heartRate         0.024
glucose           9.155 '''
# drop still missing values
x = x.dropna()
# print((x.isnull().sum())*100/ len(x))
#  dplicate values
# print(x.duplicated())
# ms.heatmap(x)
# mpl.show()
''' Transformation '''
# print(x.info())
#  text to int = no change
# print(x['age'].unique())
# print(x['age'].value_counts(bins=10, ascending=True))

#  label encoder (creating instance)
le = LabelEncoder()
# print(x['male'].value_counts())

#  date text to pandas datetime
# x['age'] = p.to_datetime(x['age'])
# print(x)

#  explore
# print(x['TenYearCHD'].value_counts(normalize=True))
# print(x.groupby('male')['TenYearCHD'].mean().plot(kind='bar'))
# x['male'] = x['male'].rename(gender')
# print(x.info())
# x['male'] = x['male'].replace([0,1], ['female','male'])
# x['Gender'] = x['male'].replace([0,1], ['female','male'])

# sb.catplot(data= x ,y = 'TenYearCHD' , x = 'Gender' , kind='bar')
# mpl.show()
# print(x.info())
# print(x['Gender'].value_counts())


x.corr()
sb.heatmap(x.corr())
# x.boxplot(grid=False)
mpl.show()
# target TenYearCHD
# print(x['TenYearCHD'].value_counts(normalize=True))
z = x.drop(['TenYearCHD'],axis=1)
# print(z.transpose())
#  split into x & y
X = z
# print(X.shape)
Y = x['TenYearCHD']
# print(Y.shape)
# smote synthetic minority oversampling
# for small data

os = SMOTE(random_state=0)
x_os,y_os=os.fit_resample(X,Y)
# print(x_os.shape,Y_os.shape)
# print(Y_os.value_counts())
# sb.countplot(x = y_os)
# mpl.show()
df_os = p.DataFrame(x_os)
df_os['TenYearCHD'] = y_os
# sb.scatterplot(data= df_os, x='BMI', y = 'age', hue= 'TenYearCHD')
sb.scatterplot(data= df_os, x='heartRate', y = 'age', hue= 'TenYearCHD')


# mpl.show()
# dividing the data into training and testing test
x_train, x_test ,y_train, y_test = train_test_split(x_os,y_os, test_size= 0.3,random_state=0)
# #  scaling values
sc_train = StandardScaler().fit(x_train)
x_train_sc = sc_train.transform(x_train)
# np.set_printoptions(precision=3)
# # print(x_train_sc[0:5,:])

# # training
# lr = LogisticRegression(solver='liblinear')
# lr.fit(x_train_sc, y_train)
# #  testing
# sc_test = StandardScaler().fit(x_test)
# x_test_sc = sc_test.transform(x_test)

# result = lr.score(x_test_sc,y_test)
# # result*100
# print(result*100)

# result = lr.score(x_train_sc,y_train)
# # result*100
# print(result*100)
c = ['blue','aqua','magenta','gold','seagreen','turquoise']
z = x.drop(['TenYearCHD'], axis=1)

# z.corrwith(x['TenYearCHD']).plot.bar(title='Co-relation with CHD',color=c)
# mpl.show()

lr = LogisticRegression(C=1.0,class_weight='balanced',dual=False,fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None,
                        max_iter=3000,multi_class='auto',n_jobs=None,penalty='l2'
                        ,random_state=1234, solver='lbfgs',tol=0.0001,verbose=0,
                        warm_start=False)

model1 = lr.fit(x_train, y_train)
prediction1 = model1.predict(x_test)
cm = confusion_matrix(y_test, prediction1)
TP = cm[0][0]
TN = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

print('Model accuracy :', (TP+TN)/(TP+TN+FN+FP))
print('Model sensitivity :', (TP/(TP+FN)))
print('Model specificity:', (TN/(TN+FP)))
print('Model precision :', (TP/(TP+FP)))

# sb.heatmap(cm,annot=True, cmap='winter', linewidths=0.3,
#            linecolor='blue', annot_kws={'size':30})

# mpl.show()


# from sklearn.linear_model import LogisticRegression
# import numpy as np

# Assume X contains your features and y contains your labels

# Create a logistic regression model
model = LogisticRegression()

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
scores = cross_val_score(model, x_train, y_train, cv=kf)
print("Cross-validation scores:", scores)

# Calculate the mean accuracy across all folds
mean_accuracy = np.mean(scores)
print("Mean accuracy:", mean_accuracy)

mpl.show()
try:
    print('Enter your details to get prediction :- ')
    gender = int(input('Your Gender '))
    a = int(input('Your age:- '))
    cs = int(input('Do you smoke ?? '))
    cig = int(input('Number of ciggartes(if you are a smoker) :'))
    bp = int(input('do you take BP medicines :'))
    stroke = int(input('Do you ever feel any prevalent stroke '))
    hyp = int(input('Do you ever feel any Hypertension : '))
    dia = int(input('Are you diabetic : '))
    totChol = int(input('Your cholestrol : '))
    sysBP = int(input('your systolic BP : '))
    diaBP = int(input('your distolic BP : '))
    bmi = int (input('Your BMI : '))
    heartRate = int (input('Your heart rate '))
    glucose = int (input('Your glucose level ')) 
    ip = [[gender,a,cs,cig,bp,stroke,hyp,dia,totChol,sysBP,diaBP,bmi,heartRate,glucose]]
    ip = [[a,gender]]
    ip = np.array(ip)
# print(type(ip))
# print(np.shape(ip))
    # ip = ip.reshape(-1,1)
# print(n)
    m = lr.fit(x_train, y_train)
    predict = m.predict(ip)


    if(predict == 1):
        print('You may have heart disease :(')
    else:
        print('no risk :)')

except Exception as e:
    print(e)
mpl.show()