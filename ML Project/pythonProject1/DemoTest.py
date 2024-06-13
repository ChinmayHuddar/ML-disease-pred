from tkinter import *
import  csv

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings("ignore")

window = Tk()

age_var = IntVar()
gender_var = StringVar()
weight_var = IntVar()
height_var = IntVar()
blood_pressure_var = StringVar()
cholesterol_var = IntVar()
disease_var = StringVar()
cough_var = StringVar()
fatigue_var = StringVar()
difficulty_breathing_var = StringVar()


Age = Label(window,text="Age:",font=('Arial',20,'bold'),bg="#CED5FF")
Age.place(x=0,y=10)

age = Entry(window,textvariable=age_var).place(x=400,y=20)

gen = Label(window,text="Gender:",font=('Arial',20,'bold'),bg="#CED5FF")
gen.place(x=0,y=50)

gender = Entry(window,textvariable=gender_var).place(x=400,y=60)

wei = Label(window,text="Weight:",font=('Arial',20,'bold'),bg="#CED5FF")
wei.place(x=0,y=100)

weight = Entry(window,textvariable=weight_var).place(x=400,y=110)

hei = Label(window,text="Height:",font=('Arial',20,'bold'),bg="#CED5FF")
hei.place(x=0,y=150)

height = Entry(window,textvariable=height_var).place(x=400,y=160)


bp = Label(window,text="Blood Pressure:",font=('Arial',20,'bold'),bg="#CED5FF")
bp.place(x=0,y=200)

blood_pressure = Entry(window,textvariable=blood_pressure_var).place(x=400,y=210)

ch = Label(window,text="Cholesterol:",font=('Arial',20,'bold'),bg="#CED5FF")
ch.place(x=0,y=250)

cholesterol = Entry(window,textvariable=cholesterol_var).place(x=400,y=260)

dis = Label(window,text="Diseases:",font=('Arial',20,'bold'),bg="#CED5FF")
dis.place(x=0,y=300)

disease = Entry(window,textvariable=disease_var).place(x=400,y=310)


coug = Label(window,text="Cough:",font=('Arial',20,'bold'),bg="#CED5FF")
coug.place(x=0,y=350)

cough = Entry(window,textvariable=cough_var).place(x=400,y=360)

Fati = Label(window,text="Fatigue:",font=('Arial',20,'bold'),bg="#CED5FF")
Fati.place(x=0,y=400)

fatigue = Entry(window,textvariable=fatigue_var).place(x=400,y=410)






df = Label(window,text="Difficulty Breathing:",font=('Arial',20,'bold'),bg="#CED5FF")
df.place(x=0,y=450)

result = Label(window,text="Suggested Medication for disease:",font=('Arial',20,'bold'),bg="#CED5FF")
result.place(x=0,y=530)

difficulty_breathing = Entry(window,textvariable=difficulty_breathing_var).place(x=400,y=450)


"""option = [
    "Low",
    "Medium",
    "High"
]"""
def show():
   """ myLable = Label(window,text=clicked.get()).pack()"""
   age = Label(window,text=age_var.get()).place(x=700,y=20)
   gender = Label(window, text=gender_var.get()).place(x=700,y=60)
   weight = Label(window, text=weight_var.get()).place(x=700,y=110)
   height = Label(window, text=height_var.get()).place(x=700,y=160)
   blood_pressure = Label(window, text=blood_pressure_var.get()).place(x=700,y=210)
   cholesterol = Label(window, text=cholesterol_var.get()).place(x=700,y=260)
   disease = Label(window, text=disease_var.get()).place(x=700,y=310)
   cough = Label(window, text=cough_var.get()).place(x=700,y=360)
   fatigue = Label(window, text=fatigue_var.get()).place(x=700,y=410)
   difficulty_breathing = Label(window, text=difficulty_breathing_var.get()).place(x=700,y=460)

   df = pd.read_csv(r'E:/TY Sem 2/ML/ML Project/patients.csv')

   df['Height '] = df['Height '] / 100

   df['BMI'] = df['Weight'] / pow(df['Height '], 2)

   # BMI
   def categorize_BMI(bmi):
      if bmi < 18.5:
         return 'Underweight'
      elif bmi >= 18.5 and bmi < 26.5:
         return 'Normal'
      elif bmi >= 26.5 and bmi < 29.5:
         return 'Overweight'
      else:
         return 'Obese'

   df['BMI_Category'] = df.apply(lambda x: categorize_BMI(x['BMI']), axis=1)

   # Blood Pressure
   df[['Systolic BP', 'Diastolic BP']] = df['Blood Pressure (mmHg)'].str.split("/", expand=True)
   df.drop('Blood Pressure (mmHg)', axis=1, inplace=True)

   def categorize_blood_pressure(systolic, diastolic):
      if int(systolic) >= 115 or int(diastolic) >= 90:
         return 'High'
      elif int(systolic) < 90 or int(diastolic) < 70:
         return 'Low'
      else:
         return 'Normal'

   df['BP_Category'] = df.apply(lambda x: categorize_blood_pressure(x['Systolic BP'], x['Diastolic BP']), axis=1)

   # Cholesterol
   def categorize_cholesterol(cholesterol):
      if cholesterol >= 220:
         return 'High'
      elif cholesterol < 160:
         return 'Low'
      else:
         return 'Normal'

   df['Cholesterol_Category'] = df.apply(lambda x: categorize_cholesterol(x['Cholesterol Level (mg/dl)']), axis=1)

   features = df.filter(
      ['Age', 'Gender', 'Disease', 'Cough', 'Fatigue', 'Difficulty Breathing', 'BP_Category', 'Cholesterol_Category',
       'BMI_Category'])
   categorical_features = features.select_dtypes(include=['object']).columns
   numerical_features = features.select_dtypes(include=['int64', 'float64']).columns
   print("\nCategorical Features:", categorical_features)
   print("\nNumerical Features:", numerical_features)

   label_encoder = LabelEncoder()
   for feature in categorical_features:
      features[feature] = label_encoder.fit_transform(features[feature])

   X = features
   y = df['Outcome Variable']

   scaler_x = StandardScaler()
   X = scaler_x.fit_transform(X)

   from sklearn.model_selection import train_test_split, GridSearchCV
   from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
   from sklearn.ensemble import RandomForestClassifier

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
   # X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 0)
   cv_params = {'n_estimators': [50, 100],
                'max_depth': [10, 50],
                'min_samples_leaf': [0.5, 1],
                'min_samples_split': [0.001, 0.01],
                'max_features': ["sqrt"],
                'max_samples': [.5, .9]}

   rf = RandomForestClassifier(random_state=0)
   rf_val = GridSearchCV(rf, cv_params, refit='f1', n_jobs=-1, verbose=1)
   rf_val.fit(X_train, y_train)

   rf_val.best_params_

   rf_opt = RandomForestClassifier(n_estimators=50, max_depth=10,
                                   min_samples_leaf=1, min_samples_split=0.001,
                                   max_features="sqrt", max_samples=0.9, random_state=0)
   rf_opt.fit(X_train, y_train)

   y_pred = rf_opt.predict(X_test)

   pc_test1 = precision_score(y_test, y_pred, pos_label="Positive")
   rc_test1 = recall_score(y_test, y_pred, pos_label="Positive")
   ac_test1 = accuracy_score(y_test, y_pred)
   f1_test1 = f1_score(y_test, y_pred, pos_label="Positive")

   test1 = pd.DataFrame(
      columns=['Age', 'Gender', 'Height', 'Weight', 'Blood Pressure', 'Cholesterol', 'Disease', 'Cough', 'Fatigue',
               'Difficulty Breathing'])

   test1 = test1._append(
      {'Age': age_var.get(), 'Gender': gender_var.get(), 'Height': height_var.get(), 'Weight': weight_var.get(), 'Blood Pressure': blood_pressure_var.get(),
       'Cholesterol': cholesterol_var.get(), 'Disease': disease_var.get(), 'Cough': cough_var.get(), 'Fatigue': fatigue_var.get(),
       'Difficulty Breathing': difficulty_breathing_var.get()}, ignore_index=True)

   # Data Transformation and Preprocessing
   test1[['Systolic BP', 'Diastolic BP']] = test1['Blood Pressure'].str.split("/", expand=True)
   test1.drop('Blood Pressure', axis=1, inplace=True)
   test1['Height'] = test1['Height'] / 100
   test1['BMI'] = test1['Weight'] / pow(test1['Height'], 2)
   test1.drop('Height', axis=1, inplace=True)
   test1.drop('Weight', axis=1, inplace=True)
   test1['BP_Category'] = test1.apply(lambda x: categorize_blood_pressure(x['Systolic BP'], x['Diastolic BP']), axis=1)
   test1['Cholesterol_Category'] = test1.apply(lambda x: categorize_cholesterol(x['Cholesterol']), axis=1)
   test1['BMI_Category'] = test1.apply(lambda x: categorize_BMI(x['BMI']), axis=1)
   test1.drop('Cholesterol', axis=1, inplace=True)
   test1.drop('Systolic BP', axis=1, inplace=True)
   test1.drop('Diastolic BP', axis=1, inplace=True)
   test1.drop('BMI', axis=1, inplace=True)
   features1 = test1.filter(
      ['Age', 'Gender', 'Disease', 'Cough', 'Fatigue', 'Difficulty Breathing', 'BP_Category', 'Cholesterol_Category',
       'BMI_Category'])
   label_encoder = LabelEncoder()
   for feature in features1:
      features1[feature] = label_encoder.fit_transform(features1[feature])

   y_pred1 = rf_opt.predict(features1)

   print("The patient tested ", rf_opt.predict(features1), " for ", test1.at[0, 'Disease'])
   print("BMI Level: ", test1.at[0, 'BMI_Category'])
   print("Blood Pressure Level: ", test1.at[0, 'BP_Category'])
   print("Cholesterol Level: ", test1.at[0, 'Cholesterol_Category'])


   csv_file =csv.reader(open('E:/TY Sem 2/ML/ML Project/Own Dataset.csv','r'))
   if y_pred1=="Positive":
      for row in csv_file:
          if disease_var.get()==row[0]:
            MyResult = Label(window,text=row).place(x=10,y=600)







"""clicked = StringVar()"""
"""clicked.set(option[0])"""

# drop down menu declared 1
"""dropdwonBp = OptionMenu(window,clicked,*option)
dropdwonBp.configure(width=30,bg="#CED5FF")
dropdwonBp.place(x=400,y=13)"""

myButton =Button(window, text="Selected", command=show)
myButton.place(x=350,y=500)

window.geometry("800x700")

window.configure(bg="#CED5FF")

window.mainloop()