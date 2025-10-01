import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("C:/Users/Sargis Matevosyan/OneDrive/Desktop/Freelanse/Project_5/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# cuyc enq talis arajin 5 toxy
#print(df.head())

#tesnum enq inch popoxakanner unenq
#print(df.info())

# cuyc kta (count,mean, std,min 25%,50%,75%, max)
#print(df.describe())

#cuyc e talis arjeqnert % ov 
attr_counts = df['Attrition'].value_counts(normalize=True) * 100
print(attr_counts)

#ցույց է տալիս քանակային bar chart գրաֆիկը
plt.figure(figsize=(5,4))  # ստեղծում է նկարի էջ 5x4 չափերով։
sns.countplot(x='Attrition', data=df, palette='Set2') #x-axis = Attrition (Yes/No), palette='Set2 ընտրում է գույները
plt.title("Employee Attrition Distribution")
plt.show()

#--Սա ցույց կտա, որ հիմնականում երիտասարդ աշխատակիցներն են
#  (25–35 տարեկան) ավելի շատ դուրս գալիս։

plt.figure(figsize=(8,5))
sns.histplot(data=df, x="Age", hue="Attrition", multiple="stack", palette="Set1", bins=20)
plt.title("leaving job by Age")
plt.show()

#-կտեսնենք թե Որ բաժիններից են մարդիկ ավելի շատ դուրս գալիս։
plt.figure(figsize=(7,5))
sns.countplot(x="Department", hue="Attrition", data=df, palette="Set2")
plt.title("Attrition by Department")
plt.xticks(rotation=30)
plt.show()

#-կտեսնենք թե ավելի քիչ աշխատավարձ ստացողները ավելի հաճախ հեռացվոում թե ոչ

plt.figure(figsize=(8,5))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, palette="Set3")
plt.title("Attrition vs Monthly Income")
plt.show()
# ցածր աշխատավարձ ունեցողներն ավելի շատ են լքում։

#----
#Սովորաբար, նրանք ովքեր շատ OverTime են անում, ավելի մեծ հավանականությամբ են լքում աշխատանքը։
plt.figure(figsize=(5,4))
sns.countplot(x="OverTime", hue="Attrition", data=df, palette="Set1")
plt.title("Attrition vs OverTime")
plt.show()


#--Կառուցում ենք  Logistic Regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#import seaborn as sns
#import matplotlib.pyplot as plt

# Copy dataset
data = df.copy()

# Target variable
y = data['Attrition']
y = y.map({'Yes':1, 'No':0})  # Yes=1, No=0

# ջնջում ենք մեզ պետք չեկող սյուները
data = data.drop(['Attrition','EmployeeNumber','EmployeeCount','Over18','StandardHours'], axis=1)

# Encode categorical variables
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

X = data

# Train/Test մասերի ենք բաժանում
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

#----------------
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))

# Feature Importance
feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()

#----------
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Կանխատեսումներ
y_pred = rf.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Attrition", "Attrition"], yticklabels=["No Attrition", "Attrition"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()

# Classification Report
print(classification_report(y_test, y_pred))

#---