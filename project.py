# Import necessary libraries
import pandas as pd
# Load the dataset
telco_data = pd.read_csv("telco_customer_churn.csv") 
telco_data.TotalCharges=pd.to_numeric(telco_data.TotalCharges, errors='coerce')
telco_data.loc[telco_data['TotalCharges'].isnull()==True]
telco_data.dropna(how='any',inplace=True)
telco_data.drop(columns=['customerID'],axis=1,inplace=True)
Data=telco_data.copy()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in telco_data.columns:
    if(telco_data[i].dtype==object):
        telco_data[i]=le.fit_transform(telco_data[i])
        
for i in telco_data.columns:
    if abs(telco_data["Churn"].corr(telco_data[i]))<0.15:
        telco_data.drop(columns=i,inplace=True)
        
        
X=telco_data.drop("Churn",axis=1)
Y=telco_data['Churn']
df=telco_data.copy()
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
sm=SMOTEENN()
X_resampled,Y_resampled=sm.fit_resample(X,Y)
Xr_train,Xr_test,Yr_train,Yr_test=train_test_split(X_resampled,Y_resampled,test_size=0.2,random_state=42)
X_train_scaled = scaler.fit_transform(Xr_train)
X_test_scaled = scaler.transform(Xr_test)

from xgboost import XGBClassifier

# Model 7: XGBoost Classifier
print("\nModel 7: XGBoost Classifier")
model_xgb = XGBClassifier()
model_xgb.fit(X_train_scaled, Yr_train)
yr_pred_xgb = model_xgb.predict(X_test_scaled)

# Taking user input:
sen=input("Enter Yes If Customer is Senior Citizen else Enter No: ")
if sen.lower()=="yes":
    senior=1
elif sen.lower()=="no":
    senior=0
else:
    senior=0
dep=input("Enter Customer's Dependency (Yes or No): ")
if dep.lower()=="yes":
    depend=1
elif dep.lower()=="no":
    depend=0
else:
    depend=0

tenure=int(input("Number of Months the customer has Stayed: "))
ois=input("Whether Customer has Online Security or not (Yes, No or No internet sevice): ")
if ois.lower()=="yes":
    OnlineSecurity=2
elif ois.lower()=="no":
    OnlineSecurity=0
else:
    OnlineSecurity=1
    
oib=input("Whether the customer has Online Backup or Not (Yes, No or No internet Service): ")
if oib.lower()=="yes":
    OnlineBackup=2
elif oib.lower()=="no":
    OnlineBackup=0
else:
    OnlineBackup=1
    
dvp=input("Whether Custom has Device Protection or not (Yes, No or No internet service): ")
if dvp.lower()=="yes":
    DeviceP=2
elif dvp.lower()=="no":
    DeviceP=0
else:
    DeviceP=1
ts=input("Whether Custom has Tech Support or not (Yes, No or No internet service): ")
if ts.lower()=="yes":
    TechS=2
elif ts.lower()=="no":
    TechS=0
else:
    TechS=1
con=input("Contract Term of the Customer (Month-to-Month, One Year or Two Year): ")
if con.lower()=="month-to-month":
    Contract=0
elif con.lower()=="one year":
    Contract=1
elif con.lower()=="two year":
    Contract=2

plb=input("Whether the customer has Paperless Billing or Not (Yes or No): ")
if plb.lower()=="yes":
    Bill=1
elif plb.lower()=="two year":
    Bill=0
else:
    Bill=1
    
month=float(input("The amount charged to the Customer Monthly: "))
total=float(input("Total Amount charged to the customer: "))

X_input=[senior,depend,tenure,OnlineSecurity,OnlineBackup,DeviceP,TechS,Contract,Bill,month,total]

X_input_df=pd.DataFrame([X_input])
X_input_df.columns=["SeniorCitizen","Dependents","tenure","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","Contract","PaperlessBilling","MonthlyCharges","TotalCharges"]
X_input_scaled=scaler.transform(X_input_df)
X_input_scaled

Y_input_pred=model_xgb.predict(X_input_scaled)
if Y_input_pred[0]==0:
    predicted="No"
else:
    predicted="Yes"
print("Churn Prediction: ",predicted)