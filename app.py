from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder



app = Flask(__name__)


classifier = xgb.Booster()
classifier.load_model("xgb.bin")

def age_group_indicator(age):
    if 0 <= age <= 30:
        return 2
    elif 30 < age <= 50:
        return 0
    elif 50 < age <= 100:
        return 1
    
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("request: ",data)
        # Get input parameters from the query string
        age = int(data['age'])
        subscription_length_months = int(data['subscription_length_months'])
        monthly_bill = float(data['monthly_bill'])
        total_usage_gb = float(data['total_usage_gb'])
        gender = data['gender']
        location = data['location']

        # Create a DataFrame with the input data
        input_data = {
            'Age': int(age),
            'Subscription_Length_Months': int(subscription_length_months),
            'Monthly_Bill': float(monthly_bill),
            'Total_Usage_GB': float(total_usage_gb),
            'Gender': gender,
            'Location':location
        }
        
        print("input_data: ",input_data)
        df = pd.DataFrame(input_data, index=[0])

        # 1. Average Monthly Data Usage
        
        df['Average_Monthly_Data_Usage'] = df['Total_Usage_GB'] / df['Subscription_Length_Months']

        # 2. Billing Change Rate
        df['Billing_Change_Rate'] = df['Monthly_Bill'].diff()

        # 3. Billing Amount as a Percentage
        df['Billing_As_Percentage'] = (df['Monthly_Bill'] / df['Monthly_Bill'].mean()) * 100

        # 4. Customer Tenure in Months
        df['Customer_Tenure_Months'] = df['Subscription_Length_Months']

        # 5. Churn History (Assuming 'Churn' is a binary column indicating churn history)
        df['Churn_History'] = 0  # Lagged version of the churn column

        # 6. Age Group Indicator (Assuming age groups are defined)
        # age_bins = [0, 30, 50, 100]  # Define your age groups as needed
        # age_labels = ['Young', 'Middle-Aged', 'Senior']
        # df['Age_Group_Indicator'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

        # 7. Remaining Subscription Length
        df['Remaining_Subscription_Length'] = df['Subscription_Length_Months'] - df.index

        # 8. Average Bill Change
        df['Average_Bill_Change'] = 0.000048
        
        # age_bins = [0, 30, 50, 100]  # Define your age groups as needed
        # age_labels = ['Young', 'Middle-Aged', 'Senior']

        df['Age_Group_Indicator'] = df['Age'].apply(age_group_indicator)

        
        print("df: ",df)

        # label_encoder = LabelEncoder()
        # Apply label encoding to the 'Age_Group_Indicator' column
        # df['Age_Group_Indicator'] = label_encoder.fit_transform(df['Age_Group_Indicator'])
        # df = pd.get_dummies(df, columns=['Gender','Location'])
        if gender == 'Male':
            df = df.drop(columns='Gender', axis=1)
            df['Gender_Male'] = 1
            df['Gender_Female'] = 0
        else:
            df = df.drop(columns='Gender', axis=1)
            df['Gender_Male'] = 0
            df['Gender_Female'] = 1


        print("df2: ",df)

        if location=='Houston':
            # df.drop(columns='Location',axis=1)
            df['Location_Houston']=1
            df['Location_Miami']=0
            df['Location_New_York']=0
            df['Location_Chicago']=0
        elif location=='Miami':
            # df.drop(columns='Location',axis=1)
            df['Location_Houston']=0
            df['Location_Miami']=1
            df['Location_New_York']=0
            df['Location_Chicago']=0
        elif location=='New_York':
            # df.drop(columns='Location',axis=1)
            df['Location_Houston']=0
            df['Location_Miami']=0
            df['Location_New_York']=1
            df['Location_Chicago']=0
        else:
            # df.drop(columns='Location',axis=1,inplace=True)
            df['Location_Houston']=0
            df['Location_Miami']=0
            df['Location_New_York']=0
            df['Location_Chicago']=1
        df.drop(columns='Location',axis=1,inplace=True)
        # if df['Average_Bill_Change'].isNa():
        df['Average_Bill_Change']=0

        # if 0<=df['Age']<=30:
        #     # df.drop(columns='Age',axis=1)
        #     df['Age_Group_Indicator']=2
        # elif 30<df['Age']<=50:
        #     # df.drop(columns='Age',axis=1)
        #     df['Age_Group_Indicator']=0
        # elif 50<df['Age']<=100:
        #     # df.drop(columns='Age',axis=1)
        #     df['Age_Group_Indicator']=1

        
        # Use the loaded model to make predictions
        print("row: ",df)

        predictions2 = classifier.predict(xgb.DMatrix(df))
        # prediction_list =  classifier.predict_proba(xgb.DMatrix(df)).values.to_list()

        print("Predictions are: \t",predictions2)
        # predictions = classifier.predict(df)
        # print("Predictions are: \t",predictions)
        # Extract the predicted class label
        predicted_label = 0 if predictions2<0.5 else 1

        # Prepare the response as JSON
        response = {'prediction': predicted_label}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
