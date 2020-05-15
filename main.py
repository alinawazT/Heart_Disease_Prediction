import web
import pandas as pd
#import model_evaluation_utils as meu
data = pd.read_csv("heart.csv")
#print(data)

X = data.loc[:, 'age' : 'thal']
Y = data.loc[:,'target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42)


from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(X_train, y_train)

NBprediction = NB.predict(X_test)
#print(NBprediction)


# # Actual Prediction

features = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
training_features = data[features]

label = ["target"]
outcome_features = data[label]

#print(training_features)

num_features = ["age","trestbps","chol","thalach","oldpeak"]
cat_features = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]
# 

label = ['target']
outcome_features = data[label]

training_features


training_features = pd.get_dummies(training_features, columns=cat_features)
training_features


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(training_features[num_features])


cat_eng_features = list(set(training_features.columns) - set(num_features))

model=NB.fit(X_train, y_train)
from sklearn.externals import joblib
import os
if not os.path.exists("Mode4"):
    os.mkdir("Mode4")
if not os.path.exists("Scale4"):
    os.mkdir("Scale4")

joblib.dump(model, r'Mode4/model4.pickle')
joblib.dump(ss, r'Scale4/scalar4.pickle')

model = joblib.load('Mode4/model4.pickle')
scaler = joblib.load('Scale4/scalar4.pickle')

urls = (
    '/(.*)', 'hello'
)
app = web.application(urls, globals())

class hello:
    def GET(self,name):
        data=web.input()
        name = '80%'
        web.header('Content-Type', 'application/json')

        new_data = pd.DataFrame([{"age":str(data.age),"sex":str(data.sex),"cp":str(data.cp),"trestbps":str(data.bps),"chol":str(data.chol),"fbs":str(data.fbs),"restecg":str(data.ecg),"thalach":str(data.thalach),"exang":str(data.exang),"oldpeak":str(data.oldpeak),"slope":str(data.slope),"ca":str(data.ca),"thal":str(data.thal)}])
        new_data = new_data[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
        new_data


        #outcome_name = ['target']
        #outcome_labels = data[outcome_name]


        prediction_features = new_data[features]
        prediction_features[num_features] = ss.transform(prediction_features[num_features])
        prediction_features = pd.get_dummies(prediction_features, columns=cat_features)
        prediction_features

        current_categorical_engineered_features = set(prediction_features.columns) - set(num_features)
        missing_features = set(cat_eng_features) - current_categorical_engineered_features
        for feature in missing_features:
            prediction_features[feature] = [0] * len(prediction_features) 
        prediction_features


        predictions = model.predict(new_data)
        new_data['target'] = predictions
        new_data

        #print(new_data['target'][0])
        return '{"result" : "' + str(new_data['target'][0]) + '"}'

if __name__ == "__main__":
    app.run()