import joblib

model = joblib.load("diabetic_80.pkl")

data = model.predict([[1,85,66,29,0,26.6,0.351,31]])

if data[0] ==0 :
    print("not diabetic")
else:
    print("diabetic")