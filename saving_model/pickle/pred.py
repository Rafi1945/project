import pickle

model = pickle.load(open("diabetic_80.pkl" , "rb" ))

data = model.predict([[1,85,66,29,0,26.6,0.351,31]])

if data[0] ==0 :
    print("not diabetic")
else:
    print("diabetic")