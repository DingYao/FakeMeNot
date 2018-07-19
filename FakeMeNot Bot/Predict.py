import pickle

var = input("Input Statement to be Verified: ")

def detecting_fake_news(var):    
    load_model = pickle.load(open('Model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return (print("Likely",prediction[0]),
        print("Truth Probability: ",prob[0][1]))


if __name__ == '__main__':
    detecting_fake_news(var)
