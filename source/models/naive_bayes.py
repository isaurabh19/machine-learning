from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import KFold

def train_nb(train_validation_data, test_data, k=10):
    kf = KFold(k, random_state=10)
    input_data = train_validation_data[:, :-1]
    targets = train_validation_data[:, -1:]
    for train, test in kf.split(input_data, targets):
        cnb = ComplementNB()
        cnb.fit(train)