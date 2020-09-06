import joblib
import re
import lxml.html


class Predict:
    def __init__(self):
        self.model = joblib.load('models/lr_model.joblib')
        self.vectorizer = joblib.load('models/vectorizer.joblib')

    # noinspection PyMethodMayBeStatic
    def clean_text(self, text: str) -> str:
        text = lxml.html.fromstring(text).text_content()
        text = re.sub('^(a-zA-Z)\s', '', text)
        return text

    def predict(self, text: str) -> tuple:
        text = self.clean_text(text=text)
        text = self.vectorizer.transform([text])
        prediction = self.model.predict(text)[0]
        probablity = self.model.predict_proba(text)[0]
        return prediction, probablity


if __name__ == '__main__':
    pp = Predict()
    sample_text = 'AssertionError in Pandas columns'
    pred = pp.predict(text=sample_text)
    print(pred)
