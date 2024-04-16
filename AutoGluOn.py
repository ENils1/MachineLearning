from autogluon.tabular import TabularDataset, TabularPredictor

data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
train_data = TabularDataset(f'{data_url}train.csv')
train_data.head()

label = 'signature'
train_data[label].describe()

predictor = TabularPredictor(label=label).fit(train_data)

test_data = TabularDataset(f'{data_url}test.csv')

y_pred = predictor.predict(test_data.drop(columns=[label]))
y_pred.head()

predictor.evaluate(test_data, silent=True)

predictor.leaderboard(test_data)