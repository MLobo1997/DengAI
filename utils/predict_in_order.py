import numpy as np

def predict_in_order(X, model, pipeline):
    #X_test_f = pd.DataFrame([], columns=attr[4:])
    predictions=[]
    for idx in range(X.shape[0]):
        x = pipeline.transform(X.loc[idx:idx,:])
        #X_test_f = X_test_f.append(x, sort=False, ignore_index=True)
        pred = model.predict(x)
        pred = int(np.round(pred))
        pipeline.named_steps['l_infected'].append_y(pred)
        predictions.append(pred)
    
    return predictions