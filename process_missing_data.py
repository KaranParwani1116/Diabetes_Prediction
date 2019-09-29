import matplotlib.pyplot as plt
import numpy as np


def processdata(Age, Insulin,figno1,figno2, label1, label2):
    AgeTransformed=[]
    InsulinTransformed=[]

    for i in range(len(Insulin)):
        if(Insulin[i]!=0):
            AgeTransformed.append(Age[i])
            InsulinTransformed.append(Insulin[i])


    from sklearn import linear_model

    Age1=np.array(AgeTransformed).reshape(-1, 1)
    reg = linear_model.LinearRegression()
    reg.fit(Age1,InsulinTransformed)

    Age2=np.array(Age).reshape(-1, 1)



    from Outlier_Cleaner import outlierCleaner
    cleaned_data = outlierCleaner(reg.predict(Age1), Age1, InsulinTransformed)
    Ages_Min=[]
    Insulin_min=[]

    for ages , insulin , diff in cleaned_data:
        Ages_Min.append(ages)
        Insulin_min.append(insulin)




    Ages_Min = np.array(Ages_Min).reshape(-1, 1)
    reg.fit(Ages_Min, Insulin_min)


    #Now Recovering Lost Data
    Insulins=reg.predict(Age2)
    print(len(Insulins))
    for i in range(len(Insulin)):
        if Insulin[i]==0:
            Insulin[i] =int(Insulins[i])

    return Insulin