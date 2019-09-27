def outlierCleaner(predictions, ages, Insulin):
    cleaned_data = []
    tuple_data = []

    for i in range(len(predictions)):
        diff = abs(predictions[i]-Insulin[i])
        tup = (ages[i][0], Insulin[i], diff)
        tuple_data.append(tup)


    tuple_data=sorted(tuple_data,key=lambda x: x[2])
    remove = int(len(tuple_data)*0.1)

    ### your code goes here

    cleaned_data = tuple_data[0:len(tuple_data)-remove]
    return cleaned_data

