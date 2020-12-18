import pandas as pd
import numpy as np

def loadData():
    data = pd.read_csv("../data_set/play_tennis2.csv");
    return data

def cal_proportion(data):
    no_rows, no_cols = data.shape
    unique_class_value = np.unique(data.iloc[:, -1])
    arr = []
    # print(data.Outlook[data.Play == "Yes"])
    # print(data.iloc[:, 0][data.iloc[:, -1] == "Yes"])
    for column_index in range(no_cols-1):
        column_value = data.iloc[:, column_index]
        arr.clear()
        for class_index in range(len(unique_class_value)):
            # proportion[unique_class_value[class_index]] = []
            column_class_value = column_value[data.iloc[:, -1] == unique_class_value[class_index]]
            p = column_class_value.value_counts()/column_class_value.count()

            obj = { unique_class_value[class_index]: p }
            # print(obj)
            arr.append(obj)
        
        print(arr)
    
    # calculating proportion for Play class
    # P = data.iloc[:, -1].value_counts()/data.iloc[:, -1].count()
    # proportion[no_cols-1] = P
    # print(proportion)
    # return proportion

# def predict(column, X, proportion):
#     P_yes = proportion["Rainy"]*proportion["Cool"]*proportion["High"]*proportion[False]*P["Yes"]
#     P_no = proportion["Rainy"]*proportion["Cool"]*proportion["High"]*proportion[False]*P["No"]

def main():
    data = loadData()
    #print(data)
    proportion = cal_proportion(data)
    # print(proportion)

    '''
    # calculating proportion for Outlook attribute
    dtOy = data.Outlook[data.Play == "Yes"]
    # print(dtOy.value_counts())
    # print(dtOy.count())
    P1_1 = dtOy.value_counts()/dtOy.count()
    print(P1_1)

    dtOn = data.Outlook[data.Play == "No"]
    P1_2 = dtOn.value_counts()/dtOn.count()
    # print(P1_2)

    # calculating proportion for Temperature attribute
    dtTy = data.Temp[data.Play == "Yes"]
    P2_1 = dtTy.value_counts()/dtTy.count()
    # print(P2_1)

    dtTn = data.Temp[data.Play == "No"]
    P2_2 = dtTn.value_counts()/dtTn.count()
    # print(P2_2)

    # calculating proportion for Humidity attribute
    dtHy = data.Humidity[data.Play == "Yes"]
    P3_1 = dtHy.value_counts()/dtHy.count()
    # print(P3_1)

    dtHn = data.Humidity[data.Play == "No"]
    P3_2 = dtHn.value_counts()/dtHn.count()
    # print(P3_2)

    # calculating proportion for Windy attribute
    dtWy = data.Windy[data.Play == "Yes"]
    P4_1 = dtWy.value_counts()/dtWy.count()
    # print(P4_1)

    dtWn = data.Windy[data.Play == "No"]
    P4_2 = dtWn.value_counts()/dtWn.count()
    # print(P4_2)

    # calculating proportion for Play class
    P = data.Play.value_counts()/data.Play.count()
    # print(P)

    P_yes = P1_1["Rainy"]*P2_1["Cool"]*P3_1["High"]*P4_1[False]*P["Yes"]
    P_no = P1_2["Rainy"]*P2_2["Cool"]*P3_2["High"]*P4_2[False]*P["No"]

    label = "Yes" if P_yes > P_no else "No"

    # print(label)
    print(P_yes)
    print(P_no)

    # PY = P_yes/(P_yes+P_no)
    # PN = P_no/(P_yes+P_no)

    # print(PY)
    # print(PN) '''


if __name__=="__main__":
    main()