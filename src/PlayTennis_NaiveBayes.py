import pandas as pd

def loadData():
    playTennis = pd.read_csv("../data_set/play_tennis2.csv");
    return playTennis

def main():
    playTennis = loadData()
    # print(playTennis)

    # calculating proportion for Outlook attribute
    dtOy = playTennis.Outlook[playTennis.Play == "Yes"]
    # print(dtOy.value_counts())
    # print(dtOy.count())
    P1_1 = dtOy.value_counts()/dtOy.count()
    # print(P1_1)

    dtOn = playTennis.Outlook[playTennis.Play == "No"]
    P1_2 = dtOn.value_counts()/dtOn.count()
    # print(P1_2)

    # calculating proportion for Temperature attribute
    dtTy = playTennis.Temp[playTennis.Play == "Yes"]
    P2_1 = dtTy.value_counts()/dtTy.count()
    # print(P2_1)

    dtTn = playTennis.Temp[playTennis.Play == "No"]
    P2_2 = dtTn.value_counts()/dtTn.count()
    # print(P2_2)

    # calculating proportion for Humidity attribute
    dtHy = playTennis.Humidity[playTennis.Play == "Yes"]
    P3_1 = dtHy.value_counts()/dtHy.count()
    # print(P3_1)

    dtHn = playTennis.Humidity[playTennis.Play == "No"]
    P3_2 = dtHn.value_counts()/dtHn.count()
    # print(P3_2)

    # calculating proportion for Windy attribute
    dtWy = playTennis.Windy[playTennis.Play == "Yes"]
    P4_1 = dtWy.value_counts()/dtWy.count()
    # print(P4_1)

    dtWn = playTennis.Windy[playTennis.Play == "No"]
    P4_2 = dtWn.value_counts()/dtWn.count()
    # print(P4_2)

    # calculating proportion for Play class
    P = playTennis.Play.value_counts()/playTennis.Play.count()
    # print(P)

    P_yes = P1_1["Rainy"]*P2_1["Cool"]*P3_1["High"]*P4_1[False]*P["Yes"]
    P_no = P1_2["Rainy"]*P2_2["Cool"]*P3_2["High"]*P4_2[False]*P["No"]

    label = "Yes" if P_yes > P_no else "No"

    print(label)
    # print(P_yes)
    # print(P_no)

    # PY = P_yes/(P_yes+P_no)
    # PN = P_no/(P_yes+P_no)

    # print(PY)
    # print(PN)


if __name__=="__main__":
    main()