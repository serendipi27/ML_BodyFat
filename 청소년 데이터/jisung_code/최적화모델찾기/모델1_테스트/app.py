import joblib
from sklearn.preprocessing import StandardScaler
from category_encoders import BinaryEncoder
from catboost import CatBoostRegressor
import pandas as pd

def cb_model(age, gender, height, weight):
       pf = joblib.load("../../"polyfeatures_1.pkl")
       ss = joblib.load("../../"standscaler_2.pkl")
       lr_model_load = joblib.load("../../"lr_model_3.pkl")
       ex_01 = pd.read_csv('../../ex_01.csv', encoding='cp949')
       ex_01 = ex_01.drop([0])

       year = 2022 #년도
       birth_year = year - int(age)  # 년생
       BMI = round(float(weight) / ( (float(height)/100) * (float(height)/100)), 2 )# BMI수치

       # 성별 인코더를 따로 하지 않고 여기에서 진행(실제 제 모델에서도 동일하게 작업함.)
       if gender == 'M':
           gender = 0
       else:
           gender = 1

       ex_01 = ex_01.append({ '측정나이': age,
                            '측정회원성별': gender,
                            '신장': height,
                            '체중': weight},ignore_index = True)
       print(ex_01)

       result = lr_model_load.predict(ss.transform(pf.transform(ex_01)))
#       df_X = binary_encoder.transform(ex_01)
#       df_X_standard_scaler = standard_scaler.transform(df_X)
#       result = cb_model_load.predict(df_X_standard_scaler)
       return result[0], BMI


def cb_model2(local, age, gender, height, weight, lastscore):
    binary_encoder = joblib.load("../../binary_encoder.pkl")
    cb_model_load = joblib.load("../../cb_model_1.pkl")
    standard_scaler = joblib.load("../../standard_scaler.pkl")
    ex_01 = pd.read_csv('../../ex_01.csv', encoding='cp949')
    ex_01 = ex_01.drop([0])

    year = 2022  # 년도
    birth_year = year - int(age)  # 년생
    BMI = round(float(weight) / ((float(height) / 100) * (float(height) / 100)), 2)  # BMI수치
    ex_01 = ex_01.append({'지역': local,
                          '측정나이': age,
                          '년도': year,
                          '년생': birth_year,
                          '측정회원성별': gender,
                          '신장': height,
                          '체중': weight,
                          'BMI': BMI}, ignore_index=True)
    print(ex_01)

    result = cb_model_load.predict(standard_scaler.transform(binary_encoder.transform(ex_01)))
    #       df_X = binary_encoder.transform(ex_01)
    #       df_X_standard_scaler = standard_scaler.transform(df_X)
    #       result = cb_model_load.predict(df_X_standard_scaler)
    return result[0], BMI

print(cb_model('충남', 15.0, 'F', '165.5', '52.7'))
print(cb_model2('충남', 15.0, 'F', '165.5', '52.7','50'))

