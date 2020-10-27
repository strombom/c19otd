
import pandas as pd
import numpy as np
from datetime import timedelta, date

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

prev_antal_fall = np.zeros(10, np.int)
prev_antal_iva = np.zeros(10, np.int)
prev_antal_dod = np.zeros(10, np.int)

fall_string = '"Datum",0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-"'
iva_string = '"Datum","0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-"'
dod_string = '"Datum","0-9","10-19","20-29","30-39","40-49","50-59","60-69","70-79","80-89","90-"'

start_date = date(2020, 3, 26)
end_date = date(2020, 10, 23)
previous_xl_file = None
for single_date in daterange(start_date, end_date):
    filename = '../../other/covid/data/FHM/Folkhalsomyndigheten_Covid19_' + single_date.strftime("%Y-%m-%d") + '.xlsx'
    print("reading", single_date)
    try:
        xl_file = pd.read_excel(filename, sheet_name='Totalt antal per åldersgrupp')
        previous_xl_file = xl_file
    except:
        xl_file = previous_xl_file
    
    antal_fall = np.zeros(10, np.int)
    antal_iva = np.zeros(10, np.int)
    antal_dod = np.zeros(10, np.int)
    for i in range(10):
        fall = xl_file['Totalt_antal_fall'][i]
        iva = xl_file['Totalt_antal_intensivvårdade'][i]
        dod = xl_file['Totalt_antal_avlidna'][i]

        if np.isnan(iva):
            iva = 0

        antal_fall[i] = fall
        antal_iva[i] = iva
        antal_dod[i] = dod

    diff_fall = antal_fall - prev_antal_fall
    diff_iva = antal_iva - prev_antal_iva
    diff_dod = antal_dod - prev_antal_dod

    fall_string += '\n' + single_date.strftime("%Y-%m-%d") + ','
    iva_string += '\n' + single_date.strftime("%Y-%m-%d") + ','
    dod_string += '\n' + single_date.strftime("%Y-%m-%d") + ','
    for i in range(10):
        fall_string += '' + str(diff_fall[i]) + ','
        iva_string += '' + str(diff_iva[i]) + ','
        dod_string += '' + str(diff_dod[i]) + ','

    prev_antal_fall = antal_fall
    prev_antal_iva = antal_iva
    prev_antal_dod = antal_dod

f = open("fall.csv", "w")
f.write(fall_string)
f.close()

f = open("iva.csv", "w")
f.write(iva_string)
f.close()

f = open("dod.csv", "w")
f.write(dod_string)
f.close()
