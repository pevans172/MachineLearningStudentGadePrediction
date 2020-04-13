
import pandas as pd
import numpy as np
import pathlib


def add_important_cats(df1):
    filler = []
    for i in range(len(df1)):
        filler.append(0.0)
    # make an empty column for the studentInfo to hold the average assesment mark for them
    df1['Average_Assesment_Score'] = filler
    # make an empty column for the studentInfo to hold the total interactions with course material from the vle
    df1['Total_no_of_material_use'] = filler

    for row in df1.itertuples():
        #################################################################
        #################### Average assessment mark ####################
        #################################################################
        a = studentAsses[studentAsses.id_student == row.id_student]
        b = a[a.code_module == row.code_module]
        averageScore = b[b.code_presentation == row.code_presentation].score.mean()
        if pd.isnull(averageScore):
            continue
        else:
            df1.at[row.Index, "Average_Assesment_Score"] = averageScore
        ################################################################
        #################### VLE sum of interaction ####################
        ################################################################
        a = studentVle[studentVle.id_student == row.id_student]
        b = a[a.code_module == row.code_module]
        c = b[b.code_presentation == row.code_presentation]
        total = c.sum_click.sum()
        if pd.isnull(total):
            continue
        else:
            df1.at[row.Index, "Total_no_of_material_use"] = total
        print(row.Index)


# see if the table with all the info we want is already here
try:
    studentInfoEdited = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentInfoEdited.csv')
    print(studentInfoEdited.head())
# if not then go ahead and make the file
# This takes 15 minutes
except:
    studentInfo = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentInfo.csv')
    studentAsses = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentAssessment.csv')
    asses = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'assessments.csv')
    studentVle = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentVle.csv')

    # joins together to get a table with all relevant info about the assesments for us
    asses.drop(['weight', 'date', 'assessment_type'], axis=1, inplace=True)
    studentAsses = pd.merge(studentAsses, asses,  on='id_assessment', how='inner')
    # remove data we dont need
    studentAsses.drop(['is_banked', 'date_submitted', 'id_assessment'], axis=1, inplace=True)
    studentVle.drop(['id_site', 'date'], axis=1, inplace=True)

    # add all the important info to the tables
    add_important_cats(studentInfo)
    # save this to a file to save time
    studentInfo.to_csv((pathlib.Path.cwd() / 'Data' / 'studentInfoEdited.csv'), index=False)
    studentInfoEdited = pd.read_csv(pathlib.Path.cwd() / 'Data' / 'studentInfoEdited.csv')
    print(studentInfoEdited.head())
