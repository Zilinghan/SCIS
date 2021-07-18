import numpy as np
import pandas as pd

def FilePreprocess(file):
    # Check the type of the files and Prep
    if type(file) == str:
        if file[len(file)-4:] == '.csv':
            df1 = pd.read_csv(file)
        elif file[len(file)-5:] == '.xlsx':
            df1 = pd.read_excel(file)
        EmptyCol1 = np.sum(df1.isnull(), axis=0) == df1.shape[0]
        EmptyColIdx1 = df1.columns[np.arange(df1.shape[1])[EmptyCol1]]
        df1.drop(EmptyColIdx1, axis=1, inplace=True)
        EmptyRow1 = np.sum(df1.isnull(), axis=1) == df1.shape[1]
        EmptyRowIdx1 = df1.index[np.arange(df1.shape[0])[EmptyRow1]]
        df1.drop(EmptyRowIdx1, axis=0, inplace=True)
    elif type(file) == pd.core.frame.DataFrame:
        df1 = file
    else:
        raise ValueError("Invalid input file type!")    
    return df1

def MergeTwoFile(file1, file2, file1RealCol):
    df1 = FilePreprocess(file1)
    df2 = FilePreprocess(file2)    
    # Start the Merging
    if file1RealCol == None:
        file1RealCol = list(df1.columns)
    file1RealCol_Copy = file1RealCol[:]
    for i in range(1, len(df2.columns)):
        Col2 = df2.columns[i]
        SameNameCounter = 0
        MatchFlag = False
        for j in range(1, len(file1RealCol)):
            Col1Real = file1RealCol[j]
            Col1 = df1.columns[j]
            if Col2 == Col1Real:
                SameNameCounter += 1
                # Create a dictionary for comparing
                Id_Col_Dict1 = {}
                for k in range(df1.shape[0]):
                    Id_Col_Dict1[df1.iat[k,0]] = (df1.iloc[k][Col1], k)
                # Start checking whether Col2 matches Col1
                MatchFlag = False
                for k in range(df2.shape[0]):
                    if df2.iat[k,0] in Id_Col_Dict1:
                        if str(df2.iloc[k][Col2]) == 'nan' or str(Id_Col_Dict1[df2.iat[k,0]][0]) == 'nan' or Id_Col_Dict1[df2.iat[k,0]][0] == df2.iloc[k][Col2]:
                            pass
                        else:
                            break
                    if k+1 == df2.shape[0]:
                        MatchFlag = True
                # When matching, fill the empty part
                if MatchFlag:
                    for k in range(df2.shape[0]):
                        if df2.iat[k,0] in Id_Col_Dict1:
                            if str(df2.iloc[k][Col2]) == 'nan' and str(Id_Col_Dict1[df2.iat[k,0]][0]) != 'nan':
                                df2.iloc[k,i] = Id_Col_Dict1[df2.iat[k,0]][0]
                            elif str(df2.iloc[k][Col2]) != 'nan' and str(Id_Col_Dict1[df2.iat[k,0]][0]) == 'nan':
                                df1.iloc[Id_Col_Dict1[df2.iat[k,0]][1],j] = df2.iloc[k][Col2]
                    df2.rename(columns={Col2: Col1}, inplace=True)
                    break
        if not MatchFlag:
            if SameNameCounter == 0:
                file1RealCol_Copy.append(Col2)
            else:
                file1RealCol_Copy.append(Col2)
                df2.rename(columns={Col2: Col2+"_"+str(SameNameCounter)}, inplace=True)
    df = pd.merge(df1, df2, how='outer')
    return df, file1RealCol_Copy

def MergeFiles(files):
    if len(files) == 0:
        return 0
    if len(files) == 1:
        return FilePreprocess(files[0])
    df, RealCol = MergeTwoFile(files[0], files[1], None)
    for i in range(2, len(files)):
        df, RealCol = MergeTwoFile(df, files[i], RealCol)
    return df


