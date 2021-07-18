import sys
from PySide2.QtWidgets import *
from PySide2.QtGui import QBrush, QColor, QIcon, QFont, QImage, QPalette, QPixmap, QIntValidator, QDoubleValidator
from PySide2.QtCore import QThread, Qt, QSize, Signal
import pandas as pd
import numpy as np
import merge

def merge_two_file(file1, file2, how):
    '''
    Description:
    Merging two files according to the user-specified methods
    
    Parameters:
    file1: str - name of the first file
    file2: str - name of the second file
    how:   str - user-specified merge method 
        *Note*: all files should have their first columns as 'index/id'

        Explanations for different choices of 'how':
        (1) 'intersection':
            When merging files, for overlapping features, we only choose the intersected samples
        (2) 'union': (default)
            When merging files, we create new samples for those do not overlap with each other
        (3) 'first'
            All other files align to the first file, i.e., all samples in the merged file is all  
            tge samples in the first file. This is commonly used if the first file contains most
            of the data and following files are used to provide some samll amount of additional info.

    Output:
    df:    pd.DataFrame - the merged DataFrame for given two files
    '''
    if type(file1) == str:
        # check the file-type first
        if file1[len(file1) - 4:] == '.csv':
            df1 = pd.read_csv(file1)
        elif file1[len(file1) - 5:] == '.xlsx':
            df1 = pd.read_excel(file1)
        else:
            print(file1[len(file1) - 5:])
            raise ValueError("Invalid input file type!!")

        # Pre-process the DataFrame
        # (1) Drop the completely empty columns        
        empty_col1 = np.sum(df1.isnull(), axis=0) == df1.shape[0]
        empty_col_idx1 = df1.columns[np.array(range(df1.shape[1]))[empty_col1]]
        df1.drop(empty_col_idx1, axis=1, inplace=True)
        # (2) Drop the completely empty rows
        empty_row1 = np.sum(df1.isnull(), axis=1) == df1.shape[1]
        empty_row_idx1 = df1.index[np.array(range(df1.shape[0]))[empty_row1]]
        df1.drop(empty_row_idx1, axis=0, inplace=True)
        # (3) Change the name of the first column
        first_col_name1 = df1.columns[0]
        df1.rename(columns={first_col_name1: '__id__'}, inplace=True)
    elif type(file1) == pd.core.frame.DataFrame:
        df1 = file1
    else:
        raise ValueError("Invalid input file type!!")

    if type(file2) == str:
        if file2[len(file2) - 4:] == '.csv':
            df2 = pd.read_csv(file2)
        elif file2[len(file2) - 5:] == '.xlsx':
            df2 = pd.read_excel(file2)
        else:
            raise ValueError("Invalid input file type")

        empty_col2 = np.sum(df2.isnull(), axis=0) == df2.shape[0]
        empty_col_idx2 = df2.columns[np.array(range(df2.shape[1]))[empty_col2]]
        df2.drop(empty_col_idx2, axis=1, inplace=True)

        empty_row2 = np.sum(df2.isnull(), axis=1) == df2.shape[1]
        empty_row_idx2 = df2.index[np.array(range(df2.shape[0]))[empty_row2]]
        df2.drop(empty_row_idx2, axis=0, inplace=True)
        first_col_name2 = df2.columns[0]
        df2.rename(columns={first_col_name2: '__id__'}, inplace=True)
    elif type(file2) == pd.core.frame.DataFrame:
        df2 = file2
    else:
        raise ValueError("Invalid input file type!!")

    # Start the merging
    # Case3: (N1, d1) and (N2, d2) - differnt samples & different features - 'intersection'
    if how == 'Intersection':
        df = pd.merge(df1, df2, how='inner')
    # Case4: (N1, d1) and (N2, d2) - differnt samples & different features - 'union'
    elif how == 'Union':
        df = pd.merge(df1, df2, how='outer')
    # Case5: (N1, d1) and (N2, d2) - differnt samples & different features - 'first'
    elif how == 'First':
        df = pd.merge(df1, df2, 'left')
    else:
        raise ValueError("The parameter how is invalid")
    return df

def merge_files(files, how='union'):
    '''
    Description:
    Merging a list of input files according to the user-specified methods
    
    Parameters:
    files: list of str - list of all input strings to be merged
    how:   str         - user-specified merge method 
        
    Output:
    df:    pd.DataFrame - the merged DataFrame for all given files
    '''
    if len(files) == 0:
        return 0
    if len(files) == 1:
        return merge_two_file(files[0], files[0], how=how)
    df = merge_two_file(files[0], files[1], how)
    for i in range(2, len(files)):
        df = merge_two_file(df, files[i], how)
    return df

def is_text(item):
    try:
        float(item)
        return False
    except:
        return True

def error_detection(df):
    '''
    Description:
    Given a panda dataframe, detect all possible errors according to *datatype* in each column
    
    Parameter:
    df:         pd.DataFrame - Input dataframe to be detected

    Output:
    col_idx:    list of column indices, which contain errors
    row_idx:    list of list of row indices
    '''
    col_idx = []
    row_idx = []
    num_sample, num_feature = df.shape
    for j in range(num_feature):
        feature_istext = np.array(df.iloc[:,j].map(lambda x : is_text(x)))
        num_text = np.sum(feature_istext)
        if num_sample == 0:
            break
        if num_text/num_sample >= 0.05:
            continue
        else:
            col_idx.append(j)
            row_idx.append(np.argwhere(feature_istext))
    return col_idx, row_idx

class TitleWidget(QWidget):
    def __init__(self):
        super(TitleWidget, self).__init__()
        self.initUI()
    def initUI(self):
        layout = QHBoxLayout()
        # the application picture
        self.app_pic = QLabel() 
        pic = QImage('fig/icon.png')
        scale_pic = pic.scaled(30, 30, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.app_pic.setPixmap(QPixmap.fromImage(scale_pic))

        # the application abbreviation
        self.app_abbr = QLabel("SCIS")
        self.app_abbr.setAlignment(Qt.AlignCenter)

        # the application name
        self.app_name = QLabel("A Large-Scale Missing Data Imputation System")
        self.app_name.setAlignment(Qt.AlignCenter)

        layout.setSpacing(15)
        layout.addWidget(self.app_pic, 0)
        layout.addWidget(self.app_abbr, 0, Qt.AlignLeft)
        layout.addWidget(self.app_name, 1, Qt.AlignLeft)
        self.setLayout(layout)
        
        widget_p = QPalette()
        widget_p.setColor(QPalette.Window, QColor(10, 40, 86))
        widget_p.setColor(QPalette.WindowText, QColor(255, 255, 255))
        widget_font = QFont()
        widget_font.setBold(False)
        widget_font.setFamily("Microsoft YaHei UI")
        widget_font.setPointSize(13)
        self.setPalette(widget_p)
        self.setAutoFillBackground(True)
        self.setFont(widget_font)

class StepWidget(QWidget):
    def __init__(self):
        super(StepWidget, self).__init__()
        self.initUI()
    def initUI(self):    
        self.layout = QHBoxLayout()
        self.SCIS = QLabel()
        SCIS_pic = QImage('fig/SCIS.png')
        scale_SCIS_pic = SCIS_pic.scaled(75, 65, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.SCIS.setPixmap(QPixmap.fromImage(scale_SCIS_pic))
        self.app_pic = QLabel() 
        pic = QImage('fig/zju2.png')
        scale_pic = pic.scaled(75, 65, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.app_pic.setPixmap(QPixmap.fromImage(scale_pic))
        self.step_layout_button1 = QPushButton("File Upload")
        self.step_layout_button2 = QPushButton("File Preview")
        self.step_layout_button3 = QPushButton("Display and Preprocess")
        self.step_layout_button4 = QPushButton("Algorithm Upload")
        self.step_layout_button5 = QPushButton("Imputation Result")
        self.placeholder = QLabel()
        self.placeholder.setMinimumHeight(80)
        self.placeholder.setMaximumHeight(80)
        self.layout.addWidget(self.SCIS, 0)
        self.layout.addWidget(self.app_pic, 0)
        self.layout.addWidget(self.step_layout_button1, 0, Qt.AlignCenter)
        self.layout.addWidget(self.step_layout_button2, 0, Qt.AlignCenter)
        self.layout.addWidget(self.step_layout_button3, 0, Qt.AlignCenter)
        self.layout.addWidget(self.step_layout_button4, 0, Qt.AlignCenter)
        self.layout.addWidget(self.step_layout_button5, 0, Qt.AlignCenter)
        self.layout.addWidget(self.placeholder, 1)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.setStyleSheet('''
            QPushButton{
                border: none;
                height: 80px;
                font-size:35px;
                font: large "Microsoft YaHei UI";
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize:30px 30px;
                text-align: left
            }
            QPushButton:hover{
                height: 80px;
                font-size:35px;
                font: bold large "Microsoft YaHei UI";
                background-color: #0e83c2;
            }
            *{
                background-color: #0c6ca1;
                color: #F7F9FF;
            }
        ''')
        self.step_layout_button1.setMaximumWidth(243)
        self.step_layout_button1.setMinimumWidth(243)
        self.step_layout_button2.setMaximumWidth(253)
        self.step_layout_button2.setMinimumWidth(253)
        self.step_layout_button3.setMaximumWidth(453)
        self.step_layout_button3.setMinimumWidth(453)
        self.step_layout_button4.setMaximumWidth(361)
        self.step_layout_button4.setMinimumWidth(361)
        self.step_layout_button5.setMaximumWidth(363)
        self.step_layout_button5.setMinimumWidth(363)
        
class Page1_Widget(QWidget):
    def __init__(self):
        super(Page1_Widget, self).__init__()
        self.initUI()
    def initUI(self):    
        # ===================================BUTTON================================
        button_layout = QHBoxLayout()
        self.select_button = QPushButton(" Select File(s)")
        self.select_button.setProperty('name', 'file_select_button')
        self.setStyleSheet('''
            QPushButton[name='file_select_button']{
                background-color: #326AA9;
                font: 45px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 40px;
                height: 150px;
                width: 470px;
                qproperty-icon:url(fig/folder.png);
                qproperty-iconSize:80px 80px;
                text-align: center
            }
            QPushButton:hover[name='file_select_button']{
                background-color: #457FBF;
                font: bold 45px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 40px;
                height: 150px;
                width: 470px;
            }
        '''
        )
        button_layout.addWidget(QLabel(),1)
        button_layout.addWidget(self.select_button,0)
        button_layout.addWidget(QLabel(),1)
        # ================================FILE_DISPLAYER=============================
        filedisplay_layout = QVBoxLayout()
        # (1) file_name sub-layout
        file_name_layout = QHBoxLayout()
        file_name = QLabel("File Name")
        file_name.setStyleSheet('''
            *{
                font: 40px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        file_name_layout.addWidget(file_name, 0)
        file_name_layout.addWidget(QLabel(), 1)
        filedisplay_layout.addLayout(file_name_layout)
        # (2) Horizontal Line
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setLineWidth(10)
        hline.setStyleSheet('''
            color: #666666;
        ''')
        filedisplay_layout.addWidget(hline)
        # (3) Display and Delete Table
        self.file_table = QTableWidget()
        self.file_table.setStyleSheet('''
            border: none;
        ''')
        self.file_table.setColumnCount(2)
        self.file_table.setRowCount(0)
        self.file_table.horizontalHeader().setStretchLastSection(True)     
        self.file_table.horizontalHeader().hide()
        self.file_table.verticalHeader().hide() 
        self.file_table.setShowGrid(False)
        self.file_table.setStyleSheet('''
            font-size: 30px;
        ''')
        self.file_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        filedisplay_layout.addWidget(self.file_table)
        # ===================================NEXT_BUTTON===========================
        nextbutton_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        ''')
        nextbutton_layout.addWidget(QLabel(), 1)
        nextbutton_layout.addWidget(self.next_button, 0)

        # =================================ALL LAYOUTS=================================
        placeholder = QLabel()
        placeholder.setMinimumHeight(20)
        placeholder.setMaximumHeight(20)
        layout = QVBoxLayout()
        layout.addWidget(placeholder)
        layout.addLayout(button_layout)
        layout.addWidget(placeholder)
        layout.addLayout(filedisplay_layout)
        layout.addLayout(nextbutton_layout)
        self.setLayout(layout)

class FileDeleteButton(QWidget):
    def __init__(self, idx):
        super(FileDeleteButton, self).__init__()
        self.initUI(idx)
    def initUI(self, idx):    
        self.delete_button = QPushButton("Delete")
        self.delete_button.setProperty('idx', idx)
        self.delete_button.setStyleSheet('''
            QPushButton{
                font: 30px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/delete.png);
                qproperty-iconSize:29px 29px;
                margin-right: 0px;
            }
            QPushButton:hover{
                font: bold 30px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/delete.png);
                margin-right: 0px;
            }
        ''')
        self.delete_button.setMaximumWidth(230)
        self.delete_button.setMinimumWidth(230)
        layout = QHBoxLayout()
        layout.addWidget(QLabel(),1)
        layout.addWidget(self.delete_button,0)
        self.setLayout(layout)

class DataPreviewWidget(QWidget):
    def __init__(self, filename, idx):
        super(DataPreviewWidget, self).__init__()
        self.initUI(filename, idx)
    def initUI(self, filename, idx):
        # Load the file into DataFrame first
        df = merge.FilePreprocess(filename)
        # ============================DATA PREVIEW TABLE===========================
        self.datapreview_table = QTableWidget()
        num_row, num_col = df.shape
        num_row = min(100, num_row)
        all_num_row = max(num_row, 30)
        all_num_col = max(num_col, 15)
        self.datapreview_table.setColumnCount(all_num_col)
        self.datapreview_table.setRowCount(all_num_row)
        self.datapreview_table.setHorizontalHeaderLabels(list(df.columns))
        self.datapreview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in range(num_row):
            for j in range(num_col):
                cur_item = str(df.iat[i,j])
                if cur_item == 'nan':
                    empty_item = QTableWidgetItem()
                    empty_item.setBackground(QBrush(QColor(201,252,255)))
                    self.datapreview_table.setItem(i, j, empty_item)
                    continue
                self.datapreview_table.setItem(i, j, QTableWidgetItem(cur_item))    
        tablefont = QFont()
        tablefont.setPointSize(11)
        self.datapreview_table.setFont(tablefont)
        self.datapreview_table.horizontalHeader().setFont(tablefont)
        self.datapreview_table.resizeColumnsToContents() 
        self.datapreview_table.setAlternatingRowColors(True)
        self.datapreview_table.setStyleSheet('''
            QTableWidget{
                alternate-background-color:#f9f9f9;
            }
        ''')
        # =============================DELETE BUTTON===============================
        self.filedelete_button = FileDeleteButton(idx)
        note = QLabel("Note: Preview mode only displays the first 100 lines for large files.")
        note.setStyleSheet('*{font: 27px "Microsoft YaHei UI";color: #666666;}')
        note_and_delete_layout = QHBoxLayout()
        note_and_delete_layout.addWidget(note, 0)
        note_and_delete_layout.addWidget(QLabel(), 1)
        note_and_delete_layout.addWidget(self.filedelete_button)

        layout = QVBoxLayout()
        layout.addWidget(self.datapreview_table)
        layout.addLayout(note_and_delete_layout)
        self.setLayout(layout)

class Page2_Widget(QWidget):
    def __init__(self, filenames, file_validflag):
        super(Page2_Widget, self).__init__()
        self.initUI(filenames, file_validflag)
    def initUI(self, filenames, file_validflag):
        # ============================DATA PREVIEW==================================
        self.datadisplay_tab = QTabWidget()
        self.alltabwidgets = []
        for i in range(len(filenames)):
            if file_validflag[i]:
                file_name_idx = max(filenames[i].rfind('/'), filenames[i].rfind('\\'))
                filename_abbr = filenames[i][file_name_idx+1:]
                self.alltabwidgets.append(DataPreviewWidget(filenames[i], i))
                excel_icon = QIcon(QPixmap('fig/excel.png'))
                self.datadisplay_tab.addTab(self.alltabwidgets[-1],excel_icon, filename_abbr)
                tabbar_font = QFont()
                tabbar_font.setFamily("Microsoft YaHei UI")
                tabbar_font.setPointSize(12)
                self.datadisplay_tab.tabBar().setFont(tabbar_font)   
                self.datadisplay_tab.tabBar().setIconSize(QSize(32,32))
        # ==============================MERGE METHODS=================================
        # merge_method_layout = QVBoxLayout()
        # # (1) merge_method_word sub-layout
        # merge_method_word_layout = QHBoxLayout()
        # merge_method_word = QLabel("Merge Methods Selection")
        # merge_method_word.setStyleSheet('''
        #     *{
        #         font: 30px large bold "Helvatica";
        #         color: #666666;
        #     }
        # ''')
        # merge_method_word_layout.addWidget(merge_method_word, 0)
        # merge_method_word_layout.addWidget(QLabel(), 1)
        # merge_method_layout.addLayout(merge_method_word_layout)
        # # (2) Horizontal Line
        # hline = QFrame()
        # hline.setFrameShape(QFrame.HLine)
        # hline.setLineWidth(10)
        # hline.setStyleSheet('''
        #     color: #666666;
        # ''')
        # merge_method_layout.addWidget(hline)
        # # (3) Selection of Merge Method
        # self.selectionbox_layout = QHBoxLayout()
        # self.radiobutton1 = QRadioButton("Union")
        # self.radiobutton2 = QRadioButton("Intersection")
        # self.radiobutton3 = QRadioButton("First")
        # self.radiobutton1.setProperty('merge_method', 'Union')
        # self.radiobutton2.setProperty('merge_method', 'Intersection')
        # self.radiobutton3.setProperty('merge_method', 'First')
        # self.radiobutton1.setStyleSheet('*{font-size:25px;}')
        # self.radiobutton2.setStyleSheet('*{font-size:25px;}')
        # self.radiobutton3.setStyleSheet('*{font-size:25px;}')
        # self.placeholder = QLabel()
        # self.selectionbox_layout.setSpacing(100)
        # self.selectionbox_layout.addWidget(self.radiobutton1, 0)
        # self.selectionbox_layout.addWidget(self.radiobutton2, 0)
        # self.selectionbox_layout.addWidget(self.radiobutton3, 0)
        # self.selectionbox_layout.addWidget(self.placeholder, 1)
        # merge_method_layout.addLayout(self.selectionbox_layout)    
        # ===================================NEXT_BUTTON===========================
        nextbutton_layout = QHBoxLayout()
        self.MergeProgressLE = QLabel("Merge Process: ")
        self.MergeProgressLE.setStyleSheet('*{font: 27px "Microsoft YaHei UI";}')
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(sum(file_validflag))
        self.progressbar.setMinimumWidth(600)
        self.progressbar.setMaximumWidth(600)
        self.progressbar.setStyleSheet('''
            QProgressBar{
                background-color: #E6F4F1;
                color: #3F4756;
                border-radius: 10px;
                border-color: transparent;
                text-align: center;
                font: bold;
            }
            QProgressBar::chunk{
                background-color: #00B1AE;
                border-radius: 10px;
            }
        ''')
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        ''')        
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        ''')
        nextbutton_layout.addWidget(self.MergeProgressLE, 0)
        nextbutton_layout.addWidget(self.progressbar, 0)
        nextbutton_layout.addWidget(QLabel(), 1)
        nextbutton_layout.addWidget(self.back_button, 0)
        nextbutton_layout.addWidget(self.next_button, 0)  
        self.MergeProgressLE.hide()
        self.progressbar.hide()      
        
        placeholder = QLabel()
        placeholder.setMinimumHeight(10)
        placeholder.setMaximumHeight(10)

        layout = QVBoxLayout()
        layout.addWidget(self.datadisplay_tab)
        layout.addWidget(placeholder)
        # layout.addLayout(merge_method_layout)
        # layout.addWidget(placeholder)
        layout.addLayout(nextbutton_layout)
        self.setLayout(layout)

class MergeFileThread(QThread):
    TwoFileMerge = Signal(int, int)
    MergeFileFinish = Signal(pd.DataFrame, int, list, list)
    def __init__(self, filenames, file_validflag, previous_pagenum):
        super(MergeFileThread, self).__init__()    
        self.filenames = filenames
        self.file_validflag = file_validflag
        self.previous_pagenum = previous_pagenum
    def run(self):
        valid_files = []
        for i in range(len(self.filenames)):
            if self.file_validflag[i]:
                valid_files.append(self.filenames[i])
        if len(valid_files) == 1:
            self.df = merge.FilePreprocess(valid_files[0])
            self.TwoFileMerge.emit(1,1)
        else:
            self.TwoFileMerge.emit(0, len(valid_files))
            self.df, RealCol = merge.MergeTwoFile(valid_files[0], valid_files[1], None)
            self.TwoFileMerge.emit(2, len(valid_files))
            for i in range(2, len(valid_files)):
                self.df, RealCol = merge.MergeTwoFile(self.df, valid_files[i], RealCol)
                self.TwoFileMerge.emit(i+1, len(valid_files))
        ErrorColIdx, ErrorRowIdx = error_detection(self.df)
        self.TwoFileMerge.emit(len(valid_files)+1, len(valid_files))
        self.MergeFileFinish.emit(self.df, self.previous_pagenum, ErrorColIdx, ErrorRowIdx)

class TransformThread(QThread):
    TenPercent = Signal(int)
    TransformFinish = Signal(dict, int)
    def __init__(self, features, col_idx):
        super(TransformThread, self).__init__()    
        self.features = features
        self.col_idx = col_idx
    def run(self):
        TotalLength = len(self.features)
        TenPercentCounter = 1
        TransformDict = {}
        ClassType = 0
        for i in range(TotalLength):
            if str(self.features[i]) == 'nan':
                pass
            elif not self.features[i] in TransformDict:
                TransformDict[self.features[i]] = ClassType
                ClassType += 1
            if (i+5) == int(TotalLength * TenPercentCounter / 10): ##############PROBLEM HERE###############
                self.TenPercent.emit(TenPercentCounter)
                TenPercentCounter += 1
        self.TransformFinish.emit(TransformDict, self.col_idx)

class SelectionThread(QThread):
    TenPercent = Signal(int)
    SelectionFinish = Signal(np.ndarray, np.ndarray, int)
    def __init__(self, features, SelectionMethod, SelectionRangeDown, SelectionRangeUp, SelectionCondition, SelectionValue, SelectionIndex, col_idx):
        super(SelectionThread, self).__init__()
        self.features = features
        self.SelectionMethod = SelectionMethod
        self.SelectionRangeDown = SelectionRangeDown
        self.SelectionRangeUp = SelectionRangeUp
        self.SelectionCondition = SelectionCondition
        self.SelectionValue = SelectionValue
        self.SelectionIndex = SelectionIndex
        self.col_idx = col_idx
    def run(self):
        TotalLength = len(self.features)
        TenPercentCounter = 1
        SelectedIndicator = np.array([False]*TotalLength)
        if self.SelectionMethod == 'Range':
            if self.SelectionRangeDown == None:
                self.SelectionRangeDown = -float('inf')
            if self.SelectionRangeUp == None:
                self.SelectionRangeUp = float('inf')
            for i in range(TotalLength):
                try:
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else:
                        if float(self.features[i]) >= self.SelectionRangeDown and float(self.features[i]) <= self.SelectionRangeUp:
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = False
                except:
                    SelectedIndicator[i] = False
                if (i+5) == int(TotalLength*TenPercentCounter/9):
                    self.TenPercent.emit(TenPercentCounter)
                    TenPercentCounter += 1
        elif self.SelectionMethod == 'Condition':
            if self.SelectionCondition == 0: # Equal
                for i in range(TotalLength):
                    try:
                        if float(self.features[i]) == float(self.SelectionValue) or str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = (str(self.SelectionValue) == str(self.features[i]) or str(self.features[i]) == 'nan')
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 1: # Not Equal
                for i in range(TotalLength):
                    try:
                        if float(self.features[i]) != float(self.SelectionValue) or str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = (str(self.SelectionValue) != str(self.features[i]) or str(self.features[i]) == 'nan')
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 2: # Greater Than
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) > self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 3: # Greater/Equal
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) >= self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 4: # Smaller Than
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) < self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 5: # Smaller/Equal
                for i in range(TotalLength):
                    try:
                        if str(self.features[i]) == 'nan':
                            SelectedIndicator[i] = True
                        else:
                            if float(self.features[i]) <= self.SelectionValue:
                                SelectedIndicator[i] = True
                            else:
                                SelectedIndicator[i] = False
                    except:
                        SelectedIndicator[i] = False
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 6: # Start with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else: 
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = False
                        else:
                            SelectedIndicator[i] = str(self.features[i])[:len(str(self.SelectionValue))] == str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 7: # Not Start with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else: 
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = str(self.features[i])[:len(str(self.SelectionValue))] != str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 8: # End with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else:
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = False
                        else:
                            SelectedIndicator[i] = str(self.features[i])[len(str(self.features[i]))-len(str(self.SelectionValue)):] == str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 9: # Not End with
                for i in range(TotalLength):
                    if str(self.features[i]) == 'nan':
                        SelectedIndicator[i] = True
                    else:
                        if len(str(self.SelectionValue)) > len(str(self.features[i])):
                            SelectedIndicator[i] = True
                        else:
                            SelectedIndicator[i] = str(self.features[i])[len(str(self.features[i]))-len(str(self.SelectionValue)):] != str(self.SelectionValue)
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 10: # Contain
                for i in range(TotalLength):
                    SelectedIndicator[i] = (str(self.SelectionValue) in str(self.features[i]) or str(self.features[i]) == 'nan')
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
            elif self.SelectionCondition == 11: # Not contain
                for i in range(TotalLength):
                    SelectedIndicator[i] = (not (str(self.SelectionValue) in str(self.features[i]))) or str(self.features[i]) == 'nan'
                    if (i+5) == int(TotalLength*TenPercentCounter/9):
                        self.TenPercent.emit(TenPercentCounter)
                        TenPercentCounter += 1
        NewSelectionIndex = np.argwhere(SelectedIndicator)
        NewSelectionIndex = np.intersect1d(NewSelectionIndex, self.SelectionIndex, assume_unique=True)
        self.TenPercent.emit(10)
        self.SelectionFinish.emit(SelectedIndicator, NewSelectionIndex, self.col_idx)

class DownloadThread(QThread):
    ThirdProgress = Signal(int)
    DownloadFinish = Signal(pd.DataFrame, str)
    def __init__(self, df, FeatureDeleteFlag, CategoricalFlag, CategoricalTransformDict, SelectionIndex, savefile):
        super(DownloadThread, self).__init__()    
        self.df = df.copy()
        self.FeatureDeleteFlag = FeatureDeleteFlag
        self.CategoricalFlag = CategoricalFlag
        self.CategoricalTransformDict = CategoricalTransformDict
        self.SelectionIndex = SelectionIndex
        self.savefile = savefile
    def run(self):
        # First: Transform
        for i in range(len(self.FeatureDeleteFlag)):
            if self.CategoricalFlag[i]:
                self.df.iloc[:,i] = self.df.iloc[:,i].map(lambda x:self.TransformToCategorical(x, i))
        self.ThirdProgress.emit(1)
        # Second: Delete Columns/Features
        DeleteCols = np.argwhere(np.array(self.FeatureDeleteFlag)).reshape(-1,)
        self.df = self.df.drop(columns=self.df.columns[DeleteCols], axis=1)
        self.ThirdProgress.emit(2)
        # Third: Delete Rows/Samples
        self.df = self.df.loc[self.SelectionIndex,:]
        self.ThirdProgress.emit(3)
        self.DownloadFinish.emit(self.df, self.savefile)
        
    def TransformToCategorical(self, x, i):
        if str(x) == 'nan':
            return np.nan
        else:
            return self.CategoricalTransformDict[i][x]

class ImputationThread(QThread):
    Progress = Signal(int, int)
    ImputationFinish = Signal(pd.DataFrame, pd.DataFrame)
    def __init__(self, df, FeatureDeleteFlag, CategoricalFlag, CategoricalTransformDict, SelectionIndex):
        super(ImputationThread, self).__init__()    
        self.df = df.copy()
        self.FeatureDeleteFlag = FeatureDeleteFlag
        self.CategoricalFlag = CategoricalFlag
        self.CategoricalTransformDict = CategoricalTransformDict
        self.SelectionIndex = SelectionIndex
    
    def run(self):
        # ====================OBTAIN THE PREPROCESSED DATA=====================
        # First: Transform
        for i in range(len(self.FeatureDeleteFlag)):
            if self.CategoricalFlag[i]:
                self.df.iloc[:,i] = self.df.iloc[:,i].map(lambda x:self.TransformToCategorical(x, i))
        self.Progress.emit(1, 4)
        # Second: Delete Columns/Features
        DeleteCols = np.argwhere(np.array(self.FeatureDeleteFlag)).reshape(-1,)
        self.df = self.df.drop(columns=self.df.columns[DeleteCols], axis=1)
        self.Progress.emit(2, 4)
        # Third: Delete Rows/Samples
        self.df = self.df.loc[self.SelectionIndex,:]
        self.dfisnull = self.df.isnull()
        self.Progress.emit(3, 4)
        # ======================START THE IMPUTATION===========================
        for column in list(self.df.columns[self.df.isnull().sum() > 0]):
            try:
                mean_val = self.df[column].mean()
                self.df[column].fillna(mean_val, inplace=True)
            except:
                self.df[column].fillna(method='pad', axis=0, inplace=True)
        self.Progress.emit(4, 4)
        self.ImputationFinish.emit(self.df, self.dfisnull)
    
    def TransformToCategorical(self, x, i):
        if str(x) == 'nan':
            return np.nan
        else:
            return self.CategoricalTransformDict[i][x]

class Page3_Widget(QWidget):
    def __init__(self, df, ErrorColIdx, ErrorRowIdx):
        super(Page3_Widget, self).__init__()
        self.initUI(df, ErrorColIdx, ErrorRowIdx)
    def initUI(self, df, ErrorColIdx, ErrorRowIdx):
        num_row, num_col = df.shape
        self.num_col = num_col
        if num_row == 0:
            self.MissingRate = 0
        else:
            self.MissingRate = 100 * df.isnull().sum().sum() / (num_row * num_col)
        # ==================================TOOL MENU===============================
        self.MissingRateLE = QLabel("Missing Rate: %f%%" %self.MissingRate)
        self.MissingRateLE.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
            }
        ''')
        self.FeatureNumLE = QLabel("Feature Number: %i" %num_col)
        self.FeatureNumLE.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
                margin-left: 38px;
            }
        ''')        
        self.SampleNumLE = QLabel("Sample Number: %i" %num_row)
        self.SampleNumLE.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
                margin-left: 38px;
            }
        ''')  

        GoToLineLE = QLabel("Go To Page")
        GoToLineLE.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
            }
        ''')
        self.InputLineNumber = QLineEdit()
        intval = QIntValidator()
        self.InputLineNumber.setValidator(intval)
        self.InputLineNumber.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
                margin: 10px;
                border: 2px solid #A0A0A0;
                border-radius: 8px;
                width: 80px;
            }
        ''')
        self.TotalNumLine = QLabel("/%i" %(int(num_row/100)+1))
        self.TotalNumLine.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
            }
        ''')
        self.UndoButton = QPushButton(" Undo")
        self.UndoButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 120px;
            }
            QPushButton:hover{
                font: bold 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
        self.UndoButton.setEnabled(False)
        self.DownloadButton = QPushButton(" Download")
        self.DownloadButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:29px 29px;
                width: 220px;
            }
            QPushButton:hover{
                font: bold 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:29px 29px;
            }
        ''')
        # self.DownloadButton.setMaximumWidth(130)
        # self.DownloadButton.setMinimumWidth(130)        

        layout = QHBoxLayout()
        layout.addWidget(self.MissingRateLE, 0)
        layout.addWidget(self.SampleNumLE, 0)
        layout.addWidget(self.FeatureNumLE, 0)
        layout.addWidget(QLabel(), 1)
        layout.addWidget(GoToLineLE, 0)
        layout.addWidget(self.InputLineNumber, 0)
        layout.addWidget(self.TotalNumLine, 0)
        layout.addWidget(self.UndoButton, 0)
        layout.addWidget(self.DownloadButton,0)
        layout.setSpacing(0)

        # ===============================MAIN WINDOW================================
        self.MainWindow = QTableWidget()
        display_num_row = min(100, num_row)
        self.MainWindow.setRowCount(max(30, display_num_row))
        self.MainWindow.setColumnCount(max(15, num_col))
        self.MainWindow.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.MainWindow.setHorizontalHeaderLabels(list(df.columns))
        for i in range(display_num_row):
            for j in range(num_col):
                CurItem = str(df.iat[i, j])
                if CurItem == 'nan':
                    EmptyTableItem = QTableWidgetItem()
                    EmptyTableItem.setBackground(QBrush(QColor(201,252,255)))
                    self.MainWindow.setItem(i, j, EmptyTableItem)
                    continue
                elif j in ErrorColIdx:
                    if [i] in ErrorRowIdx[ErrorColIdx.index(j)]:
                        ErrorTableItem = QTableWidgetItem(CurItem)
                        ErrorTableItem.setBackground(QBrush(QColor(255, 79, 66, 150)))
                        self.MainWindow.setItem(i, j, ErrorTableItem)
                        continue
                self.MainWindow.setItem(i, j, QTableWidgetItem(CurItem))
        tablefont = QFont()
        tablefont.setPointSize(11)
        self.MainWindow.setFont(tablefont)
        self.MainWindow.horizontalHeader().setFont(tablefont)
        self.MainWindow.resizeColumnsToContents()
        self.MainWindow.setContextMenuPolicy(Qt.CustomContextMenu)
        self.MainWindow.setAlternatingRowColors(True)
        self.MainWindow.setStyleSheet('''
            QTableWidget{
                alternate-background-color:#f9f9f9;
            }
        ''')
        # =============================BACK-NEXT BUTTON===============================
        nextbutton_layout = QHBoxLayout()
        self.PreprocessProgressLE = QLabel("Preprocess: ")
        self.PreprocessProgressLE.setStyleSheet('*{font-size:25px;}')
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(10)
        self.progressbar.setMinimumWidth(600)
        self.progressbar.setMaximumWidth(600)
        self.progressbar.setStyleSheet('''
            QProgressBar{
                background-color: #E6F4F1;
                color: #3F4756;
                border-radius: 10px;
                border-color: transparent;
                text-align: center;
            }
            QProgressBar::chunk{
                background-color: #00B1AE;
                border-radius: 10px;
            }
        ''')
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        ''')        
        self.next_button = QPushButton("Next")
        self.next_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        ''')
        nextbutton_layout.addWidget(self.PreprocessProgressLE, 0)
        nextbutton_layout.addWidget(self.progressbar, 0)
        nextbutton_layout.addWidget(QLabel(), 1)
        nextbutton_layout.addWidget(self.back_button, 0)
        nextbutton_layout.addWidget(self.next_button, 0)  
        self.PreprocessProgressLE.hide()
        self.progressbar.hide()
        # ==================================OVERALL LAYOUT================================
        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(layout)
        OverallLayout.addWidget(self.MainWindow)
        OverallLayout.addLayout(nextbutton_layout)   
        self.setLayout(OverallLayout) 

class SelectionDialog(QDialog):
    def __init__(self):
        super(SelectionDialog, self).__init__()
        self.initUI()

    def initUI(self):  
        self.setWindowTitle("Selection Methods")
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        layout = QVBoxLayout()
        placeholder = QLabel()
        # ===============LAYOUT 1===================
        layout1 = QHBoxLayout()
        SelectionMethodLE = QLabel("Selection Methods")
        SelectionMethodLE.setProperty('name','in')
        layout1.addWidget(QLabel(),1)
        layout1.addWidget(SelectionMethodLE,0)
        layout1.addWidget(QLabel(),1)

        # ===============LAYOUT 2===================
        layout2 = QHBoxLayout()
        self.SelectButton1 = QRadioButton("Range Selection")
        self.SelectButton1.setProperty('name','in')
        layout2.addWidget(self.SelectButton1, 0)
        layout2.addWidget(placeholder, 0)

        # ===============LAYOUT 3===================
        layout3 = QHBoxLayout()
        self.RangeLE1 = QLineEdit()
        ToLE = QLabel(" to ")
        ToLE.setProperty('name','in')
        self.RangeLE2 = QLineEdit()
        self.RangeLE1.setEnabled(False)
        self.RangeLE2.setEnabled(False)
        self.RangeLE1.setProperty('name','in')
        self.RangeLE2.setProperty('name','in')
        doubleval = QDoubleValidator()
        self.RangeLE1.setValidator(doubleval)
        self.RangeLE2.setValidator(doubleval)
        layout3.addWidget(self.RangeLE1, 0)
        layout3.addWidget(ToLE, 0)
        layout3.addWidget(self.RangeLE2, 0)
        layout3.addWidget(placeholder, 1)

        # ===============LAYOUT 4===================
        layout4 = QHBoxLayout()
        self.SelectButton2 = QRadioButton("Conditional Selection")
        self.SelectButton2.setProperty('name','in')
        layout4.addWidget(self.SelectButton2, 0)
        layout4.addWidget(placeholder, 1)

        # ===============LAYOUT 5===================
        layout5 = QHBoxLayout()
        self.SelectionMethodComboBox = QComboBox()
        self.SelectionMethodComboBox.addItems(['Equal', 'Not Equal', 'Greater than', 'Greater/Equal', 'Smaller than', 'Smaller/Equal', 'Start with', 'Not Start With', 'End with', 'Not End with', 'Contain', 'Not Contain'])
        self.ValueLE = QLineEdit()
        self.SelectionMethodComboBox.setEnabled(False)
        self.ValueLE.setEnabled(False)
        self.SelectionMethodComboBox.setProperty('name','in')
        self.ValueLE.setProperty('name','in')
        layout5.addWidget(self.SelectionMethodComboBox)
        layout5.addWidget(self.ValueLE)
        layout5.addWidget(placeholder, 1)

        # ===============LAYOUT 6===================
        layout6 = QHBoxLayout()
        self.OKButton = QPushButton("OK")
        self.CancelButton = QPushButton("Cancel")
        self.OKButton.setProperty('name','confirm')
        self.CancelButton.setProperty('name','confirm')
        layout6.addWidget(QLabel(), 1)
        layout6.addWidget(self.OKButton, 0)
        layout6.addWidget(self.CancelButton, 0)
        layout6.addWidget(QLabel(), 1)

        # ================LAYOUT ALL==================
        layout.addLayout(layout1)
        layout.addLayout(layout2)
        layout.addLayout(layout3)
        layout.addLayout(layout4)
        layout.addLayout(layout5)
        layout.addLayout(layout6)
        self.setStyleSheet('''
            *[name='in']{
                font: 25px "Microsoft YaHei UI";
            }
            QPushButton[name='confirm']{
                background-color: #326aa9;
                font: 25px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 50px;
                width: 125px;
                margin: 15px;
            }
            QPushButton:hover[name='confirm']{
                background-color: #457fbf;
                font: bold 25px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 50px;
                width: 125px;
            }
            QLineEdit{
                border: 2px solid #A0A0A0;
                border-radius: 8px;
            }
            QComboBox{
                border: 2px solid #A0A0A0;
                border-radius: 8px;
                height: 40px;
                width: 220px;
            }
        ''')
        SelectionMethodLE.setStyleSheet('''
        *{
            font-size: 28px;
        }
        ''')
        self.setLayout(layout)

class RewriteDialog(QDialog):
    def __init__(self, currentvalue):
        super(RewriteDialog, self).__init__()
        self.initUI(currentvalue)

    def initUI(self, currentvalue): 
        self.setWindowTitle("Rewrite Value")
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        CurrentValueLE = QLabel("Current Value: "+currentvalue)
        CurrentValueLE.setStyleSheet('font: 25px "Microsoft YaHei UI";')
        NewValueLE = QLabel("Rewrite Value: ")
        NewValueLE.setStyleSheet('font: 25px "Microsoft YaHei UI";')
        self.Newvalue = QLineEdit()
        self.Newvalue.setStyleSheet('''
            *{
                font: 25px "Microsoft YaHei UI";
                border: 2px solid #A0A0A0;
                border-radius: 8px;
            }
        ''')
        hlayout1 = QHBoxLayout()
        hlayout1.addWidget(CurrentValueLE, 0)
        hlayout1.addWidget(QLabel(), 1)
        hlayout2 = QHBoxLayout()
        hlayout2.addWidget(NewValueLE, 0)
        hlayout2.addWidget(self.Newvalue, 0)
        hlayout2.addWidget(QLabel(), 1)
        self.OKButton = QPushButton("OK")
        self.CancelButton = QPushButton("Cancel")
        self.OKButton.setProperty('name', 'page5_confirm')
        self.CancelButton.setProperty('name', 'page5_confirm')
        self.setStyleSheet('''
            QPushButton[name='page5_confirm']{
                background-color: #326aa9;
                font: 25px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 50px;
                width: 125px;
                margin: 15px;
            }
            QPushButton:hover[name='page5_confirm']{
                background-color: #457fbf;
                font: bold 25px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 50px;
                width: 125px;
            }
        ''')
        hlayout3 = QHBoxLayout()
        hlayout3.addWidget(QLabel(), 1)
        hlayout3.addWidget(self.OKButton, 0)
        hlayout3.addWidget(self.CancelButton, 0)
        hlayout3.addWidget(QLabel(), 1)
        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(hlayout1)
        OverallLayout.addLayout(hlayout2)
        OverallLayout.addLayout(hlayout3)
        self.setLayout(OverallLayout)
        
class Page4_Widget(QWidget):
    def __init__(self):
        super(Page4_Widget, self).__init__()
        self.initUI()
    def initUI(self): 
        # ==========================ALGORITHM UPLOAD===========================
        # (1) Row1: Title
        AlgLabelLayout = QHBoxLayout()
        AlgLabel = QLabel("Upload Algorithm File")
        AlgLabel.setStyleSheet('''
            *{
                font: 37px "Microsoft YaHei UI";
                margin-bottom: 19px;
                color: #666666;
            }
        ''')
        AlgLabelLayout.addWidget(QLabel(), 1)
        AlgLabelLayout.addWidget(AlgLabel, 0)
        AlgLabelLayout.addWidget(QLabel(), 1)
        # (2) Row2: Button
        AlgUploadButtonLayout = QHBoxLayout()
        self.AlgUploadButton = QPushButton("Select File")
        self.AlgUploadButton.setProperty('name', 'upload_button')
        AlgUploadButtonLayout.addWidget(QLabel(), 1)
        AlgUploadButtonLayout.addWidget(self.AlgUploadButton, 0)
        AlgUploadButtonLayout.addWidget(QLabel(), 1)
        AlgSelectLayout = QVBoxLayout()
        AlgSelectLayout.addWidget(QLabel(), 1)
        AlgSelectLayout.addLayout(AlgLabelLayout, 0)
        AlgSelectLayout.addLayout(AlgUploadButtonLayout, 0)
        AlgSelectLayout.addWidget(QLabel(), 1)
        # (3) Row3: File Name
        FileNameLabel = QLabel("File Name")
        FileNameLabel.setStyleSheet('''
            *{
                font: 35px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        # (4) Row4: A Horizontal Line
        hline = QFrame()
        hline.setFrameShape(QFrame.HLine)
        hline.setLineWidth(5)
        hline.setStyleSheet('''
            color: #666666;
        ''')
        # (5) Row5: Display and Delete Table
        self.AlgFileTable = QTableWidget()
        self.AlgFileTable.setStyleSheet('''
            border: none;
        ''')
        self.AlgFileTable.setColumnCount(2)
        self.AlgFileTable.setRowCount(0)
        self.AlgFileTable.horizontalHeader().setStretchLastSection(True)
        self.AlgFileTable.horizontalHeader().hide()
        self.AlgFileTable.verticalHeader().hide()
        self.AlgFileTable.setShowGrid(False)
        self.AlgFileTable.setStyleSheet('''
            font-size: 29px;
        ''')
        self.AlgFileTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.AlgFileTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        AlgTableLayout = QVBoxLayout()
        AlgTableLayout.addWidget(QLabel(), 1)
        AlgTableLayout.addWidget(FileNameLabel, 0)
        AlgTableLayout.addWidget(hline, 0)
        AlgTableLayout.addWidget(self.AlgFileTable, 0)
        AlgTableLayout.addWidget(QLabel(), 1)
        AlgLayout = QHBoxLayout()
        AlgLayout.addLayout(AlgSelectLayout, 0)
        AlgLayout.addLayout(AlgTableLayout, 1)
        # ==========================PARAMETER UPLOAD===========================
        ParSelectLayout = QVBoxLayout()
        # (1) Row1: Title
        ParLabelLayout = QHBoxLayout()
        ParLabel = QLabel("Upload Parameter File")
        ParLabel.setStyleSheet('''
            *{
                font: 37px "Microsoft YaHei UI";
                margin-bottom: 19px;
                color: #666666;
            }
        ''')
        ParLabelLayout.addWidget(QLabel(), 1)
        ParLabelLayout.addWidget(ParLabel, 0)
        ParLabelLayout.addWidget(QLabel(), 1)
        # (2) Row2: Button
        ParUploadButtonLayout = QHBoxLayout()
        self.ParUploadButton = QPushButton("Select File")
        self.ParUploadButton.setProperty('name', 'upload_button')
        ParUploadButtonLayout.addWidget(QLabel(), 1)
        ParUploadButtonLayout.addWidget(self.ParUploadButton, 0)
        ParUploadButtonLayout.addWidget(QLabel(), 1)
        ParSelectLayout.addWidget(QLabel(), 1)
        ParSelectLayout.addLayout(ParLabelLayout, 0)
        ParSelectLayout.addLayout(ParUploadButtonLayout, 0)
        ParSelectLayout.addWidget(QLabel(), 1)

        # (3) Row3: File Name
        FileNameLabel1 = QLabel("File Name")
        FileNameLabel1.setStyleSheet('''
            *{
                font: 35px "Microsoft YaHei UI";
                color: #666666;
            }
        ''')
        # (4) Row4: A Horizontal Line
        hline1 = QFrame()
        hline1.setFrameShape(QFrame.HLine)
        hline1.setLineWidth(5)
        hline1.setStyleSheet('''
            color: #666666;
        ''')
        # (5) Row5: Display and Delete Table
        self.ParFileTable = QTableWidget()
        self.ParFileTable.setStyleSheet('''
            border: none;
        ''')
        self.ParFileTable.setColumnCount(2)
        self.ParFileTable.setRowCount(0)
        self.ParFileTable.horizontalHeader().setStretchLastSection(True)
        self.ParFileTable.horizontalHeader().hide()
        self.ParFileTable.verticalHeader().hide()
        self.ParFileTable.setShowGrid(False)
        self.ParFileTable.setStyleSheet('''
            font-size: 29px;
        ''')
        self.ParFileTable.setSelectionMode(QAbstractItemView.NoSelection)
        self.ParFileTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        ParTableLayout = QVBoxLayout()
        ParTableLayout.addWidget(QLabel(), 1)
        ParTableLayout.addWidget(FileNameLabel1, 0)
        ParTableLayout.addWidget(hline1, 0)
        ParTableLayout.addWidget(self.ParFileTable, 0)
        ParTableLayout.addWidget(QLabel(), 1)   
        ParLayout = QHBoxLayout()
        ParLayout.addLayout(ParSelectLayout, 0)
        ParLayout.addLayout(ParTableLayout, 1)
        # =========================NEXT-BACK BUTTON=========================
        nextbutton_layout = QHBoxLayout()
        self.ImputeProgressLE = QLabel("Imputation Process: ")
        self.ImputeProgressLE.setStyleSheet('*{font-size:25px;}')
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMinimumWidth(600)
        self.progressbar.setMaximumWidth(600)
        self.progressbar.setStyleSheet('''
            QProgressBar{
                background-color: #E6F4F1;
                color: #3F4756;
                border-radius: 10px;
                border-color: transparent;
                text-align: center;
            }
            QProgressBar::chunk{
                background-color: #00B1AE;
                border-radius: 10px;
            }
        ''')
        self.back_button = QPushButton("Back")
        self.back_button.setProperty('name','next_back_button')       
        self.next_button = QPushButton("Next")
        self.next_button.setProperty('name','next_back_button')
        nextbutton_layout.addWidget(self.ImputeProgressLE, 0)
        nextbutton_layout.addWidget(self.progressbar, 0)
        nextbutton_layout.addWidget(QLabel(), 1)
        nextbutton_layout.addWidget(self.back_button, 0)
        nextbutton_layout.addWidget(self.next_button, 0)  
        self.ImputeProgressLE.hide()
        self.progressbar.hide()  
        # ============================OVERALL LAYOUT============================
        self.setStyleSheet('''
            QPushButton[name='upload_button']{
                background-color: #326AA9;
                font: 37px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 20px;
                height: 103px;
                width: 310px;
                qproperty-icon:url(fig/folder.png);
                qproperty-iconSize:55px 55px;
                text-align: center;
                margin-bottom: 39px;
            }
            QPushButton:hover[name='upload_button']{
                background-color: #457FBF;
                font: bold 37px "Microsoft YaHei UI";
                color: #F5F9FF;
                border: none;
                border-radius: 20px;
                height: 103px;
                width: 310px;
            }
            QPushButton[name='next_back_button']{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover[name='next_back_button']{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        '''
        )
        OverallLayout = QVBoxLayout()
        placeholder = QLabel()
        placeholder.setMaximumHeight(30)
        placeholder.setMinimumHeight(30)
        OverallLayout.addLayout(AlgLayout)
        OverallLayout.addLayout(ParLayout)
        OverallLayout.addWidget(placeholder)
        OverallLayout.addLayout(nextbutton_layout)
        self.setLayout(OverallLayout)    

class Page5_Widget(QWidget):
    def __init__(self, impute_df, dfisnull):
        super(Page5_Widget, self).__init__()
        self.initUI(impute_df, dfisnull)
    def initUI(self, impute_df, dfisnull):
        num_row, num_col = impute_df.shape
        # ==============================TOOL MENU============================
        GoToLineLE= QLabel("Go To Page")
        GoToLineLE.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
            }
        ''')
        self.InputLineNumber = QLineEdit()
        intval = QIntValidator()
        self.InputLineNumber.setValidator(intval)
        self.InputLineNumber.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
                margin: 10px;
                border: 2px solid #A0A0A0;
                border-radius: 8px;
                width: 80px;
            }
        ''')
        self.TotalNumLine = QLabel("/%i" %(int(num_row/100)+1))
        self.TotalNumLine.setStyleSheet('''
            *{
                font: 28px "Microsoft YaHei UI";
            }
        ''')
        self.RedoButton = QPushButton(" Re-impute")
        self.RedoButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/redo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 220px;
            }
            QPushButton:hover{
                font-size: 29px;
                font: bold;
                border: none;
                background-color: none;
                color: #bfbfbf;
                qproperty-icon:url(fig/redo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
        self.RedoButton.setEnabled(False)
        self.DownloadButton = QPushButton(" Download")
        self.DownloadButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:29px 29px;
                width: 220px;
            }
            QPushButton:hover{
                font: bold 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/download.png);
                qproperty-iconSize:29px 29px;
            }
        ''')
        ToolMenuLayout = QHBoxLayout()
        ToolMenuLayout.addWidget(QLabel(), 1)
        ToolMenuLayout.addWidget(GoToLineLE, 0)
        ToolMenuLayout.addWidget(self.InputLineNumber, 0)
        ToolMenuLayout.addWidget(self.TotalNumLine, 0)
        # ToolMenuLayout.addWidget(self.RedoButton, 0)
        ToolMenuLayout.addWidget(self.DownloadButton, 0)
        # =============================MAIN WINDOW=============================
        self.MainWindow = QTableWidget()
        display_num_row = min(100, num_row)
        self.MainWindow.setRowCount(max(30, display_num_row))
        self.MainWindow.setColumnCount(max(15, num_col))
        self.MainWindow.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.MainWindow.setHorizontalHeaderLabels(list(impute_df.columns))
        for i in range(display_num_row):
            for j in range(num_col):
                CurItem = QTableWidgetItem(str(impute_df.iat[i, j]))
                if dfisnull.iat[i, j]:
                    CurItem.setBackground(QBrush(QColor(201,252,255)))
                self.MainWindow.setItem(i, j, CurItem)
        tablefont = QFont()
        tablefont.setPointSize(11)
        self.MainWindow.setFont(tablefont)
        self.MainWindow.horizontalHeader().setFont(tablefont)
        self.MainWindow.resizeColumnsToContents()
        self.MainWindow.setContextMenuPolicy(Qt.CustomContextMenu)
        self.MainWindow.setAlternatingRowColors(True)
        self.MainWindow.setStyleSheet('''
            QTableWidget{
                alternate-background-color:#f9f9f9;
            }
        ''')
        # ===============================BACK BUTTON============================
        backbutton_layout = QHBoxLayout()
        ImputationProcessLE = QLabel("Imputation Process: ")
        ImputationProcessLE.setStyleSheet('*{font-size:25px;}')
        self.progressbar = QProgressBar()
        self.progressbar.setMinimum(0)
        self.progressbar.setMaximum(10)
        self.progressbar.setMinimumWidth(600)
        self.progressbar.setMaximumWidth(600)
        self.progressbar.setStyleSheet('''
            QProgressBar{
                background-color: #E6F4F1;
                color: #3F4756;
                border-radius: 10px;
                border-color: transparent;
                text-align: center;
            }
            QProgressBar::chunk{
                background-color: #00B1AE;
                border-radius: 10px;
            }
        ''')
        
        self.back_button = QPushButton("Back")
        self.back_button.setStyleSheet('''
            QPushButton{
                background-color: #326aa9;
                font: 30px "Microsoft YaHei UI";
                color: white;
                border: none;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
            QPushButton:hover{
                background-color: #457fbf;
                font: bold 30px "Microsoft YaHei UI";
                color: white;
                border: white;
                border-radius: 10px;
                height: 60px;
                width: 125px;
            }
        ''')  
        backbutton_layout.addWidget(ImputationProcessLE, 0)
        backbutton_layout.addWidget(self.progressbar, 0)
        backbutton_layout.addWidget(QLabel(), 1)
        backbutton_layout.addWidget(self.back_button)
        ImputationProcessLE.hide()
        self.progressbar.hide()
        # ============================OVERALL LAYOUT===========================
        OverallLayout = QVBoxLayout()
        OverallLayout.addLayout(ToolMenuLayout)
        OverallLayout.addWidget(self.MainWindow)
        OverallLayout.addLayout(backbutton_layout)
        self.setLayout(OverallLayout)

class Imputation_System(QWidget):
    def __init__(self):
        super(Imputation_System, self).__init__()
        self.initUI()

    def initUI(self):
        ''' 
        Title_Widget - Global Title Widget for all 5 pages
        '''
        self.setObjectName("MainWindow")
        self.setWindowTitle("SCIS A Large-Scale Missing Data Imputation System")
        self.setStyleSheet('''
            #MainWindow{
                background-color: #f3f2f1;
            }
        ''')
        #self.Title_Widget = TitleWidget()

        '''
        Step_Widget - Global Step Widget for all 5 pages: It is used to display the previous 
        and current steps. User can also uses it to change to previous step. 
        '''
        self.Step_Widget = StepWidget()

        self.Step_Widget.step_layout_button3.hide()
        self.Step_Widget.step_layout_button4.hide()
        self.Step_Widget.step_layout_button5.hide()
        self.Step_Widget.step_layout_button1.clicked.connect(self.GOTO_Page1)
        self.Step_Widget.step_layout_button2.clicked.connect(self.GOTO_Page2)
        self.Step_Widget.step_layout_button3.clicked.connect(self.GOTO_Page3)
        self.Step_Widget.step_layout_button4.clicked.connect(self.GOTO_Page4)
        self.Step_Widget.step_layout_button5.clicked.connect(self.GOTO_Page5)
        self.button_select_style = '''
            QPushButton{
                border: none;
                height: 80px;
                font-size:35px;
                font: large "Microsoft YaHei UI";
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize:30px 30px;
                text-align: left;
                background-color: #0e83c2;
            }
            QPushButton:hover{
                height: 80px;
                font-size:35px;
                font: bold large "Microsoft YaHei UI";
                background-color: #0e83c2;
            }
        '''
        self.button_release_style = '''
            QPushButton{
                border: none;
                height: 80px;
                font-size:35px;
                font: large "Microsoft YaHei UI";
                qproperty-icon:url(fig/go_right.png);
                qproperty-iconSize:30px 30px;
                text-align: left;
                background-color: #0c6ca1;
            }
            QPushButton:hover{
                height: 80px;
                font-size:35px;
                font: bold large "Microsoft YaHei UI";
                background-color: #0e83c2;
            }
        '''
        self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)

        '''
        Page1_Widget - Local Widget for first page: It is used to upload files and delete them if needed
        '''
        self.pagenum = 1
        self.FirstFileIndex = 0
        self.page2_combobox_init = False
        self.page2_init = False
        self.page3_init = False
        self.page4_init = False
        self.page5_init = False
        self.Page1_Widget = Page1_Widget()
        self.filenames = []
        self.file_validflag = []  # indicate whether the file is still there or deleted
        self.num_allfiles = 0     # number of all ever-uploaded files
        self.AlgorithmFile = None
        self.ParameterFile = None
        self.Page1_Widget.select_button.clicked.connect(self.upload_files)
        self.Page1_Widget.next_button.clicked.connect(self.GOTO_Page2)
        '''
        Overall Display
        '''
        self.Overall_Layout = QVBoxLayout()
        placeholder = QLabel()
        placeholder.setMinimumHeight(20)
        placeholder.setMaximumHeight(20)
        #self.Overall_Layout.addWidget(self.Title_Widget)
        self.Overall_Layout.addWidget(placeholder)
        self.Overall_Layout.addWidget(self.Step_Widget)
        self.Overall_Layout.addWidget(self.Page1_Widget)
        self.setLayout(self.Overall_Layout)

    def upload_files(self):
        newfiles, _ = QFileDialog.getOpenFileNames(self, 'Open Files', '.', '(*.csv)')
        self.filenames.extend(newfiles)
        self.file_validflag += [True]*(len(newfiles))
        current_num_row = self.Page1_Widget.file_table.rowCount()
        self.Page1_Widget.file_table.setRowCount(current_num_row+len(newfiles))
        for i in range(len(newfiles)):
            self.Page1_Widget.file_table.setItem(current_num_row+i, 0, QTableWidgetItem(newfiles[i]))
            self.Page1_Widget.file_table.setCellWidget(current_num_row+i, 1, FileDeleteButton(self.num_allfiles+i))
            self.Page1_Widget.file_table.cellWidget(current_num_row+i, 1).delete_button.clicked.connect(self.delete_file)
        self.Page1_Widget.file_table.resizeColumnsToContents()
        self.Page1_Widget.file_table.horizontalHeader().setStretchLastSection(True)
        
        if self.page2_init:
            for i in range(len(newfiles)):
                newfile_name_idx = max(newfiles[i].rfind('/'), newfiles[i].rfind('\\'))
                newfilename_abbr = newfiles[i][newfile_name_idx+1:]
                self.Page2_Widget.alltabwidgets.append(DataPreviewWidget(newfiles[i], self.num_allfiles+i))
                excel_icon = QIcon(QPixmap('fig/excel.png'))
                self.Page2_Widget.datadisplay_tab.addTab(self.Page2_Widget.alltabwidgets[-1], excel_icon, newfilename_abbr)
                tabbar_font = QFont()
                tabbar_font.setFamily("Microsoft YaHei UI")
                tabbar_font.setPointSize(12)
                self.Page2_Widget.datadisplay_tab.tabBar().setFont(tabbar_font)   
                self.Page2_Widget.datadisplay_tab.tabBar().setIconSize(QSize(32,32))
                self.Page2_Widget.alltabwidgets[-1].filedelete_button.delete_button.clicked.connect(self.delete_file_page2)
                # if self.page2_combobox_init:
                #     self.firstfile_selectionBox.addItem(newfilename_abbr)  
                # When Uploaded Files get changed, we remove page3, page4 and page5
                if self.page2_init:
                    self.Step_Widget.step_layout_button3.hide()
                if self.page3_init:
                    self.page3_init = False
                    self.Step_Widget.step_layout_button4.hide()
                if self.page4_init:
                    self.page4_init = False
                    self.Step_Widget.step_layout_button5.hide()
                if self.page5_init:
                    self.page5_init = False
                              
        self.num_allfiles += len(newfiles)

    def delete_file(self):
        insert_idx = self.sender().property('idx')
        row_idx = sum(self.file_validflag[:insert_idx])
        self.file_validflag[insert_idx] = False
        self.Page1_Widget.file_table.removeRow(row_idx)
        if self.page2_init:
            del self.Page2_Widget.alltabwidgets[row_idx]
            self.Page2_Widget.datadisplay_tab.removeTab(row_idx) 
            # if self.page2_combobox_init:
            #     self.firstfile_selectionBox.removeItem(row_idx)             
            # When Uploaded Files get changed, we remove page3, page4 and page5
            if self.page2_init:
                self.Step_Widget.step_layout_button3.hide()
            if self.page3_init:
                self.page3_init = False
                self.Step_Widget.step_layout_button4.hide()
            if self.page4_init:
                self.page4_init = False
                self.Step_Widget.step_layout_button5.hide()
            if self.page5_init:
                self.page5_init = False        
        # print(self.file_validflag)       

    def delete_file_page2(self):
        insert_idx = self.sender().property('idx')
        row_idx = sum(self.file_validflag[:insert_idx])
        self.file_validflag[insert_idx] = False
        self.Page1_Widget.file_table.removeRow(row_idx)
        del self.Page2_Widget.alltabwidgets[row_idx]
        self.Page2_Widget.datadisplay_tab.removeTab(row_idx)  
        # if self.page2_combobox_init:
        #     self.firstfile_selectionBox.removeItem(row_idx) 

        if self.page3_init:
            self.page3_init = False
            self.Step_Widget.step_layout_button4.hide()
        if self.page4_init:
            self.page4_init = False
            self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False 

    # def set_merge_method(self):
    #     previous_merge_method = self.merge_method
    #     self.merge_method = self.sender().property('merge_method')
    #     if self.merge_method == 'First' and previous_merge_method != 'First':
    #         self.FirstFileIndex = 0
    #         self.Page2_Widget.selectionbox_layout.removeWidget(self.Page2_Widget.placeholder)
           
    #         if self.page2_combobox_init == False:
    #              # ==============Create a ComBox for user to choose 'first' file=================
    #             self.page2_combobox_init= True
    #             layout = QHBoxLayout()
    #             information_label = QLabel("Select the 'first' file to be aligned:")
    #             information_label.setStyleSheet('*{font-size:25px;}')
    #             self.firstfile_selectionBox = QComboBox()
    #             for i in range(len(self.filenames)):            
    #                 if self.file_validflag[i]:
    #                     file_name_idx = max(self.filenames[i].rfind('/'), self.filenames[i].rfind('\\'))
    #                     filename_abbr = self.filenames[i][file_name_idx+1:]    
    #                     self.firstfile_selectionBox.addItem(filename_abbr)
    #             self.firstfile_selectionBox.setStyleSheet('''
    #                 *{
    #                     font-size: 25px;
    #                 }
    #             ''')
    #             self.firstfile_selectionBox.currentIndexChanged.connect(self.setFirstFile)
    #             layout.addWidget(information_label)
    #             layout.addWidget(self.firstfile_selectionBox)
    #             layout.setSpacing(30)
    #             self.firstfile_widget = QWidget()
    #             self.firstfile_widget.setLayout(layout)
    #             # ==============================================================================
    #             self.Page2_Widget.selectionbox_layout.addWidget(self.firstfile_widget, 0)
    #             self.Page2_Widget.selectionbox_layout.addWidget(self.Page2_Widget.placeholder, 1)
    #         else: 
    #             self.Page2_Widget.selectionbox_layout.removeWidget(self.Page2_Widget.placeholder)
    #             self.firstfile_widget.show()
    #             self.Page2_Widget.selectionbox_layout.addWidget(self.Page2_Widget.placeholder, 1)
    #     elif previous_merge_method == 'First' and self.merge_method != 'First':
    #         self.firstfile_widget.hide()
    #     if self.merge_method != previous_merge_method:
    #         if self.page3_init:
    #             self.page3_init = False
    #             self.Step_Widget.step_layout_button4.hide()
    #         if self.page4_init:
    #             self.page4_init = False
    #             self.Step_Widget.step_layout_button5.hide()
    #         if self.page5_init:
    #             self.page5_init = False 
    
    def setFirstFile(self, i):
        self.FirstFileIndex = i
        if self.page3_init:
            self.page3_init = False
            self.Step_Widget.step_layout_button4.hide()
        if self.page4_init:
            self.page4_init = False
            self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False 

    def GOTO_Page1(self):
        if self.pagenum == 1:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 1
        if previous_pagenum == 2:
            self.Page2_Widget.hide()
            self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
            if self.page3_init == False:
                self.Step_Widget.step_layout_button3.hide()
        elif previous_pagenum == 3:
            self.Page3_Widget.hide()
            self.Step_Widget.step_layout_button3.setStyleSheet(self.button_release_style)
            if self.page4_init == False:
                self.Step_Widget.step_layout_button4.hide()
        elif previous_pagenum == 4:
            self.Page4_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            if self.page5_init == False:
                self.Step_Widget.step_layout_button5.hide()
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            self.Step_Widget.step_layout_button5.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button1.setStyleSheet(self.button_select_style)
        self.Page1_Widget.show()

    def GOTO_Page2(self):

        # First check whether jump from elsewhere
        if self.pagenum == 2:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 2
        # This part only applies when jump from the first page: if there is no file uploader
        num_upload_file = sum(self.file_validflag)
        if num_upload_file == 0:
            QMessageBox.information(self, "No File Uploaded", "There are no files uploaded. Please upload at least one file.", QMessageBox.Yes, QMessageBox.Yes)
            self.pagenum = 1
            return
        # Check whether page2 has already be initalized
        page2_init_copy = self.page2_init
        if self.page2_init == False:
            self.page2_init = True
            self.Page2_Widget = Page2_Widget(self.filenames, self.file_validflag)
            for i in range(len(self.Page2_Widget.alltabwidgets)):
                self.Page2_Widget.alltabwidgets[i].filedelete_button.delete_button.clicked.connect(self.delete_file_page2)
            self.Page2_Widget.back_button.clicked.connect(self.GOTO_Page1)
            # self.Page2_Widget.radiobutton1.clicked.connect(self.set_merge_method)
            # self.Page2_Widget.radiobutton2.clicked.connect(self.set_merge_method)
            # self.Page2_Widget.radiobutton3.clicked.connect(self.set_merge_method)
            self.Page2_Widget.next_button.clicked.connect(self.GOTO_Page3)
            self.Overall_Layout.addWidget(self.Page2_Widget)

        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
            self.Step_Widget.step_layout_button3.show()
        elif previous_pagenum == 3:
            self.Page3_Widget.hide()
            self.Step_Widget.step_layout_button3.setStyleSheet(self.button_release_style)
            if self.page4_init == False:
                self.Step_Widget.step_layout_button4.hide()
        elif previous_pagenum == 4:
            self.Page4_Widget.hide()
            self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            if self.page5_init == False:
                self.Step_Widget.step_layout_button5.hide()
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            self.Step_Widget.step_layout_button5.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button2.setStyleSheet(self.button_select_style)
        self.Page2_Widget.show()
        # if self.page3_init:
        #     self.Page2_Widget.progressbar.hide()
        #     self.Page2_Widget.MergeProgressLE.hide()

    def GOTO_Page3(self):
        # This part only applies when jumping from the first and second part: when no files are 
        # uploaded or all of them are deleted
        num_upload_file = sum(self.file_validflag)
        if num_upload_file == 0:
            QMessageBox.information(self, "No File Uploaded", "There are no files uploaded. Please upload at least one file.", QMessageBox.Yes, QMessageBox.Yes)
            return        

        # Check whether jump from elsewhere
        if self.pagenum == 3:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 3

        # self.Step_Widget.step_layout_button3.setStyleSheet(self.button_select_style)

        # Check whether page3 has already be initialized
        if self.page3_init == False:
            self.page3_init = True
            self.page3_MergeFileThread = MergeFileThread(self.filenames, self.file_validflag, previous_pagenum)
            self.page3_MergeFileThread.MergeFileFinish.connect(self.Page3_DisplayTable)
            self.page3_MergeFileThread.TwoFileMerge.connect(self.DisplayPage2ProgressBar)
            self.DisablePage2Buttons()
            self.page3_MergeFileThread.start()
        else: 
            if previous_pagenum == 1:
                self.Page1_Widget.hide()
                self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 2:
                self.Page2_Widget.hide()
                self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 4:
                self.Page4_Widget.hide()
                self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
                if self.page5_init == False:
                    self.Step_Widget.step_layout_button5.hide()
            elif previous_pagenum == 5:
                self.Page5_Widget.hide()
                self.Step_Widget.step_layout_button5.setStyleSheet(self.button_release_style)
            self.Step_Widget.step_layout_button4.show()
            self.Step_Widget.step_layout_button3.setStyleSheet(self.button_select_style)
            self.Page3_Widget.show()

    def GOTO_Page4(self):
        # First check whether jump from elsewhere
        if self.pagenum == 4:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 4

        # Check whether page4 has already been initialized
        if self.page4_init == False:
            self.page4_init = True
            self.AlgorithmFile = None
            self.ParameterFile = None
            self.Page4_Widget = Page4_Widget()
            self.Overall_Layout.addWidget(self.Page4_Widget)
            self.Page4_Widget.AlgUploadButton.clicked.connect(self.Page4UploadAlg)
            self.Page4_Widget.ParUploadButton.clicked.connect(self.Page4UploadPar)
            self.Page4_Widget.back_button.clicked.connect(self.GOTO_Page3)
            self.Page4_Widget.next_button.clicked.connect(self.GOTO_Page5)
            
        
        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 2:
            self.Page2_Widget.hide()
            self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
            self.Step_Widget.step_layout_button4.show()
        elif previous_pagenum == 3:
            self.Page3_Widget.hide()
            self.Step_Widget.step_layout_button3.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 5:
            self.Page5_Widget.hide()
            self.Step_Widget.step_layout_button5.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button4.setStyleSheet(self.button_select_style)
        self.Page4_Widget.show()
        self.Step_Widget.step_layout_button5.show()

    def GOTO_Page5(self):
        # This part only applies when jumping from page4: when no algorithm/parameter file are uploaded
        if self.AlgorithmFile == None:
            QMessageBox.information(self, "No File Uploaded", "There are no algorithm file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
            return 
        if self.ParameterFile == None:
            QMessageBox.information(self, "No File Uploaded", "There are no parameter file uploaded. Please upload one.", QMessageBox.Yes, QMessageBox.Yes)
            return 
        
        # Check whether jump from elsewhere
        if self.pagenum == 5:
            return
        else:
            previous_pagenum = self.pagenum
            self.pagenum = 5

        # Check whether page5 has already been initialized
        if self.page5_init == False:
            self.page5_init = True
            self.page5_ImputationThread = ImputationThread(self.df, self.FeatureDeleteFlag, self.CategoricalFlag, self.CategoricalTransformDict, self.SelectionIndex)
            self.page5_ImputationThread.ImputationFinish.connect(self.Page5_DisplayTable)
            self.page5_ImputationThread.Progress.connect(self.DisplayPage4ProgressBar)
            self.DisablePage4Buttons()
            self.page5_ImputationThread.start()
        else:
            if previous_pagenum == 1:
                self.Page1_Widget.hide()
                self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 2:
                self.Page2_Widget.hide()
                self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 3:
                self.Page3_Widget.hide()
                self.Step_Widget.step_layout_button3.setStyleSheet(self.button_release_style)
            elif previous_pagenum == 4:
                self.Page4_Widget.hide()
                self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
            self.Step_Widget.step_layout_button5.setStyleSheet(self.button_select_style)
            self.Page5_Widget.show()

    

    def Page3_DisplayTable(self, df, previous_pagenum, ErrorColIdx, ErrorRowIdx):
        self.EnablePage2Buttons()
        self.df = df
        self.MissingNum = self.df.isnull().sum().sum()
        self.ErrorColIdx = ErrorColIdx
        self.ErrorRowIdx = ErrorRowIdx
        num_sample, num_feature = df.shape
        self.FeatureDeleteFlag = [False] * num_feature
        self.CategoricalFlag = [False] * num_feature
        self.CategoricalTransformDict = [None] * num_feature
        self.SelectionFlag = [False] * num_feature
        self.SelectionMethod = None
        self.SelectionRangeUp = None
        self.SelectionRangeDown = None
        self.SelectionCondition = 0
        self.SelectionValue = None
        self.SelectionIndicator = [None] * num_feature
        self.SelectionIndex = np.array([i for i in range(num_sample)])
        self.ActionStack = []
        self.Page3_Widget = Page3_Widget(self.df, ErrorColIdx, ErrorRowIdx)
        self.Page3_Widget.back_button.clicked.connect(self.GOTO_Page2)
        self.Page3_Widget.next_button.clicked.connect(self.GOTO_Page4)
        self.Page3_Widget.InputLineNumber.editingFinished.connect(lambda: self.ChangePageDisplay())
        self.Page3_Widget.MainWindow.customContextMenuRequested.connect(self.generateMenu)
        self.Page3_Widget.DownloadButton.clicked.connect(self.DownloadPreprocessFile)
        self.Page3_Widget.UndoButton.clicked.connect(self.UndoPreprocess)
        self.Overall_Layout.addWidget(self.Page3_Widget)
        if previous_pagenum == 1:
            self.Page1_Widget.hide()
            self.Step_Widget.step_layout_button1.setStyleSheet(self.button_release_style)
        elif previous_pagenum == 2:
            self.Page2_Widget.hide()
            self.Step_Widget.step_layout_button2.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button4.show()
        self.Step_Widget.step_layout_button3.setStyleSheet(self.button_select_style)
        self.Page3_Widget.show()

    def DisplayPage2ProgressBar(self, ProgressState, TotalFile):
        if ProgressState == 0:
            self.Page2_Widget.progressbar.setMaximum(TotalFile+1)
            self.Page2_Widget.MergeProgressLE.show()
            self.Page2_Widget.progressbar.show()
            self.Page2_Widget.progressbar.setValue(2)
        else:
            self.Page2_Widget.progressbar.setValue(ProgressState)
            if ProgressState == TotalFile + 1:
                self.Page2_Widget.progressbar.hide()
                self.Page2_Widget.MergeProgressLE.hide()

    def ChangePageDisplay(self):
        # =======================Dealing with page-jumping======================
        num_row, num_col = self.df.shape
        display_num_row = len(self.SelectionIndex)
        try:
            PageNumber = int(self.Page3_Widget.InputLineNumber.text())
        except:
            PageNumber = 1
        TotalPage = int(display_num_row/100) + 1
        self.Page3_Widget.TotalNumLine.setText("/"+str(TotalPage))
        if PageNumber > TotalPage:
            PageNumber = TotalPage
            self.Page3_Widget.InputLineNumber.setText(str(TotalPage))
            QMessageBox.warning(self, "Warning", "Page number out of, jump to the last page!", QMessageBox.Yes, QMessageBox.Yes)
        elif PageNumber <= 0:
            PageNumber = 1
            self.Page3_Widget.InputLineNumber.setText(str(1))
            QMessageBox.warning(self, "Warning", "Page number out of, jump to the first page!", QMessageBox.Yes, QMessageBox.Yes)
        display_num_col = num_col - sum(self.FeatureDeleteFlag)
        NewMissingRate = self.MissingNum * 100 / (display_num_col*display_num_row)
        self.Page3_Widget.MissingRateLE.setText("Missing Rate: %f%%" %NewMissingRate)
        self.Page3_Widget.SampleNumLE.setText("Sample Number: %i" %display_num_row)
        self.Page3_Widget.FeatureNumLE.setText("Feature Number: %i" %display_num_col)
        self.Page3_Widget.MainWindow.clear()
        self.Page3_Widget.MainWindow.setColumnCount(max(15, display_num_col))
        # get the new headers
        HorizontalHeader = [' '] * max(15, display_num_col)
        i_idx = 0
        for i in range(len(self.df.columns)):
            if self.FeatureDeleteFlag[i] == False:
                HorizontalHeader[i_idx] = self.df.columns[i]
                i_idx += 1
        self.Page3_Widget.MainWindow.setHorizontalHeaderLabels(HorizontalHeader)

        start_row = 100 * (PageNumber - 1)
        end_row = min(display_num_row, PageNumber * 100)
        for i in range(start_row, end_row):
            i_df = self.SelectionIndex[i]
            j_idx = 0
            for j in range(num_col):
                if self.FeatureDeleteFlag[j]:
                    continue
                if self.CategoricalFlag[j]:
                    if str(self.df.iat[i_df, j]) == 'nan':
                        CurItem = 'nan'
                    else:
                        try: 
                            CurItem = str(self.CategoricalTransformDict[j][self.df.iat[i_df, j]])
                        except:
                            print(j)
                            print(len(self.CategoricalTransformDict))
                            print(self.df.iat[i_df,j])
                            print(self.df.shape)
                            str(self.CategoricalTransformDict[j][self.df.iat[i_df, j]])
                else:
                    CurItem = str(self.df.iat[i_df, j])
                if CurItem == 'nan':
                    EmptyTableItem = QTableWidgetItem()
                    EmptyTableItem.setBackground(QBrush(QColor(201,252,255)))
                    self.Page3_Widget.MainWindow.setItem(i-start_row, j_idx, EmptyTableItem)
                    j_idx += 1  
                    continue
                elif j in self.ErrorColIdx:
                    if [i_df] in self.ErrorRowIdx[self.ErrorColIdx.index(j)]:
                        ErrorTableItem = QTableWidgetItem(CurItem)
                        ErrorTableItem.setBackground(QBrush(QColor(255, 79, 66, 150)))
                        self.Page3_Widget.MainWindow.setItem(i-start_row, j_idx, ErrorTableItem)
                        j_idx += 1  
                        continue
                self.Page3_Widget.MainWindow.setItem(i-start_row, j_idx, QTableWidgetItem(CurItem))         
                j_idx += 1   
        vertical_header = [str(i+1) for i in range(start_row, start_row+self.Page3_Widget.MainWindow.rowCount())]
        self.Page3_Widget.MainWindow.setVerticalHeaderLabels(vertical_header)
        self.Page3_Widget.MainWindow.verticalHeader().show()
        self.Page3_Widget.MainWindow.resizeColumnsToContents()

    def DisablePage2Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        self.Step_Widget.step_layout_button2.setEnabled(False)
        self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        self.Step_Widget.step_layout_button5.setEnabled(False)
        # ========================DELETION BUTTONS=============================
        for i in range(len(self.Page2_Widget.alltabwidgets)):
            self.Page2_Widget.alltabwidgets[i].filedelete_button.delete_button.setEnabled(False)
        # ========================MERGE SELECTION==============================
        # self.Page2_Widget.radiobutton1.setEnabled(False)
        # self.Page2_Widget.radiobutton2.setEnabled(False)
        # self.Page2_Widget.radiobutton3.setEnabled(False)
        # ========================BACK-NEXT BUTTON=============================
        self.Page2_Widget.back_button.setEnabled(False)
        self.Page2_Widget.next_button.setEnabled(False)

    def EnablePage2Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        self.Step_Widget.step_layout_button2.setEnabled(True)
        self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        self.Step_Widget.step_layout_button5.setEnabled(True)
        # ========================DELETION BUTTONS=============================
        for i in range(len(self.Page2_Widget.alltabwidgets)):
            self.Page2_Widget.alltabwidgets[i].filedelete_button.delete_button.setEnabled(True)
        # ========================MERGE SELECTION==============================
        # self.Page2_Widget.radiobutton1.setEnabled(True)
        # self.Page2_Widget.radiobutton2.setEnabled(True)
        # self.Page2_Widget.radiobutton3.setEnabled(True)
        # ========================BACK-NEXT BUTTON=============================
        self.Page2_Widget.back_button.setEnabled(True)
        self.Page2_Widget.next_button.setEnabled(True)    

    def generateMenu(self, pos):
        colNum = float('inf')
        for i in self.Page3_Widget.MainWindow.selectionModel().selection().indexes():
            colNum = i.column()
        _, num_col = self.df.shape
        num_col = num_col - sum(self.FeatureDeleteFlag)
        if colNum < num_col:
            menu = QMenu()
            self.InfoAction = menu.addAction("Feature Operations")
            self.DeleteAction = menu.addAction("Delete")
            self.TransformAction = menu.addAction("Transform")
            self.SelectAction = menu.addAction("Select")
            # self.AnalyzeAction = menu.addAction("Analyze")
            self.InfoAction.setEnabled(False)
            self.InfoAction.setCheckable(False)
            self.DeleteAction.setIcon(QIcon(QPixmap('fig/delete.png')))
            self.TransformAction.setIcon(QIcon(QPixmap('fig/transform.png')))
            self.SelectAction.setIcon(QIcon(QPixmap('fig/selection.png')))
            # self.AnalyzeAction.setIcon(QIcon(QPixmap('analysis.png')))
            self.DeleteAction.triggered.connect(lambda: self.DeletePreprocessQuery(colNum))
            self.TransformAction.triggered.connect(lambda: self.TransformPreprocessQuery(colNum))
            self.SelectAction.triggered.connect(lambda: self.SelectPreprocess(colNum))
            # self.AnalyzeAction.triggered.connect(lambda: self.AnalyzePreprocess(colNum))
            menu.setStyleSheet('*{font-size: 25px;}')
            screenpos = self.Page3_Widget.MainWindow.mapToGlobal(pos)
            _ = menu.exec_(screenpos)

    def DeletePreprocessQuery(self, colNum):
        DeleteQueryButtonYes = QMessageBox.Yes
        DeleteQueryButtonNo = QMessageBox.No
        result = QMessageBox.question(self, "Comfirmation", "Delete this column?", DeleteQueryButtonYes | DeleteQueryButtonNo, DeleteQueryButtonYes)
        if result == QMessageBox.Yes:
            self.DeletePreprocess(colNum)

    def DeletePreprocess(self, colNum):
        num_remain = 0
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                num_remain += 1
            if num_remain == colNum + 1:
                break
        self.FeatureDeleteFlag[i] = True
        self.ActionStack.append(('Delete', i)) # delete col-i
        # change number of missing values
        DeleteMissingNum = self.df.iloc[:,i].isnull().sum()
        self.MissingNum -= DeleteMissingNum
        if len(self.ActionStack) == 1:
            self.Page3_Widget.UndoButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 120px;
            }
            QPushButton:hover{
                font: bold 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
            self.Page3_Widget.UndoButton.setEnabled(True)
        self.ChangePageDisplay()
        # Whenever there are operations in page3 and page4/5 already initialized, 
        # we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init =False

    def TransformPreprocessQuery(self, colNum):
        result = QMessageBox.question(self, "Comfirmation", "Transform this column to categorical data?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.TransformPreprocess(colNum)        

    def TransformPreprocess(self, colNum):
        num_remain = 0
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                num_remain += 1
            if num_remain == colNum + 1:
                break        
        if self.CategoricalFlag[i]:
            return
        self.CategoricalFlag[i] = True
        self.ActionStack.append(('Transform', i))
        if len(self.ActionStack) == 1:
            self.Page3_Widget.UndoButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 120px;
            }
            QPushButton:hover{
                font: bold 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
            self.Page3_Widget.UndoButton.setEnabled(True)
        if self.CategoricalTransformDict[i] == None:
            self.TransformPreprocessThread = TransformThread(self.df.iloc[:,i], i)
            self.TransformPreprocessThread.TransformFinish.connect(self.TransformDisplay)
            self.TransformPreprocessThread.TenPercent.connect(self.TransformProgressDisplay)
            self.DisablePage3Buttons()
            self.TransformPreprocessThread.start()
        else:
            self.ChangePageDisplay()
            # Whenever there are operations in page3 and page4/5 already initialized, 
            # we have to re-initializa them
            if self.page4_init:
                self.page4_init = False
                self.Step_Widget.step_layout_button5.hide()
            if self.page5_init:
                self.page5_init =False


    def SelectPreprocess(self, colNum):
        self.selectionbox = SelectionDialog()
        self.selectionbox.SelectButton1.clicked.connect(self.AssertRangeSelection)
        self.selectionbox.SelectButton2.clicked.connect(self.AssertConditionSelection)
        self.selectionbox.RangeLE1.editingFinished.connect(self.ChangeRangeDown)
        self.selectionbox.RangeLE2.editingFinished.connect(self.ChangeRangeUp)
        self.selectionbox.SelectionMethodComboBox.currentIndexChanged.connect(self.ChangeConditionMethod)
        self.selectionbox.ValueLE.editingFinished.connect(self.ChangeValueLE)
        self.selectionbox.CancelButton.clicked.connect(self.SelectionCancel)
        self.selectionbox.OKButton.clicked.connect(lambda: self.AssertSelectionPreprocess(colNum))
        self.selectionbox.show()
        self.selectionbox.exec_()
        

        


    # def AnalyzePreprocess(self, colNum):
    #     pass

    def TransformDisplay(self, TransformDict, col_idx):
        self.EnablePage3Buttons()
        self.CategoricalTransformDict[col_idx] = TransformDict
        self.ChangePageDisplay()
        # Whenever there are operations in page3 and page4/5 already initialized, 
        # we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init =False

    def TransformProgressDisplay(self, counter):
        if counter == 1:
            self.Page3_Widget.PreprocessProgressLE.setText("Transform Process: ")
            self.Page3_Widget.PreprocessProgressLE.show()
            self.Page3_Widget.progressbar.setMaximum(10)
            self.Page3_Widget.progressbar.show()
            self.Page3_Widget.progressbar.setValue(counter)
        else:
            self.Page3_Widget.progressbar.setValue(counter)
            if counter == 10:
                self.Page3_Widget.progressbar.hide()
                self.Page3_Widget.PreprocessProgressLE.hide()

    def DownloadPreprocessFile(self):
        savefile, _ = QFileDialog.getSaveFileName(self, 'Save File', '.', '(*.csv)')
        if savefile == '':
            return
        self.DownloadThread = DownloadThread(self.df, self.FeatureDeleteFlag, self.CategoricalFlag, self.CategoricalTransformDict, self.SelectionIndex, savefile)
        self.DownloadThread.ThirdProgress.connect(self.DownloadProgressDisplay)
        self.DownloadThread.DownloadFinish.connect(self.DownloadFinish)
        self.DisablePage3Buttons()
        self.DownloadThread.start()

        # savefile, _ = QFileDialog.getSaveFileName(self, 'Save Files', '.', '(*.csv)')
        # CurrentDF.to_csv(savefile)
        # QMessageBox.information(self, "Download Successfully", "The file has been downloaded successfully!", QMessageBox.Yes, QMessageBox.Yes)

    def AssertRangeSelection(self):
        # Enable the range-selection LEs
        self.selectionbox.RangeLE1.setEnabled(True)
        self.selectionbox.RangeLE2.setEnabled(True)
        # Disable the condition-selection LEs
        self.selectionbox.SelectionMethodComboBox.setEnabled(False)
        self.selectionbox.ValueLE.setEnabled(False)
        # Change the value of some variables
        self.SelectionMethod = "Range"
    
    def AssertConditionSelection(self):
        # Enable the condition-selection LEs
        self.selectionbox.SelectionMethodComboBox.setEnabled(True)
        self.selectionbox.ValueLE.setEnabled(True)
        # Disable the range-selection LEs
        self.selectionbox.RangeLE1.setEnabled(False)
        self.selectionbox.RangeLE2.setEnabled(False)
        # Change the value of some variable
        self.SelectionMethod = "Condition"

    def ChangeRangeDown(self):
        try:
            self.SelectionRangeDown = float(self.selectionbox.RangeLE1.text())
        except:
            pass
    
    def ChangeRangeUp(self):
        try:
            self.SelectionRangeUp = float(self.selectionbox.RangeLE2.text())
        except:
            pass

    def ChangeConditionMethod(self, i):
        self.SelectionCondition = i

    def ChangeValueLE(self):
        self.SelectionValue = self.selectionbox.ValueLE.text()
        if self.SelectionCondition in [2,3,4,5]:
            try:
                self.SelectionValue = float(self.SelectionValue)
            except:
                self.selectionbox.ValueLE.setText('0')
                self.SelectionValue = 0
                QMessageBox.critical(self.selectionbox, "Error", "Please enter a numerical value!", QMessageBox.Yes, QMessageBox.Yes)
                
    def SelectionCancel(self):
        # None-lize all the variables
        self.SelectionMethod = None
        self.SelectionRangeUp = None
        self.SelectionRangeDown = None
        self.SelectionCondition = 0
        self.SelectionValue = None
        # Close the QDialog
        self.selectionbox.close()

    def AssertSelectionPreprocess(self, colNum):
        # Check for "Empty" conditions
        if self.SelectionMethod == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please choose one selection method!", QMessageBox.Yes, QMessageBox.Yes)
            return
        elif self.SelectionMethod == 'Range' and self.SelectionRangeDown == None and self.SelectionRangeUp == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please specify at least one bound for the range!", QMessageBox.Yes, QMessageBox.Yes)
            return
        elif self.SelectionMethod == 'Condition' and self.SelectionValue == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please specify the value for the chosen condition!", QMessageBox.Yes, QMessageBox.Yes)
            return
        # Start the Selection
        num_remain = 0
        for i in range(len(self.FeatureDeleteFlag)):
            if self.FeatureDeleteFlag[i] == False:
                num_remain += 1
            if num_remain == colNum + 1:
                break  
        if self.CategoricalFlag[i] == False:
            feature = self.df.iloc[:,i]
        else:
            feature = self.df.iloc[:,i].map(lambda x:self.TransformToCategorical(x, i))
            # print(self.df.iloc[:,i][:20])
            # print(feature[:20])
        self.SelectionPreprocessThread = SelectionThread(feature, self.SelectionMethod, self.SelectionRangeDown, self.SelectionRangeUp, self.SelectionCondition, self.SelectionValue, self.SelectionIndex, i)
        self.SelectionPreprocessThread.SelectionFinish.connect(self.SelectionDisplay)
        self.SelectionPreprocessThread.TenPercent.connect(self.SelectionProgressDisplay)
        self.DisablePage3Buttons()
        self.SelectionPreprocessThread.start()

    def TransformToCategorical(self, x, i):
        if str(x) == 'nan':
            return np.nan
        else:
            return self.CategoricalTransformDict[i][x]

    def SelectionDisplay(self, SelectedIndicator, SelectionIndex, col_idx):
        self.EnablePage3Buttons()
        self.SelectionIndex = SelectionIndex
        if self.SelectionFlag[col_idx] == False:
            self.SelectionFlag[col_idx] = True
            self.SelectionIndicator[col_idx] = [SelectedIndicator]
            self.ActionStack.append(('Select', col_idx))
            if len(self.ActionStack) == 1:
                self.Page3_Widget.UndoButton.setStyleSheet('''
                QPushButton{
                    font: 28px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:29px 29px;
                    margin-left: 35px;
                    width: 120px;
                }
                QPushButton:hover{
                    font: bold 28px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:29px 29px;
                    margin-left: 35px;
                }
            ''')
                self.Page3_Widget.UndoButton.setEnabled(True)
        else:
            self.SelectionIndicator[col_idx].append(SelectedIndicator)
            self.ActionStack.append(('Select', col_idx))
            if len(self.ActionStack) == 1:
                self.Page3_Widget.UndoButton.setStyleSheet('''
                QPushButton{
                    font: 28px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:29px 29px;
                    margin-left: 35px;
                    width: 120px;
                }
                QPushButton:hover{
                    font: bold 28px "Microsoft YaHei UI";
                    border: none;
                    background-color: none;
                    color: #3B8BB9;
                    qproperty-icon:url(fig/undo.png);
                    qproperty-iconSize:29px 29px;
                    margin-left: 35px;
                }
            ''')
                self.Page3_Widget.UndoButton.setEnabled(True)
        self.ChangePageDisplay()
        self.selectionbox.close()
        self.SelectionMethod = None
        self.SelectionRangeUp = None
        self.SelectionRangeDown = None
        self.SelectionCondition = 0
        self.SelectionValue = None
        # Whenever there are operations in page3 and page4/5 already initialized, 
        # we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False

    def SelectionProgressDisplay(self, counter):
        if counter == 1:
            self.Page3_Widget.PreprocessProgressLE.setText("Selection Process: ")
            self.Page3_Widget.PreprocessProgressLE.show()
            self.Page3_Widget.progressbar.setMaximum(10)
            self.Page3_Widget.progressbar.show()
            self.Page3_Widget.progressbar.setValue(counter)
        else:
            self.Page3_Widget.progressbar.setValue(counter)
            if counter == 10:
                self.Page3_Widget.progressbar.hide()
                self.Page3_Widget.PreprocessProgressLE.hide()        

    def DownloadProgressDisplay(self, counter):
        if counter == 1:
            self.Page3_Widget.PreprocessProgressLE.setText("Download Process: ")
            self.Page3_Widget.PreprocessProgressLE.show()
            self.Page3_Widget.progressbar.setMaximum(3)
            self.Page3_Widget.progressbar.show()
            self.Page3_Widget.progressbar.setValue(counter)
        else:
            self.Page3_Widget.progressbar.setValue(counter)
            if counter == 3:
                self.Page3_Widget.progressbar.hide()
                self.Page3_Widget.PreprocessProgressLE.hide()

    def DownloadFinish(self, df, savefile):
        self.EnablePage3Buttons()
        try:
            df.to_csv(savefile, index=False)
            QMessageBox.information(self, "Download Successfully", "The file has been downloaded successfully!", QMessageBox.Yes, QMessageBox.Yes)       
        except PermissionError:
            QMessageBox.critical(self, "Permission Error", "Permission Error: The saved file is currently opened!", QMessageBox.Yes, QMessageBox.Yes)

    def UndoPreprocess(self):
        LastAction, col_idx = self.ActionStack[-1]
        self.ActionStack.pop(-1)
        if len(self.ActionStack) == 0:
            self.Page3_Widget.UndoButton.setStyleSheet('''
            QPushButton{
                font: 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 120px;
            }
            QPushButton:hover{
                font: bold 28px "Microsoft YaHei UI";
                border: none;
                background-color: none;
                color: #3B8BB9;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
            self.Page3_Widget.UndoButton.setEnabled(False)
        if LastAction == 'Delete':
            self.FeatureDeleteFlag[col_idx] = False
            AddMissingNum = self.df.iloc[:,col_idx].isnull().sum()
            self.MissingNum += AddMissingNum
        elif LastAction == 'Transform':
            self.CategoricalFlag[col_idx] = False
        elif LastAction ==  'Select':
            self.SelectionIndicator[col_idx].pop(-1)
            if len(self.SelectionIndicator[col_idx]) == 0:
                self.SelectionFlag[col_idx] = False
                self.SelectionIndicator[col_idx] = None
            num_sample, _ = self.df.shape
            OverallSelectionIndicator = np.array([True]*num_sample)
            for i in range(len(self.SelectionIndicator)):
                if self.SelectionIndicator[i] == None:
                    continue
                else:
                    for indicator in self.SelectionIndicator[i]:
                        try:
                            OverallSelectionIndicator *= indicator
                        except:
                            print(OverallSelectionIndicator[0:20])
                            print(indicator[0:20])
                            OverallSelectionIndicator *= indicator
            self.SelectionIndex = np.argwhere(OverallSelectionIndicator).reshape(-1,)
        self.ChangePageDisplay()
        # Whenever there are operations in page3 and page4/5 already initialized, 
        # we have to re-initializa them
        if self.page4_init:
            self.page4_init = False
            self.Step_Widget.step_layout_button5.hide()
        if self.page5_init:
            self.page5_init = False
        
    def DisablePage3Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        self.Step_Widget.step_layout_button2.setEnabled(False)
        self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        self.Step_Widget.step_layout_button5.setEnabled(False)        
        # ======================BUTTONS IN THE TOOLBAR=========================
        self.Page3_Widget.InputLineNumber.setEnabled(False)
        self.Page3_Widget.UndoButton.setEnabled(False)
        self.Page3_Widget.DownloadButton.setEnabled(False)
        # ========================BACK-NEXT BUTTONS============================
        self.Page3_Widget.back_button.setEnabled(False)
        self.Page3_Widget.next_button.setEnabled(False)

    def EnablePage3Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        self.Step_Widget.step_layout_button2.setEnabled(True)
        self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        self.Step_Widget.step_layout_button5.setEnabled(True)        
        # ======================BUTTONS IN THE TOOLBAR=========================
        self.Page3_Widget.InputLineNumber.setEnabled(True)
        if len(self.ActionStack) == 0:
            self.Page3_Widget.UndoButton.setEnabled(False)
        else:
            self.Page3_Widget.UndoButton.setEnabled(True)
        self.Page3_Widget.DownloadButton.setEnabled(True)
        # ========================BACK-NEXT BUTTONS============================
        self.Page3_Widget.back_button.setEnabled(True)
        self.Page3_Widget.next_button.setEnabled(True)        

    # See funtion upload_file
    def Page4UploadAlg(self):
        if self.AlgorithmFile != None:
            QMessageBox.critical(self, "Multiple File Error", "You can only upload one files", QMessageBox.Yes, QMessageBox.Yes)
            return
        newfile, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '(*.py)')
        if newfile == '':
            return
        self.AlgorithmFile = newfile
        current_num_row = self.Page4_Widget.AlgFileTable.rowCount()
        self.Page4_Widget.AlgFileTable.setRowCount(current_num_row+1)
        self.Page4_Widget.AlgFileTable.setItem(current_num_row, 0, QTableWidgetItem(newfile))
        self.Page4_Widget.AlgFileTable.setCellWidget(current_num_row, 1, FileDeleteButton(0))
        self.Page4_Widget.AlgFileTable.resizeColumnsToContents()
        self.Page4_Widget.AlgFileTable.horizontalHeader().setStretchLastSection(True)
        self.Page4_Widget.AlgFileTable.cellWidget(current_num_row, 1).delete_button.clicked.connect(self.Page4DeleteAlgFile)

    def Page4UploadPar(self):
        if self.ParameterFile != None:
            QMessageBox.critical(self, "Multiple File Error", "You can only upload one files", QMessageBox.Yes, QMessageBox.Yes)
            return
        newfile, _ = QFileDialog.getOpenFileName(self, 'Open File', '.', '(*.py)')
        if newfile == '':
            return
        self.ParameterFile = newfile
        current_num_row = self.Page4_Widget.ParFileTable.rowCount()
        self.Page4_Widget.ParFileTable.setRowCount(current_num_row+1)
        self.Page4_Widget.ParFileTable.setItem(current_num_row, 0, QTableWidgetItem(newfile))
        self.Page4_Widget.ParFileTable.setCellWidget(current_num_row, 1, FileDeleteButton(0))
        self.Page4_Widget.ParFileTable.resizeColumnsToContents()
        self.Page4_Widget.ParFileTable.horizontalHeader().setStretchLastSection(True)    
        self.Page4_Widget.ParFileTable.cellWidget(current_num_row, 1).delete_button.clicked.connect(self.Page4DeleteParFile)

    def Page4DeleteAlgFile(self):
        self.Page4_Widget.AlgFileTable.removeRow(0)
        self.AlgorithmFile = None
        if self.page5_init:
            self.page5_init = False

    def Page4DeleteParFile(self):
        self.Page4_Widget.ParFileTable.removeRow(0)
        self.ParameterFile = None
        if self.page5_init:
            self.page5_init = False

    def DisablePage4Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(False)
        self.Step_Widget.step_layout_button2.setEnabled(False)
        self.Step_Widget.step_layout_button3.setEnabled(False)
        self.Step_Widget.step_layout_button4.setEnabled(False)
        self.Step_Widget.step_layout_button5.setEnabled(False)
        # ==========================SELECT BUTTONS=============================
        self.Page4_Widget.AlgUploadButton.setEnabled(False)
        self.Page4_Widget.ParUploadButton.setEnabled(False)
        # ==========================DELETE BUTTONS=============================
        self.Page4_Widget.AlgFileTable.cellWidget(0, 1).delete_button.setEnabled(False)
        self.Page4_Widget.ParFileTable.cellWidget(0, 1).delete_button.setEnabled(False)
        # ========================BACK-NEXT BUTTONS============================
        self.Page4_Widget.back_button.setEnabled(False)
        self.Page4_Widget.next_button.setEnabled(False)

    def EnablePage4Buttons(self):
        # ======================BUTTONS IN THE STEP BAR========================
        self.Step_Widget.step_layout_button1.setEnabled(True)
        self.Step_Widget.step_layout_button2.setEnabled(True)
        self.Step_Widget.step_layout_button3.setEnabled(True)
        self.Step_Widget.step_layout_button4.setEnabled(True)
        self.Step_Widget.step_layout_button5.setEnabled(True)
        # ==========================SELECT BUTTONS=============================
        self.Page4_Widget.AlgUploadButton.setEnabled(True)
        self.Page4_Widget.ParUploadButton.setEnabled(True)
        # ==========================DELETE BUTTONS=============================
        self.Page4_Widget.AlgFileTable.cellWidget(0, 1).delete_button.setEnabled(True)
        self.Page4_Widget.ParFileTable.cellWidget(0, 1).delete_button.setEnabled(True)
        # ========================BACK-NEXT BUTTONS============================
        self.Page4_Widget.back_button.setEnabled(True)
        self.Page4_Widget.next_button.setEnabled(True)

    def DisplayPage4ProgressBar(self, CurrentNum, TotalNum):
        if CurrentNum == 1:
            self.Page4_Widget.ImputeProgressLE.show()
            self.Page4_Widget.progressbar.setMaximum(TotalNum)
            self.Page4_Widget.progressbar.setValue(CurrentNum)
            self.Page4_Widget.progressbar.show()
        else:
            self.Page4_Widget.progressbar.setValue(CurrentNum)
            if CurrentNum == TotalNum:
                self.Page4_Widget.progressbar.hide()
                self.Page4_Widget.ImputeProgressLE.hide()        

    def Page5_DisplayTable(self, df, dfisnull):
        self.EnablePage4Buttons()
        self.impute_df = df
        self.dfisnull = dfisnull
        self.RewriteNewValue = None
        self.Page5_Widget = Page5_Widget(self.impute_df, self.dfisnull)
        self.Overall_Layout.addWidget(self.Page5_Widget)
        self.Page5_Widget.back_button.clicked.connect(self.GOTO_Page4)
        self.Page5_Widget.DownloadButton.clicked.connect(self.Page5DownloadFile)
        self.Page5_Widget.RedoButton.clicked.connect(self.ReImpute)
        self.Page5_Widget.InputLineNumber.editingFinished.connect(self.Page5ChangeDisplay)
        self.Page5_Widget.MainWindow.customContextMenuRequested.connect(self.generateMenuPage5)
        # It seems that people can only jump to page5 from page4
        self.Page4_Widget.hide()
        self.Step_Widget.step_layout_button4.setStyleSheet(self.button_release_style)
        self.Step_Widget.step_layout_button5.setStyleSheet(self.button_select_style)
        self.Page5_Widget.show()
        
    def Page5DownloadFile(self):
        savefile, _ = QFileDialog.getSaveFileName(self, 'Save File', '.', '(*.csv)')
        if savefile == '':
            return
        try:
            self.impute_df.to_csv(savefile, index=False)
            QMessageBox.information(self, "Download Successfully", "The file has been downloaded successfully!", QMessageBox.Yes, QMessageBox.Yes) 
        except PermissionError:
            QMessageBox.critical(self, "Permission Error", "Permission Error: The saved file is currently opened!", QMessageBox.Yes, QMessageBox.Yes)

    def Page5ChangeDisplay(self):
        num_row, num_col = self.impute_df.shape
        try:
            PageNumber = int(self.Page5_Widget.InputLineNumber.text())
        except:
            PageNumber = 1
        TotalPage = int(num_row/100)+1
        if PageNumber > TotalPage:
            PageNumber = TotalPage
            self.Page5_Widget.InputLineNumber.setText(str(TotalPage))
            QMessageBox.warning(self, "Warning", "Page number out of, jump to the last page!", QMessageBox.Yes, QMessageBox.Yes)
        elif PageNumber <= 0:
            PageNumber = 1
            self.Page5_Widget.InputLineNumber.setText(str(1))
            QMessageBox.warning(self, "Warning", "Page number out of, jump to the first page!", QMessageBox.Yes, QMessageBox.Yes)
        self.Page5_Widget.MainWindow.clear()
        start_row = 100 * (PageNumber - 1)
        end_row = min(num_row, 100 * PageNumber)
        for i in range(start_row, end_row):
            for j in range(num_col):
                CurItem = QTableWidgetItem(str(self.impute_df.iat[i, j]))
                if self.dfisnull.iat[i, j]:
                    if str(self.impute_df.iat[i, j]) == 'nan':
                        CurItem = QTableWidgetItem()
                    CurItem.setBackground(QBrush(QColor(201,252,255)))
                self.Page5_Widget.MainWindow.setItem(i-start_row, j, CurItem)
        vertical_header = [str(i+1) for i in range(start_row, start_row+self.Page5_Widget.MainWindow.rowCount())]
        self.Page5_Widget.MainWindow.setVerticalHeaderLabels(vertical_header)
        self.Page5_Widget.MainWindow.verticalHeader().show()
        self.Page5_Widget.MainWindow.setHorizontalHeaderLabels(list(self.impute_df.columns))
        self.Page5_Widget.MainWindow.resizeColumnsToContents()

    def generateMenuPage5(self, pos):
        colNum = float('inf')
        for i in self.Page5_Widget.MainWindow.selectionModel().selection().indexes():
            colNum = i.column()
            rowNum = i.row()
        _, num_col = self.impute_df.shape
        if colNum < num_col:
            menu = QMenu()
            self.RejectAction = menu.addAction("Reject Value")
            self.RewriteAction = menu.addAction("Rewrite Value")
            self.RejectAction.setIcon(QIcon(QPixmap('fig/reject.png')))
            self.RewriteAction.setIcon(QIcon(QPixmap('fig/rewrite.png')))
            self.RejectAction.triggered.connect(lambda: self.RejectValueProcess(rowNum, colNum))
            self.RewriteAction.triggered.connect(lambda: self.RewriteValueProcess(rowNum, colNum))
            menu.setStyleSheet('*{font-size:25px}')
            screenpos = self.Page5_Widget.MainWindow.mapToGlobal(pos)
            _ = menu.exec_(screenpos)

    def RejectValueProcess(self, rowNum, colNum):
        try:
            pagenum = int(self.Page5_Widget.InputLineNumber.text())
        except:
            pagenum = 1
        rowNum_all = (pagenum-1)*100 + rowNum
        currentvalue = str(self.impute_df.iat[rowNum_all, colNum])
        question = "Reject this value: "+currentvalue+"?"
        result = QMessageBox.question(self, "Comfirmation", question, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.impute_df.iat[rowNum_all, colNum] = np.nan
            self.dfisnull.iat[rowNum_all, colNum] = True
            EmptyTableItem = QTableWidgetItem()
            EmptyTableItem.setBackground(QBrush(QColor(201,252,255)))
            self.Page5_Widget.MainWindow.setItem(rowNum, colNum, EmptyTableItem)
            self.Page5_Widget.RedoButton.setEnabled(True)
            self.Page5_Widget.RedoButton.setStyleSheet('''
            QPushButton{
                font-size: 26px;
                font: bold;
                border: none;
                background-color: none;
                color: #1371ec;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 220px;
            }
            QPushButton:hover{
                font-size: 29px;
                font: bold;
                border: none;
                background-color: none;
                color: #1371ec;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
    
    def RewriteValueProcess(self, rowNum, colNum):
        try:
            pagenum = int(self.Page5_Widget.InputLineNumber.text())
        except:
            pagenum = 1
        rowNum_all = (pagenum-1)*100 + rowNum
        currentvalue = str(self.impute_df.iat[rowNum_all, colNum])
        self.page5RewriteDialog = RewriteDialog(currentvalue)
        self.page5RewriteDialog.Newvalue.editingFinished.connect(self.GetRewriteValue)
        self.page5RewriteDialog.OKButton.clicked.connect(lambda: self.AssertRewriteValue(rowNum, colNum))
        self.page5RewriteDialog.CancelButton.clicked.connect(self.CancelRewriteValue)
        self.page5RewriteDialog.show()
        self.page5RewriteDialog.exec_()

    def GetRewriteValue(self):
        try:
            self.RewriteNewValue = int(self.page5RewriteDialog.Newvalue.text())
        except:
            try:
                self.RewriteNewValue = float(self.page5RewriteDialog.Newvalue.text())
            except:
                self.RewriteNewValue = str(self.page5RewriteDialog.Newvalue.text())

    def CancelRewriteValue(self):
        self.RewriteNewValue = None
        self.page5RewriteDialog.close()

    def AssertRewriteValue(self, rowNum, colNum):
        if self.RewriteNewValue == None:
            QMessageBox.critical(self.selectionbox, "Error", "Please enter new value!", QMessageBox.Yes, QMessageBox.Yes)
            return
        try:
            pagenum = int(self.Page5_Widget.InputLineNumber.text())
        except:
            pagenum = 1
        rowNum_all = (pagenum-1)*100 + rowNum
        self.impute_df.iat[rowNum_all, colNum] = self.RewriteNewValue
        self.dfisnull.iat[rowNum_all, colNum] = False
        self.Page5_Widget.MainWindow.setItem(rowNum, colNum, QTableWidgetItem(str(self.RewriteNewValue)))
        self.Page5_Widget.RedoButton.setEnabled(True)
        self.Page5_Widget.RedoButton.setStyleSheet('''
            QPushButton{
                font-size: 26px;
                font: bold;
                border: none;
                background-color: none;
                color: #1371ec;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
                width: 220px;
            }
            QPushButton:hover{
                font-size: 29px;
                font: bold;
                border: none;
                background-color: none;
                color: #1371ec;
                qproperty-icon:url(fig/undo.png);
                qproperty-iconSize:29px 29px;
                margin-left: 35px;
            }
        ''')
        self.RewriteNewValue = None
        self.page5RewriteDialog.close()

    def ReImpute(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('fig/zju.png'))
    main = Imputation_System()
    main.showMaximized()
    sys.exit(app.exec_())
    