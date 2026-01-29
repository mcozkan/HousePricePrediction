###############################################################################################################################################################
# LIBRARIES
###############################################################################################################################################################
import pandas as pd
import numpy as np
import logging
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
import sweetviz as sv

# We add the missing alert class into Numpy lirary in order to solve the version problem...
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = type('VisibleDeprecationWarning', (DeprecationWarning,), {})
    
from typing import Optional, Dict, Any


###############################################################################################################################################################
# DATA QUALITY
###############################################################################################################################################################
class InvalidColumnError(Exception):
    """Column names starts with invalid character such as digits, please corretct!!!"""
    pass



class DataQuality():
    def __init__(self, df_combined: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame):
        
        self.df_combined = df_combined.copy()
        self.df_train = train_df.copy()
        self.df_test = test_df.copy()
        self.num_dict = {1: 'fisrt_', 2: 'second_', 3: 'third_', 4 : 'fourth_'}
        self.numeric_cols = [col for col in df_combined.columns if str(col)[0].isdigit()]

    def name_check(self, df_combined):
        try:
            if len(self.numeric_cols) > 0:
                rname = {col : self.num_dict[int(str(col)[0])] +''+ col for col in self.numeric_cols}
                self.df_combined.rename(columns = rname, inplace = True)
                print("Column naming proccess is done successfully!!!")

            else:
                raise InvalidColumnError("Columns start with invalid numbers")

        except ValueError as e:
            raise ValueError(f"Data validation failed: {e}")
        
        
    
    def validate(self, col_target : str, col_id : str):
        try:
            
            actual_rows = int(self.df_combined[col_target].isnull().sum().item())
            expected_test_rows = (len(self.df_test[col_id]))

            """
            Birleştirilmiş verideki NaN hedef değer sayısı ile 
            orijinal test setindeki satır sayısını karşılaştırır.
            """
    
            if  expected_test_rows == actual_rows:
                
                print(f"Row Counts are Equal - the count... : {actual_rows}")
            else:
                raise ValueError(
                        f"❌ Veri Bütünlüğü Bozuk! \n"
                        f"Hedefteki Boş Sayısı: {actual_rows} | "
                        f"Beklenen (Test): {expected_test_rows}"
                    )
        except KeyError:
            raise KeyError(f"Sütun bulunamadı: {col_target}")
        except Exception as e:
            logging.error(f"Beklenmedik bir hata oluştu: {e}")
            raise

    


###############################################################################################################################################################
# EDA REPORT 
###############################################################################################################################################################

class EdaReports():

    def __init__(self, df: pd.DataFrame, target:str):
        self.df_eda = df.copy()
        self.null_cols = [col for col in self.df_eda.columns if self.df_eda[col].isnull().any()]
        self.cat_cols, _, _, _, _ = self.grab_col_names(cat_th = 10, car_th = 20)
        self.target = target

    def null_check_percentage(self, threshold=0.0, sorted_=True):
        """
        Calculates the percentage of missing values per column.

        Args:
            df_combined (DataFrame): Input data.
            threshold (float): Only returns columns with missing % higher than this.
            sorted_ (bool): If True, sorts the result in descending order.
        """
        null_percentage = {}
        for col in self.null_cols:
            percentage = round(
                ((self.df_eda[col].isnull().sum() / len(self.df_eda)) * 100).item(), 2
            )
        # percentage = self.df_eda.isnull().mean() * 100

            if percentage > threshold:
                null_percentage[col] = percentage
        if sorted_:
            null_percentage = dict(
                sorted(null_percentage.items(), key=lambda item: item[1], reverse=True)
            )

        return null_percentage
    

    def basics(self, threshold=0.0, sorted_=True):
        print(f"Data Row Count..: {self.df_eda.shape[0]}")
        print(f"Data Feature Count..: {self.df_eda.shape[1]}")
        print(f"Data Central Stats..: {self.df_eda.describe().T}")
        print(f"Data Types..: {self.df_eda.dtypes}")
        print(f"Null Values...: {self.df_eda.isnull().sum()}")
        print(f"Null Percentages..: {self.null_check_percentage(threshold, sorted_)}")
        
    
    
    def grab_col_names(self,  cat_th = 10, car_th = 20):
        cat_cols = [col for col in self.df_eda.columns if self.df_eda[col].dtypes == 'O']
        num_but_cat = [col for col in self.df_eda.columns if self.df_eda[col].nunique() < cat_th and self.df_eda[col].dtypes == 'O']
        cat_but_car = [col for col in self.df_eda.columns if self.df_eda[col].nunique() > car_th and self.df_eda[col].dtypes == 'O']
        #update cat_cols
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        
        num_cols = [col for col in self.df_eda.columns if self.df_eda[col].dtypes != 'O']

        print(f"Observations count ...: {self.df_eda.shape[0]}")
        print(f"Variable ...: {self.df_eda.shape[1]}")
        print(f"cat_cols count ...: {len(cat_cols)}")
        print(f"num_cols ...: {len(num_cols)}")
        print(f"cat_but_car ...:{len(cat_but_car)}")
        print(f"num_but_cat ...: {len(num_but_cat)}")

        return cat_cols, num_but_cat, cat_but_car, cat_cols, num_cols

    
    
    def get_plot_base64(self):
        """Grafiği string formatına çevirir"""
        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    def target_summary_with_cat(self):
                
        """
        Kategorik değişkenlerin hedef değişken üzerindeki etkisini analiz eder 
        ve hem ekrana hem de HTML raporuna basar.
        """
        html_content = "<html><head><meta charset='UTF-8'></head><body><h1>Categorical Feature Analysis</h1>"

        for col in self.cat_cols:
            # 1. Ekrana (Console) özet basma
            print(f"####### Analysis for: {col} #######")
            summary_stats = self.df_eda.groupby(col)[self.target].mean().to_frame(name="TARGET_MEAN")
            print(summary_stats, end="\n\n")

            # 2. HTML Tablosu Hazırlama
            # Hem adet (Count), hem oran (Ratio), hem hedef ortalaması (Target Mean)
            counts = self.df_eda[col].value_counts()
            ratios = 100 * counts / len(self.df_eda)
            
            summary_df = pd.DataFrame({
                "Count": counts,
                "Ratio (%)": ratios.round(2),
                "Target Mean": self.df_eda.groupby(col)[self.target].mean()
            })

            html_content += f"<h2>Feature: {col}</h2>"
            html_content += summary_df.to_html()

            # 3. Grafik Oluşturma ve Gömme
            plt.figure(figsize=(8, 5))
            sns.countplot(x=col, data=self.df_eda)
            plt.title(f"Distribution of {col}")
            
            plot_url = self.get_plot_base64()
            html_content += f'<br><img src="data:image/png;base64,{plot_url}"><hr>'
            plt.close() # Belleği temiz tutalım

        html_content += "</body></html>"

        # 4. Raporu Tek Seferde Yazma
        with open("eda_categorical_report.html", "w", encoding="utf-8") as f:
            f.write(html_content)
        print("✅ Rapor başarıyla 'eda_categorical_report.html' adıyla kaydedildi.")

    
    def sweetviz_report(self):
        # You may also get the plots as a HTML Report page by using Sweetviz library...
        my_report = sv.analyze(self.df_eda)
        my_report.show_html() # present us a Smart Report on Browser
        

    def ydataProfiling(self, onNotebook = True, toFile = True):
        # Creating Report (with the usage of the Big Dataset, it results faster and better to setting the explorative  = True)
        profile = ProfileReport(self.df_eda, title="Data Analysis Report", explorative=True)
        if onNotebook:
            ### In order to see in Jupyter Notebook directly:
            profile.to_notebook_iframe()
        if toFile:
            # 3 Or in order to save it as a HTML file ( That is the best)
            profile.to_file("data_analysis_rerport.html")


###############################################################################################################################################################
# Ptoliflied Feature Distribution
###############################################################################################################################################################


class plotlify_feature_distribution():
    def __init__(self, df: pd.DataFrame, target: str):
        self.df = df.copy()
        self.target = target

        # EdaReports'u burada kullanabilmek için örneklememiz gerekir
        self.eda = EdaReports(self.df, self.target)
        self.cat_cols = self.eda.cat_cols

    def get_plot_base64(self):
        """Grafiği string formatına çevirir"""
        img = io.BytesIO()
        plt.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode()

    def target_summary_with_cat(self, df, target_a, cat_cols):
        print(pd.DataFrame({"TARGET_MEAN": df.groupby(self.cat_cols)[self.target].mean()}), end = "\n\n\n")


    def create_html_report(self):
    
        html_content = "<html><body><h1>Categorical Feature Analysis</h1>"

        if self.cat_cols:
            for col in self.cat_cols:
                # 1. Tabloyu HTML'e çevir
                summary_df = pd.DataFrame(
                    {
                        col: df[col].value_counts(),
                        "Ratio": 100 * df[col].value_counts() / len(df),
                    }
                )
        if target_a:
            for col in self.cat_cols:
                target_summary_with_cat(df,target_a,col)
        
                html_content += f"<h2>{col}</h2>"
                html_content += summary_df.to_html()

                # 2. Grafiği oluştur ve HTML'e ekle
                plt.figure(figsize=(6, 4))
                sns.countplot(x=df[col], data=df)
                plot_url = self.get_plot_base64()
                html_content += f'<br><img src="data:image/png;base64,{plot_url}"><hr>'
            plt.close()

        html_content += "</body></html>"

        with open("my_eda_report.html", "w") as f:
            f.write(html_content)




"""if __name__=="__main__":
    #Test Sample:
    test_df_ = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': ['cat', 'dog', 'cat', None],
        'C': [None, None, None, None]
    })

    print("--- Running Utils.py Standalone Test ---")

    # test for null_check_percentage function
    print("Checking Null Percentages:")
    print(null_check_percentage(test_df_))"""