import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
class PlotDrawer:
    def draw_plots(self,json_file):
        
        df = pd.read_json(json_file)
        labelencoder = LabelEncoder()
        df['gt_corners'] = labelencoder.fit_transform(df['gt_corners'] )
        df['rb_corners'] = labelencoder.fit_transform(df['rb_corners'])
        y_test = df['gt_corners']
        y_pred = df['rb_corners']
        
        fig = plt.figure(figsize=(17, 10))
        plt.title(f"Prediction with")
        plt.scatter(range(df['name'].shape[0]), df['gt_corners'], color='red', label='Real')
        plt.scatter(range(df['name'].shape[0]), df['rb_corners'], marker='.', label='Predict')
        plt.legend(loc=2, prop={'size': 25})
        
        plot_path = os.path.join('plots', 'corners_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        cm = confusion_matrix(y_test, y_pred)
    
        # Вычисление отчета о классификации
        cr = classification_report(y_test, y_pred, output_dict=True)
        cr_df = pd.DataFrame(cr).transpose()
        
        # Построение графика для точности, полноты и F1-меры
        fig = plt.figure(figsize=(12, 6))
        sns.barplot(x=cr_df.index, y=cr_df['precision'], color='blue', alpha=0.5, label='Precision')
        sns.barplot(x=cr_df.index, y=cr_df['recall'], color='green', alpha=0.5, label='Recall')
        sns.barplot(x=cr_df.index, y=cr_df['f1-score'], color='red', alpha=0.5, label='F1-score')
        plt.title('Classification Metrics')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.legend()
        plot_path = os.path.join('plots', 'plot_Classification_Metrics.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Создание тепловой карты для матрицы ошибок
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plot_path = os.path.join('plots', 'plot_Confusion_Matrix.png')
        plt.savefig(plot_path)
        plt.close()

        
        return plot_path
        