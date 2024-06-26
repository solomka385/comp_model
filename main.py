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
        df['rb_corners'] = labelencoder.fit_transform(df[	'rb_corners'])
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
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
        accuracy = cr_df.loc['accuracy', 'precision']
        # Отображение точности
        sns.barplot(x=cr_df.index, y=cr_df['precision'], color='blue', alpha=0.5, ax=axes[0])
        axes[0].set_title('Precision')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Score')

        # Отображение полноты
        sns.barplot(x=cr_df.index, y=cr_df['recall'], color='green', alpha=0.5, ax=axes[1])
        axes[1].set_title('Recall')
        axes[1].set_xlabel('Class')
        axes[1].set_ylabel('Score')

        # Отображение F1-меры и добавление линии для точности
        sns.barplot(x=cr_df.index, y=cr_df['f1-score'], color='red', alpha=0.5, ax=axes[2])
        axes[2].set_title('F1-score')
        axes[2].set_xlabel('Class')
        axes[2].set_ylabel('Score')

        # Регулирование расстояния между графиками
        plt.tight_layout()
        # Сохранение графика
        plot_path = os.path.join('plots', 'plot_classification_report.png')
        plt.savefig(plot_path)
        plt.close()
        # Отображение графика
        
        
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
