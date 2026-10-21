"""Praktikum klasifikasi hujan esok hari dengan XGBoost, pemisahan kronologis.
Jalankan: python MochammadMarilLivio_Test1_KlasifikasiXGBoost.py
CSV mentah tidak dimodifikasi. Imputasi, normalisasi dan encoding di-fit hanya pada data train.
"""
from pathlib import Path
import json
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / 'data' / 'kolkata_weather_jan_june_2026.csv'
OUT = ROOT / 'outputs'
FIG = ROOT / 'figures'
NUMERIC = [
    'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min',
    'rain_sum', 'precipitation_hours', 'wind_speed_10m_max',
    'wind_gusts_10m_max', 'wind_direction_10m_dominant',
    'sunshine_hours', 'month_sin', 'month_cos'
]
CATEGORICAL = ['weather_code']
FEATURES = NUMERIC + CATEGORICAL
TARGET = 'rain_tomorrow'
CODE_NAMES = {
    '0': 'Cerah', '1':'Umumnya cerah', '2':'Berawan sebagian', '3':'Mendung',
    '51':'Gerimis ringan', '53':'Gerimis sedang', '55':'Gerimis lebat',
    '61':'Hujan ringan', '63':'Hujan sedang', '65':'Hujan lebat'
}
NAMES = {
    'temperature_2m_mean':'Suhu rata-rata (°C)', 'temperature_2m_max':'Suhu maksimum (°C)',
    'temperature_2m_min':'Suhu minimum (°C)', 'rain_sum':'Curah hujan hari ini (mm)',
    'precipitation_hours':'Jam presipitasi hari ini', 'wind_speed_10m_max':'Angin maksimum (km/jam)',
    'wind_gusts_10m_max':'Hembusan angin (km/jam)',
    'wind_direction_10m_dominant':'Arah angin dominan (°)',
    'sunshine_hours':'Lama sinar matahari (jam)',
    'month_sin':'Musim: sinus bulan', 'month_cos':'Musim: cosinus bulan',
    'weather_code':'Kode cuaca hari ini',
}


def raw_data():
    df = pd.read_csv(DATA_PATH)
    df['time'] = pd.to_datetime(df['time'], errors='raise')
    return df.sort_values('time').reset_index(drop=True)


def prepare(raw):
    """Target hari t+1 dibuat lewat shift(-1); baris terakhir dihapus.
    Fitur semuanya berasal dari hari t, bukan t+1.
    """
    df = raw.copy().sort_values('time').reset_index(drop=True)
    df['sunshine_hours'] = df['sunshine_duration'] / 3600.0
    df['month_sin'] = np.sin(2*np.pi*df['time'].dt.month / 12)
    df['month_cos'] = np.cos(2*np.pi*df['time'].dt.month / 12)
    df['weather_code'] = df['weather_code'].astype(str)
    df[TARGET] = df['rain_sum'].shift(-1).gt(0).astype('Int64')
    df = df.iloc[:-1].copy().reset_index(drop=True)
    return df


def split_time(df):
    cutoff = int(len(df)*0.8)
    train, test = df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()
    return train, test


def make_pipeline():
    pre = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', MinMaxScaler(clip=True)),
        ]), NUMERIC),
        ('cat', Pipeline([
            ('fill', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), CATEGORICAL),
    ], remainder='drop', verbose_feature_names_out=False)
    model = XGBClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42,
        objective='binary:logistic', eval_metric='logloss', n_jobs=2,
        tree_method='hist'
    )
    return Pipeline([('prep', pre), ('model', model)])


def metrics_from_predictions(ytrue, pred, prob):
    cm = confusion_matrix(ytrue, pred, labels=[0, 1])
    return {
        'accuracy':float(accuracy_score(ytrue, pred)),
        'precision':float(precision_score(ytrue,pred,zero_division=0)),
        'recall':float(recall_score(ytrue,pred,zero_division=0)),
        'f1':float(f1_score(ytrue,pred,zero_division=0)),
        'roc_auc':float(roc_auc_score(ytrue,prob)),
        'confusion_matrix':cm.tolist(),
        'tn':int(cm[0,0]), 'fp':int(cm[0,1]), 'fn':int(cm[1,0]), 'tp':int(cm[1,1]),
    }


def feature_importance_grouped(pipe):
    feature_names = pipe['prep'].get_feature_names_out()
    fi = pd.DataFrame({'encoded_feature':feature_names, 'importance':pipe['model'].feature_importances_})
    fi['feature'] = fi.encoded_feature.map(lambda s: 'weather_code' if s.startswith('weather_code_') else s)
    imp = fi.groupby('feature', as_index=False)['importance'].sum().sort_values('importance',ascending=False)
    imp['feature_label'] = imp['feature'].map(NAMES)
    return imp, fi


def save_figure(name):
    plt.tight_layout()
    plt.savefig(FIG/name, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()


def make_charts(raw, df, train, test, model, ytrue, pred, prob, result):
    sns.set_theme(style='whitegrid',font_scale=0.9)
    palette=['#246B83','#E9A23B']
    # 1 proporsi label untuk semua baris bersampel target
    counts=df[TARGET].value_counts().reindex([0,1],fill_value=0)
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    bars=ax.bar(['Tidak hujan','Hujan'],counts.values,color=palette,width=0.58)
    ax.set(title='Distribusi kelas: hujan keesokan hari',ylabel='Jumlah observasi')
    for b,val in zip(bars,counts.values):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+15,f'{val} ({val/len(df):.1%})',ha='center',fontweight='bold')
    ax.set_ylim(0,max(counts.values)*1.18)
    save_figure('01_distribusi_target.png')
    # 2 histogram utama
    fields=['temperature_2m_mean','temperature_2m_max','temperature_2m_min',
            'rain_sum','wind_speed_10m_max','sunshine_hours']
    fig,axs=plt.subplots(2,3,figsize=(12.0,6.7))
    for ax,col in zip(axs.ravel(),fields):
        sns.histplot(df[col],bins=28,color='#246B83',ax=ax)
        ax.set(xlabel=NAMES[col],ylabel='Frekuensi',title=NAMES[col]); ax.tick_params(labelsize=8)
    fig.suptitle('Distribusi enam fitur cuaca utama',fontsize=14,fontweight='bold',y=1.02)
    save_figure('02_histogram_fitur.png')
    # 3 heatmap numeric (drop derived seasonal cyclic for interpretability)
    corr_cols=['temperature_2m_mean','temperature_2m_max','temperature_2m_min',
               'rain_sum','precipitation_hours','wind_speed_10m_max','wind_gusts_10m_max','sunshine_hours']
    corr=df[corr_cols].corr()
    labels=['Suhu mean','Suhu max','Suhu min','Hujan (mm)','Jam hujan','Angin max','Hembusan','Sinar (jam)']
    fig,ax=plt.subplots(figsize=(9,6.6))
    sns.heatmap(corr,ax=ax,annot=True,fmt='.2f',cmap='BrBG',center=0,vmin=-1,vmax=1,
                xticklabels=labels,yticklabels=labels,square=True,annot_kws={'size':8},cbar_kws={'shrink':.75})
    ax.set_title('Korelasi Pearson antarfitur numerik',fontsize=13,fontweight='bold')
    plt.xticks(rotation=36,ha='right'); plt.yticks(rotation=0)
    save_figure('03_heatmap_korelasi.png')
    # 4 period monthly climate signal
    monthly=raw.assign(month=pd.to_datetime(raw['time']).dt.month).groupby('month').agg(
        rain_avg=('rain_sum','mean'),rain_frequency=('rain_sum',lambda x:(x>0).mean()))
    fig,ax=plt.subplots(figsize=(9.2,4.3))
    ax.bar(range(1,13),monthly['rain_frequency']*100,color='#246B83',width=0.69)
    ax.set(xticks=range(1,13),xlabel='Bulan',ylabel='Hari hujan (%)',ylim=(0,105),
           title='Persentase hari hujan menurut bulan (seluruh periode)')
    save_figure('04_pola_musiman.png')
    # 5 confusion matrix
    cm=np.array(result['confusion_matrix'])
    fig,ax=plt.subplots(figsize=(6.5,5.1))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=False,ax=ax,annot_kws={'size':18})
    ax.set(xticklabels=['Tidak hujan','Hujan'],yticklabels=['Tidak hujan','Hujan'],
           xlabel='Prediksi model',ylabel='Kondisi aktual',title='Confusion matrix pada data uji (20%)')
    save_figure('05_confusion_matrix.png')
    # 6 roc curve
    fpr,tpr,thresholds=roc_curve(ytrue,prob)
    fig,ax=plt.subplots(figsize=(6.5,5.1))
    ax.plot(fpr,tpr,lw=2.5,color='#246B83',label=f"XGBoost (AUC = {result['roc_auc']:.3f})")
    ax.plot([0,1],[0,1],'--',color='#8996A4',label='Acak (AUC = 0.500)')
    ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='Kurva ROC data uji',xlim=(0,1),ylim=(0,1.02))
    ax.legend(loc='lower right')
    save_figure('06_roc_curve.png')
    # 7 grouped feature importance
    importance,_=feature_importance_grouped(model)
    sub=importance.iloc[:12].iloc[::-1]
    fig,ax=plt.subplots(figsize=(9.5,6.0))
    ax.barh(sub['feature_label'],sub['importance'],color='#246B83')
    ax.set(title='Feature importance XGBoost (diakumulasi per fitur)',xlabel='Total importance (gain bawaan model, relatif)')
    save_figure('07_feature_importance.png')
    # 8 train & test timeline
    fig,ax=plt.subplots(figsize=(10,3.4))
    ax.axvspan(train['time'].min(),train['time'].max(),alpha=0.3,color='#246B83',label='Training 80%')
    ax.axvspan(test['time'].min(),test['time'].max(),alpha=0.36,color='#E9A23B',label='Testing 20%')
    ax.plot(raw['time'],raw['rain_sum'],color='#48636C',lw=0.6,alpha=.65)
    ax.set(title='Pembagian data kronologis (tanpa pengacakan)',ylabel='Hujan harian (mm)',xlabel='Tanggal')
    ax.legend()
    save_figure('08_split_kronologis.png')
    return importance, corr


def execute():
    OUT.mkdir(parents=True,exist_ok=True)
    FIG.mkdir(parents=True,exist_ok=True)
    raw=raw_data()
    assert raw['time'].is_unique,'Waktu duplikat, validasi data terlebih dahulu.'
    assert raw['time'].is_monotonic_increasing
    df=prepare(raw)
    train,test=split_time(df)
    pipe=make_pipeline()
    pipe.fit(train[FEATURES],train[TARGET].astype(int))
    ytrue=test[TARGET].astype(int).to_numpy()
    pred=pipe.predict(test[FEATURES])
    prob=pipe.predict_proba(test[FEATURES])[:,1]
    scores=metrics_from_predictions(ytrue,pred,prob)
    importance,corr=make_charts(raw,df,train,test,pipe,ytrue,pred,prob,scores)
    scores.update({
        'raw_rows':len(raw),'raw_columns':raw.shape[1], 'samples':len(df),
        'n_train':len(train),'n_test':len(test),
        'train_start':str(train['time'].iloc[0].date()), 'train_end':str(train['time'].iloc[-1].date()),
        'test_start':str(test['time'].iloc[0].date()), 'test_end':str(test['time'].iloc[-1].date()),
        'raw_start':str(raw['time'].iloc[0].date()), 'raw_end':str(raw['time'].iloc[-1].date()),
        'missing_total':int(raw.isna().sum().sum()),'duplicates':int(raw.duplicated().sum()),
        'class_counts':{str(int(i)):int(v) for i,v in df[TARGET].value_counts().items()},
        'train_class_counts':{str(int(i)):int(v) for i,v in train[TARGET].value_counts().items()},
        'test_class_counts':{str(int(i)):int(v) for i,v in test[TARGET].value_counts().items()},
        'majority_baseline':float(max(ytrue.mean(),1-ytrue.mean())),
        'positive_rate_train':float(train[TARGET].mean()),
        'positive_rate_test':float(test[TARGET].mean()),
        'corr_rain_precip_hours':float(corr.loc['rain_sum','precipitation_hours']),
        'corr_max_mean':float(corr.loc['temperature_2m_mean','temperature_2m_max']),
        'top_features':importance[['feature','feature_label','importance']].head(8).to_dict(orient='records'),
        'model_params':{'n_estimators':100,'learning_rate':0.1,'max_depth':5,'random_state':42},
    })
    joblib.dump(pipe, OUT/'model_xgboost.joblib')
    importance.to_csv(OUT/'feature_importance.csv',index=False)
    test_preds=test[['time']].copy()
    test_preds['aktual_hujan_esok']=ytrue
    test_preds['prediksi_hujan_esok']=pred
    test_preds['probabilitas_hujan_esok']=prob
    test_preds.to_csv(OUT/'prediksi_data_uji.csv',index=False)
    raw.isna().sum().rename('jumlah_missing').to_csv(OUT/'missing_values.csv')
    raw.describe().to_csv(OUT/'statistik_deskriptif.csv')
    corr.to_csv(OUT/'korelasi.csv')
    (OUT/'metrik_evaluasi.json').write_text(json.dumps(scores,indent=2,ensure_ascii=False),encoding='utf8')
    (OUT/'laporan_klasifikasi.txt').write_text(classification_report(ytrue,pred,target_names=['Tidak Hujan','Hujan'],digits=4,zero_division=0),encoding='utf8')

    # Ekspor data penelitian yang telah dinormalisasi dan di-encoding.
    # Skala, imputasi dan kategori di-fit hanya pada data train.
    # Urutan baris train kemudian test dijaga untuk audit pemisahan waktu.
    mat_train = pipe['prep'].transform(train[FEATURES])
    mat_test = pipe['prep'].transform(test[FEATURES])
    feature_cols = pipe['prep'].get_feature_names_out().tolist()
    table_train = pd.DataFrame(mat_train, columns=feature_cols)
    table_test = pd.DataFrame(mat_test, columns=feature_cols)
    table_train.insert(0, 'split_data', 'train')
    table_test.insert(0, 'split_data', 'test')
    table_train.insert(0, 'tanggal', train['time'].dt.strftime('%Y-%m-%d').to_numpy())
    table_test.insert(0, 'tanggal', test['time'].dt.strftime('%Y-%m-%d').to_numpy())
    table_train[TARGET] = train[TARGET].astype(int).to_numpy()
    table_test[TARGET] = test[TARGET].astype(int).to_numpy()
    normalized = pd.concat([table_train, table_test], ignore_index=True)
    submission_csv = ROOT / 'MochammadMarilLivio_Test1_KlasifikasiXGBoost.csv'
    normalized.to_csv(submission_csv, index=False, float_format='%.8g')
    print('CSV NORMALISASI:', submission_csv.name, normalized.shape)

    print('DATASET',raw.shape,'RANGE',scores['raw_start'],scores['raw_end'])
    print('TRAIN',scores['n_train'],scores['train_start'],'→',scores['train_end'])
    print('TEST',scores['n_test'],scores['test_start'],'→',scores['test_end'])
    print('SCORES', {k:round(scores[k],4) for k in ['accuracy','precision','recall','f1','roc_auc','majority_baseline']})
    print('CM',scores['confusion_matrix'])
    print('IMPORTANCE',[(f['feature'],round(f['importance'],4)) for f in scores['top_features']])
    return scores


if __name__=='__main__':
    execute()
