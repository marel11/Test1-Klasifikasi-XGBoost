"""Dashboard Streamlit — XGBoost klasifikasi hujan esok hari Kolkata.
Mulai: streamlit run app.py
"""
from pathlib import Path
from datetime import date
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from analysis_pipeline import (DATA_PATH, ROOT, OUT, FIG, FEATURES, NUMERIC,
                               CODE_NAMES, NAMES, raw_data, prepare, split_time)

st.set_page_config(page_title='Kolkata | Prediksi Hujan XGBoost',page_icon='🌧️',layout='wide',initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container {padding-top:1.8rem;max-width:1340px;}
h1,h2,h3 {letter-spacing:-0.025em;}
[data-testid="stMetric"] {background-color:rgba(111,143,157,.085);border:1px solid rgba(120,120,120,.13);border-radius:12px;padding:15px;}
.report-note {padding:12px 14px;border-left:3px solid #246B83;background:rgba(124,149,170,.08);border-radius:4px;font-size:.92rem;}
</style>''',unsafe_allow_html=True)

@st.cache_data
def load_data():
    raw=raw_data()
    df=prepare(raw)
    training, testing=split_time(df)
    return raw, df, training, testing

@st.cache_resource
def load_model():
    return joblib.load(OUT/'model_xgboost.joblib')

@st.cache_data
def load_results():
    metrics=json.loads((OUT/'metrik_evaluasi.json').read_text(encoding='utf-8'))
    importance=pd.read_csv(OUT/'feature_importance.csv')
    preds=pd.read_csv(OUT/'prediksi_data_uji.csv',parse_dates=['time'])
    return metrics, importance, preds

if not DATA_PATH.exists():
    st.error('Dataset tidak ditemukan. Letakkan CSV pada folder data/ sesuai README.md.')
    st.stop()
if not (OUT/'model_xgboost.joblib').exists():
    st.error('Model belum dilatih. Jalankan dahulu: python MochammadMarilLivio_Test1_KlasifikasiXGBoost.py')
    st.stop()

raw,df,train,test=load_data()
score,importance,holdout=load_results()

st.sidebar.markdown('### 🌧️ Kolkata Weather Lab')
st.sidebar.caption('Klasifikasi cuaca menggunakan XGBoost')
st.sidebar.markdown('**Dataset:** 30 Juni 2021 – 30 Juni 2026')
st.sidebar.markdown('**Target:** hujan pada hari berikutnya')
st.sidebar.markdown('**Pemisahan:** 80/20 kronologis')
st.sidebar.divider()
st.sidebar.info('Model menggunakan data pengamatan satu hari penuh untuk menilai peluang hujan pada hari berikutnya. Ini adalah demonstrasi praktikum, bukan prakiraan cuaca resmi.')

st.title('🌦️ Klasifikasi Hujan Esok Hari dengan XGBoost')
st.caption('Studi kasus data cuaca Kolkata, India · Dataset CSV terlampir · Dashboard interaktif Streamlit')

overview, eda, evaluation, prediction, method = st.tabs([
    '📌 Ringkasan Dataset','📊 Eksplorasi Data (EDA)','🎯 Evaluasi Model','🔮 Prediksi Manual','📘 Metode & Catatan'
])

with overview:
    a,b,c,d=st.columns(4)
    a.metric('Total observasi',f'{len(raw):,}')
    b.metric('Kolom asli',f'{raw.shape[1]}')
    c.metric('Sampel klasifikasi',f'{len(df):,}')
    d.metric('Missing values',f'{int(raw.isna().sum().sum())}')
    st.subheader('Distribusi kelas target')
    lc,rc=st.columns([1,1])
    with lc:
        counts=df['rain_tomorrow'].value_counts().reindex([0,1],fill_value=0)
        chart=px.pie(names=['Tidak hujan','Hujan'],values=counts.values,hole=.42,
                     color_discrete_sequence=['#246B83','#E9A23B'])
        chart.update_traces(textposition='inside',textinfo='percent+label')
        chart.update_layout(height=350,showlegend=False,margin=dict(l=5,r=5,b=5,t=5))
        st.plotly_chart(chart,use_container_width=True)
        st.caption(f'Dari {len(df):,} sampel, {int(counts[1]):,} berlabel hujan esok hari ({counts[1]/len(df):.1%}). Kelas relatif berimbang.')
    with rc:
        st.image(str(FIG/'08_split_kronologis.png'),caption='Gambar: Pemisahan train/test menurut tanggal',use_container_width=True)
        st.caption('Data sebelum batas waktu digunakan untuk pelatihan, sementara data terbaru dipertahankan untuk pengujian.')
    st.subheader('Pratinjau CSV')
    st.dataframe(raw.head(12),use_container_width=True,hide_index=True)
    with st.expander('Tipe data, nilai hilang, dan statistik deskriptif'):
        st.dataframe(pd.DataFrame({'kolom':raw.columns,'tipe_data':raw.dtypes.astype(str).values,
                                   'missing':raw.isna().sum().values,'unik':raw.nunique().values}),hide_index=True)
        st.dataframe(raw.describe().T,use_container_width=True)

with eda:
    st.header('Exploratory Data Analysis (EDA)')
    choices=['temperature_2m_mean','temperature_2m_max','temperature_2m_min','rain_sum',
             'precipitation_hours','wind_speed_10m_max','wind_gusts_10m_max','sunshine_hours']
    chosen=st.selectbox('Pilih variabel numerik untuk histogram',choices,
                        format_func=lambda x:NAMES.get(x,x))
    chart=px.histogram(df,x=chosen,nbins=32, color_discrete_sequence=['#246B83'],
                       labels={chosen:NAMES[chosen]},title=f'Distribusi {NAMES[chosen]}')
    chart.update_layout(height=380,bargap=.05)
    st.plotly_chart(chart,use_container_width=True)
    st.caption('Histogram memperlihatkan rentang nilai, konsentrasi pengamatan, dan kemungkinan nilai ekstrem. Curah hujan umumnya menceng ke kanan karena banyak hari tanpa atau dengan sedikit hujan.')
    st.subheader('Korelasi antarfitur')
    corr=pd.read_csv(OUT/'korelasi.csv',index_col=0)
    fig=px.imshow(corr, color_continuous_scale='BrBG',zmin=-1,zmax=1,text_auto='.2f',aspect='auto',
                  title='Heatmap korelasi Pearson')
    fig.update_layout(height=550)
    st.plotly_chart(fig,use_container_width=True)
    st.caption('Korelasi bukan kausalitas. Perhatikan kemiripan antara suhu minimum/rata-rata/maksimum dan hubungan curah hujan dengan lama presipitasi; fitur duplikat persis precipitation_sum telah dikeluarkan.')
    st.subheader('Pola musim dan distribusi target')
    left,right=st.columns(2)
    with left:
        mo=raw.assign(month=raw.time.dt.month).groupby('month').agg(persen_hujan=('rain_sum',lambda s:100*(s>0).mean())).reset_index()
        fg=px.bar(mo,x='month',y='persen_hujan',title='Persentase hari hujan menurut bulan',
                  labels={'month':'Bulan','persen_hujan':'Hari hujan (%)'},color_discrete_sequence=['#246B83'])
        fg.update_xaxes(dtick=1)
        st.plotly_chart(fg,use_container_width=True)
        st.caption('Periode Juli–September lebih sering mengalami hari hujan dibanding Desember–Februari; pola musiman relevan untuk prediksi esok hari.')
    with right:
        f=px.bar(x=['Tidak hujan','Hujan'],y=[int(counts[0]),int(counts[1])],
                 labels={'x':'Target esok hari','y':'Jumlah'},title='Proporsi kelas target',
                 color=['Tidak hujan','Hujan'],color_discrete_map={'Tidak hujan':'#246B83','Hujan':'#E9A23B'})
        f.update_layout(showlegend=False)
        st.plotly_chart(f,use_container_width=True)
        st.caption('Kelas target tidak sepenuhnya sama besar, sehingga evaluasi tidak hanya mengandalkan accuracy.')
    st.subheader('Pemeriksaan kualitas data')
    missing=raw.isna().sum().sort_values(ascending=False).rename_axis('kolom').reset_index(name='missing')
    st.dataframe(missing,use_container_width=True,hide_index=True)
    st.success('Tidak ditemukan nilai kosong maupun baris duplikat pada CSV asli.')

with evaluation:
    st.header('Performa XGBoost pada Data Uji')
    cols=st.columns(5)
    for col,(label,key) in zip(cols,[('Accuracy','accuracy'),('Precision','precision'),
                                 ('Recall','recall'),('F1-Score','f1'),('ROC-AUC','roc_auc')]):
        col.metric(label,f'{score[key]*100:.2f}%')
    st.caption(f"Training {score['n_train']} sampel ({score['train_start']} s.d. {score['train_end']}) · "
               f"Testing {score['n_test']} sampel ({score['test_start']} s.d. {score['test_end']})")
    metr=pd.DataFrame([{'Metrik':k,'Nilai':score[k],'Persentase':f'{100*score[k]:.2f}%'}
        for k in ['accuracy','precision','recall','f1','roc_auc']])
    st.dataframe(metr,use_container_width=True,hide_index=True)
    st.subheader('Confusion matrix dan ROC curve')
    c1,c2=st.columns(2)
    with c1:
        st.image(str(FIG/'05_confusion_matrix.png'),use_container_width=True)
        st.markdown(f"**Interpretasi:** TP = {score['tp']} hari hujan dikenali benar, TN = {score['tn']} hari tidak hujan dikenali benar, FP = {score['fp']} alarm hujan keliru, dan FN = {score['fn']} hujan terlewat.")
    with c2:
        st.image(str(FIG/'06_roc_curve.png'),use_container_width=True)
        st.markdown(f"**Interpretasi:** ROC-AUC {score['roc_auc']:.3f} menunjukkan model cukup baik membedakan kelas pada berbagai ambang probabilitas; angka ini adalah hasil uji pada periode tertentu.")
    st.subheader('Variabel berpengaruh menurut model')
    ranked=importance.sort_values('importance',ascending=True)
    feature_plot=px.bar(ranked,x='importance',y='feature_label',orientation='h',
                         color_discrete_sequence=['#246B83'],
                         title='Feature Importance (akumulasi skor penting per fitur asal)',
                         labels={'importance':'Importance relatif','feature_label':'Fitur'})
    feature_plot.update_layout(height=560)
    st.plotly_chart(feature_plot,use_container_width=True)
    st.caption(f"Variabel paling menonjol ialah {importance.iloc[0]['feature_label']} ({importance.iloc[0]['importance']:.1%}). Nilai importance bukan bukti hubungan sebab-akibat atau pengaruh statistik yang pasti.")
    st.subheader('Perbandingan hasil klasifikasi')
    classes={0:'Tidak hujan',1:'Hujan'}
    display=holdout.copy()
    display['Kondisi aktual']=display['aktual_hujan_esok'].map(classes)
    display['Prediksi model']=display['prediksi_hujan_esok'].map(classes)
    display['Probabilitas hujan (%)']=(display['probabilitas_hujan_esok']*100).round(2)
    st.dataframe(display[['time','Kondisi aktual','Prediksi model','Probabilitas hujan (%)']].head(120),
                 use_container_width=True,hide_index=True)
    st.caption(f"Akurasi tebakan kelas terbanyak tanpa model pada data uji hanya {score['majority_baseline']*100:.2f}%, sehingga XGBoost memberikan peningkatan bermakna pada set pengujian ini.")

with prediction:
    st.header('Simulasi Prediksi Hari Berikutnya')
    st.write('Masukkan hasil pengamatan cuaca **hari ini**. Setelah selesai satu hari pengamatan, model memperkirakan apakah esok hari terjadi hujan.')
    with st.form('manual_pred'):
        obs_date=st.date_input('Tanggal pengamatan hari ini',value=raw.time.max().date())
        l,r=st.columns(2)
        vals={}
        with l:
            for key in ['temperature_2m_mean','temperature_2m_max','temperature_2m_min','rain_sum','precipitation_hours']:
                vals[key]=st.number_input(NAMES[key],value=float(round(raw[key].median(),2)),min_value=0.0 if key in ['rain_sum','precipitation_hours'] else None,
                                          max_value=24.0 if key=='precipitation_hours' else None,step=0.1)
        with r:
            for key in ['wind_speed_10m_max','wind_gusts_10m_max','wind_direction_10m_dominant']:
                vals[key]=st.number_input(NAMES[key],value=float(round(raw[key].median(),2)),
                                          min_value=0.0,max_value=360.0 if key=='wind_direction_10m_dominant' else None,step=0.1)
            sunshine_default=float(round(raw['sunshine_duration'].median()/3600,2))
            vals['sunshine_hours']=st.number_input(NAMES['sunshine_hours'],min_value=0.0,max_value=24.0,
                                                  value=sunshine_default,step=0.1)
            opt=sorted(CODE_NAMES,key=lambda x:int(x))
            choice=st.selectbox('Kode cuaca hari ini (WMO)',options=opt,index=opt.index('3'),
                                format_func=lambda x:f'{x} — {CODE_NAMES[x]}')
        submit=st.form_submit_button('Prediksi kemungkinan hujan esok hari',type='primary',use_container_width=True)
    if submit:
        errors=[]
        if not vals['temperature_2m_min'] <= vals['temperature_2m_mean'] <= vals['temperature_2m_max']:
            errors.append('Suhu minimum ≤ suhu rata-rata ≤ suhu maksimum.')
        if vals['wind_gusts_10m_max'] < vals['wind_speed_10m_max']:
            errors.append('Hembusan angin maksimum semestinya tidak lebih kecil dari kecepatan angin maksimum.')
        if errors:
            st.warning('Harap perbaiki input: '+ ' '.join(errors))
        else:
            month=obs_date.month
            vals.update({'weather_code':choice,'month_sin':np.sin(2*np.pi*month/12),
                         'month_cos':np.cos(2*np.pi*month/12)})
            data=pd.DataFrame([vals])[FEATURES]
            prob=float(load_model().predict_proba(data)[0,1])
            label='HUJAN' if prob>=.5 else 'TIDAK HUJAN'
            st.subheader(f'Hasil prediksi: {label}')
            st.metric('Probabilitas hujan pada hari berikutnya',f'{prob*100:.1f}%')
            st.progress(int(round(prob*100)))
            st.caption('Ambang klasifikasi = 50%. Probabilitas prediktif belum dikalibrasi; hasil hanya estimasi model praktikum.')
            st.markdown('<div class="report-note">Catatan: fitur saat ini merupakan pengamatan harian lengkap. Jangan menggunakan variabel hari esok untuk memprediksi hari esok karena akan menimbulkan kebocoran data.</div>',unsafe_allow_html=True)

with method:
    st.header('Metodologi')
    st.markdown('''
1. **Akuisisi**: membaca CSV cuaca Kolkata, 1.827 hari berurutan.
2. **Target**: `rain_tomorrow = 1` jika `rain_sum` **hari t+1** lebih dari 0 mm, selain itu 0. Hari terakhir tidak digunakan karena tidak ada label esoknya.
3. **Fitur**: suhu, hujan hari t, jam presipitasi hari t, angin, matahari, kode WMO, dan fitur sin/cos bulan.
4. **Pra-pemrosesan**: `weather_code` sebagai kategori via One-Hot Encoding; nilai kosong (jika muncul) diisi median atau modus dari **data latih saja**. Variabel numerik dinormalisasi dengan MinMaxScaler yang di-fit pada data latih saja.
5. **Menghindari kebocoran**: data train dan test dipisah 80:20 **urut waktu**. Tidak ada `precipitation_sum` karena sama dengan `rain_sum`, dan label selalu berasal dari keesokan hari.
6. **Model**: XGBoost `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42`.
7. **Evaluasi**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix serta feature importance.
''')
    st.warning('Sumber asli berkas CSV tidak disertakan. Nama kolom dan satuannya cocok dengan dokumentasi Open-Meteo Historical Weather API, tetapi itu tidak membuktikan CSV memang diunduh dari layanan tersebut.')
    st.link_button('Dokumentasi parameter Open-Meteo','https://open-meteo.com/en/docs/historical-weather-api')
    st.info('Kinerja pada satu rentang uji historis belum tentu sama pada kota lain, tahun berbeda, atau sistem prakiraan cuaca riil. Gunakan data eksternal dan validasi berulang untuk pengembangan.')
    st.write('Kode Python, CSV, metrik, laporan, dan semua gambar telah disediakan dalam paket praktikum.')
