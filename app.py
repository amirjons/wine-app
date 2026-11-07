import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

# Почему кэширование устроено именно так.
# st.cache_resource хранит сам объект (одну копию на весь сервер, без копирования и сериализации):
# так нужно для модели - тяжёлого объекта, который не должен дублироваться для каждой сессии.
# st.cache_data при каждом вызове возвращает копию результата (DataFrame, массивы), поэтому
# подходит для данных и вычисляемых из них таблиц: их нельзя случайно испортить между сессиями.
# Без декоратора joblib.load вызывался бы при каждом действии пользователя (в том числе при
# каждом движении слайдера): модель каждый раз читалась бы с диска, приложение работало бы медленнее.

APP_DIR = Path(__file__).parent

st.set_page_config(page_title='Wine Quality', page_icon='🍷', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / 'test.csv')


@st.cache_data
def load_config():
    return json.loads((APP_DIR / 'config.json').read_text())


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


@st.cache_data
def test_curves(y, proba):
    fpr, tpr, _ = metrics.roc_curve(y, proba)
    prec, rec, _ = metrics.precision_recall_curve(y, proba)
    return fpr, tpr, prec, rec


model = load_model()
test = load_test()
config = load_config()

FEATURES = [c for c in test.columns if c != 'target']
X_test = test[FEATURES]
y_test = test['target'].to_numpy()
proba_test = test_proba(model, X_test)

DECIMALS = {'fixed_acidity': 1, 'volatile_acidity': 3, 'citric_acid': 2, 'residual_sugar': 1,
            'chlorides': 3, 'free_sulfur_dioxide': 1, 'total_sulfur_dioxide': 1,
            'density': 5, 'pH': 2, 'sulphates': 2, 'alcohol': 1}

st.title('🍷 Рекомендатель хороших вин')

st.sidebar.header('Настройки модели')
threshold = st.sidebar.slider('Порог вероятности', 0.0, 1.0, float(config['threshold']), 0.001, key='threshold')
st.sidebar.caption('Вино считается хорошим, если вероятность класса 1 не меньше порога.')

tab_one, tab_metrics, tab_batch, tab_weights = st.tabs(
    ['Оценка вина', 'Модельные метрики', 'Пакетное предсказание', 'Что выучила модель'])

with tab_one:
    st.subheader('Параметры вина')
    values = {}
    cols = st.columns(3)
    numeric = [f for f in FEATURES if f != 'is_red']
    for i, f in enumerate(numeric):
        dec = DECIMALS.get(f, 2)
        with cols[i % 3]:
            values[f] = st.number_input(
                f, min_value=float(X_test[f].min()), max_value=float(X_test[f].max()),
                value=float(X_test[f].median()), step=10.0 ** -dec, format=f'%.{dec}f', key=f'in_{f}')
    with cols[len(numeric) % 3]:
        is_red = st.toggle('Красное вино (выкл. - белое)', value=bool(X_test['is_red'].median() >= 0.5))
    values['is_red'] = int(is_red)

    row = pd.DataFrame([values])[FEATURES]
    p_one = float(model.predict_proba(row)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric('Вероятность хорошего вина', f'{p_one:.1%}')
    c2.metric('Порог', f'{threshold:.3f}')
    if p_one >= threshold:
        st.success('Решение модели: хорошее вино - рекомендовать в подборку.')
    else:
        st.warning('Решение модели: не рекомендовать в подборку.')

with tab_metrics:
    pred = (proba_test >= threshold).astype(int)
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    m1, m2, m3 = st.columns(3)
    m1.metric('Precision', f'{precision:.3f}')
    m2.metric('Recall', f'{recall:.3f}')
    m3.metric('F1', f'{f1:.3f}')

    fpr, tpr, prec_c, rec_c = test_curves(y_test, proba_test)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['обычное', 'хорошее']).plot(
        ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].grid(False)
    axes[0].set_title('Матрица ошибок')
    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].scatter([fp / (fp + tn)], [tp / (tp + fn)], color='red', s=70, zorder=3, label='текущий порог')
    axes[1].set(xlabel='FPR', ylabel='TPR', title='ROC-кривая')
    axes[1].legend()
    axes[2].plot(rec_c, prec_c)
    axes[2].axhline(y_test.mean(), color='k', linestyle='--', alpha=0.5, label='случайная модель')
    axes[2].scatter([recall], [precision], color='red', s=70, zorder=3, label='текущий порог')
    axes[2].set(xlabel='Recall', ylabel='Precision', title='PR-кривая')
    axes[2].legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.write(f'При пороге {threshold:.3f} в подборку попадает {tp + fp} вин из {len(y_test)}, '
             f'из них хороших - {tp}; пропущено хороших вин - {fn}.')
    st.info('**Для сомелье.** Если порог увеличить, в подборку попадает меньше вин, но среди них больше '
            'действительно хороших (precision растёт), зато больше хороших вин пропускается (recall падает). '
            'Если порог уменьшить, подборка становится шире и полнее, но в неё чаще попадают посредственные вина.')

with tab_batch:
    st.subheader('Загрузка CSV с признаками вин')
    st.caption('Нужны столбцы: ' + ', '.join(FEATURES) + '. Столбец target не нужен. '
               f'Порог берётся из бокового слайдера (сейчас {threshold:.3f}).')
    uploaded = st.file_uploader('CSV-файл', type='csv')
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f'Не удалось прочитать файл как CSV: {e}')
            batch = None
        if batch is not None:
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error('В файле не хватает столбцов: ' + ', '.join(missing))
            else:
                X_batch = batch[FEATURES].apply(pd.to_numeric, errors='coerce')
                bad_rows = int(X_batch.isna().any(axis=1).sum())
                if bad_rows:
                    st.error(f'В {bad_rows} строках есть пропуски или нечисловые значения в признаках.')
                elif len(X_batch) == 0:
                    st.error('В файле нет строк с данными.')
                else:
                    p_batch = model.predict_proba(X_batch)[:, 1]
                    result = batch.copy()
                    result['proba_good'] = p_batch
                    result['decision'] = np.where(p_batch >= st.session_state['threshold'], 'хорошее', 'не рекомендовать')
                    st.dataframe(result, width='stretch')
                    st.download_button('Скачать результат (CSV)', result.to_csv(index=False).encode('utf-8'),
                                       file_name='predictions.csv', mime='text/csv')

with tab_weights:
    st.subheader('Коэффициенты логистической регрессии (признаки стандартизованы)')
    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(coefs.index, coefs.values, color=np.where(coefs.values >= 0, 'tab:green', 'tab:red'))
    ax.axvline(0, color='k', linewidth=0.8)
    ax.set_xlabel('Коэффициент')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    pos = coefs[coefs > 0].sort_values(ascending=False).head(3).index.tolist()
    neg = coefs[coefs < 0].head(3).index.tolist()
    st.write('Сильнее всего «голосуют» **за** хорошее вино: ' + ', '.join(pos) + '.')
    st.write('Сильнее всего «голосуют» **против**: ' + ', '.join(neg) + '.')
