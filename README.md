# Public Transport Delay Prediction

This is a learning project. The goal was to go through a complete data analysis and machine learning workflow from start to finish: load the raw data, explore it, build features, train a model, evaluate it. The data describes public transport trips (bus, metro, tram, train), and the task is to estimate how many minutes a given trip will be late on arrival.

## About the data

2000 rows, one month of trips (January 2023). Each row has the scheduled departure and arrival, the transport type, the route, weather info (temperature, humidity, wind, precipitation, weather category), a traffic congestion index, any nearby event (concert, sports, protest), holiday and peak hour flags, and the actual arrival delay in minutes. That last one is the target.

The raw CSV is not tracked (see `.gitignore`), only the processed version made it into the repo.

## Folder structure

```
data/
  raw/         the raw CSV, not in git
  processed/   transport_processed.csv, the model ready table
models/        the trained model and the train/test indices
notebooks/     the four notebooks, in order
```

## The notebooks

**01_eda.ipynb**
First look at the data. Types, missing values, basic statistics. Only one column has gaps, `event_type`, with 1173 empty cells. I did not drop those rows, I filled them with `No events`, because the gap itself carries information here: there simply was no event that day. Beyond that, a few plots showing delay by weather, transport type and season.

**02_feature_engineering.ipynb**
This is where the model ready table gets built. What I did:

- computed the scheduled travel time from the difference between departure and arrival
- pulled the hour out of the timestamp
- dropped the identifiers (`trip_id`, `route_id`, station names), since a model could only memorize those
- dropped `actual_departure_delay_min` as well, which was the most important decision here. Departure delay almost perfectly gives away the arrival delay, so leaving it in produces an unrealistically good score while being useless in practice, because we want to predict before the trip starts.
- turned the categorical columns into dummy variables. The `drop_first=False` is intentional, a redundant column is not an issue for a tree based model.

**03_modeling.ipynb**
Random Forest regressor with 200 trees and an 80/20 train/test split. I saved the split indices to `.npy` files so the evaluation notebook sees exactly the same test set and never accidentally measures on rows the model was trained on.

**04_evaluation.ipynb**
Loads the saved model and the test indices, then reports MAE and RMSE, an actual versus predicted scatter plot, a residual plot, and feature importances.

## Results

MAE: 7.96 minutes
RMSE: 9.51 minutes

These numbers need context though. If I simply guessed the training mean (13.39 minutes) for every trip, I would get MAE 7.70 and RMSE 9.19. So the Random Forest is a hair worse than the dumbest possible baseline.

This is not a bug in the code, it is just what the data is like. It is fairly likely a synthetic dataset where the delay is mostly random and the explanatory variables carry almost no signal. The residual plot shows the same thing: the predictions cluster in a narrow band around the mean while the real values spread much wider.

I decided to leave it as is rather than keep tuning until the numbers looked nicer on paper. The takeaway is worth something on its own: if there is no signal in the data, the choice of model will not save the project, and without a baseline comparison you cannot tell whether a result is any good.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in order. Notebooks 01 and 02 need the raw CSV at `data/raw/public_transport_delays.csv`, which is not in the repo. If you only care about the modeling part, 03 and 04 run fine on the processed CSV that is already there.

# Tömegközlekedési késés előrejelzés

Ez egy tanulóprojekt. A célja az volt, hogy végigcsináljak egy teljes adatelemzési és gépi tanulási folyamatot: nyers adat beolvasása, feltárás, feature engineering, modell tanítás, kiértékelés. Az adat tömegközlekedési utakat ír le (busz, metró, villamos, vonat), és azt próbáljuk megbecsülni, hogy egy adott út hány percet fog késni érkezéskor.

## Mit tartalmaz az adat

2000 sor, egyetlen hónapnyi (2023 január) út. Minden sorban ott van az indulási és érkezési menetrend, a jármű típusa, az útvonal, az időjárás (hőmérséklet, páratartalom, szél, csapadék, időjárás kategória), forgalmi index, esetleges esemény a környéken (koncert, sport, tüntetés), ünnepnap és csúcsidő jelző, valamint a tényleges érkezési késés percben. Ez utóbbi a célváltozó.

A nyers CSV nincs verziókövetve (lásd `.gitignore`), csak a feldolgozott változat került be a repóba.

## Könyvtárszerkezet

```
data/
  raw/         a nyers CSV, nincs gitben
  processed/   transport_processed.csv, a modellezésre kész tábla
models/        a betanított modell és a train/test indexek
notebooks/     a négy munkafüzet, sorrendben
```

## A notebookok

**01_eda.ipynb**
Első ránézés az adatra. Típusok, hiányzó értékek, alapstatisztikák. Egyetlen oszlopban van hiány, az `event_type`, ott 1173 üres cella. Ezeket nem dobtam el, hanem `No events` értékre töltöttem, mert a hiány itt tényleges információ: nem volt esemény aznap. Ezen kívül néhány ábra: késés eloszlása időjárás, járműtípus és évszak szerint.

**02_feature_engineering.ipynb**
Itt készül el a modellezhető tábla. Amit csináltam:

- kiszámoltam a menetrend szerinti menetidőt az indulás és érkezés különbségéből
- kiszedtem az óra értéket az időpontból
- eldobtam az azonosítókat (`trip_id`, `route_id`, állomásnevek), mert ezekből a modell csak memorizálni tudna
- eldobtam az `actual_departure_delay_min` oszlopot is, ez volt a legfontosabb döntés. Az indulási késés majdnem tökéletesen elárulja az érkezésit, szóval ha bent marad, a modell irreálisan jó eredményt ad, viszont a valóságban nem tudnánk használni, hiszen indulás előtt akarunk becsülni.
- a kategóriákból dummy változókat csináltam. A `drop_first=False` szándékos, fa alapú modellnél nem probléma a redundáns oszlop.

**03_modeling.ipynb**
Random Forest regresszor, 200 fával, 80/20 arányú train/test bontással. A split indexeit kimentettem `.npy` fájlba, hogy a kiértékelő notebook pontosan ugyanazt a teszthalmazt lássa, és véletlenül se olyan sorokon mérjek, amiken a modell tanult.

**04_evaluation.ipynb**
Betölti a kimentett modellt és a teszt indexeket, majd MAE és RMSE, tényleges és becsült értékek szórásdiagramja, reziduál ábra, és feature importance.

## Eredmények

MAE: 7.96 perc
RMSE: 9.51 perc

Ezt viszont érdemes kontextusba tenni. Ha egyszerűen a tanító halmaz átlagát (13.39 perc) tippelném minden útra, akkor MAE 7.70 és RMSE 9.19 jönne ki. Vagyis a Random Forest egy hajszállal rosszabb, mint a legbutább lehetséges baseline.

Ez nem hiba a kódban, egyszerűen erről szól az adat. Elég valószínű, hogy szintetikus adathalmazról van szó, ahol a késés nagyrészt véletlen, és a magyarázó változók alig hordoznak jelet. A reziduál ábrán is jól látszik: a becslések szűk sávba tömörülnek az átlag körül, miközben a valóság szórása sokkal nagyobb.

Ezt inkább meghagytam így, mint hogy addig csavarjam a modellt, amíg papíron szebb szám jön ki. A tanulság önmagában értékes: ha nincs jel az adatban, a modellválasztás nem menti meg a projektet, és a baseline összehasonlítás nélkül nem tudni, hogy egy eredmény jó vagy sem.

## Futtatás

```bash
pip install -r requirements.txt
jupyter notebook
```

A notebookokat sorrendben érdemes futtatni. A 01 és a 02 igényli a nyers CSV-t a `data/raw/public_transport_delays.csv` útvonalon, ez nincs a repóban. Ha csak a modellezés érdekel, a 03 és 04 elfut a már bent lévő feldolgozott CSV-vel is.
