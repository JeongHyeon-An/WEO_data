## World Economics Outlook Data Analysis

import pandas as pd

# load data
df = pd.read_csv('WEOApr2020all.csv')
# print(df.head(3))

# describe
# print(df.describe())

# check columns
# print(df.columns)

# # check country
# print(df['Country'].nunique()) # --> 194개의 나라 존재 확인
#
# # access by iloc == country 확인
# print(df.iloc[:,3])
#
# # check 'Subject Descriptor'
# print(df['Subject Descriptor'])
#
# # check first 7 rows
# print(df['Subject Descriptor'].head(7))
#
# # check 'Subject Descriptor' and '2020'
# print(df[['Subject Descriptor', '2020']]) # ---> 하나의 리스트로 만들어줘야함
#
# # check 'Country', 'Subject Descriptor' and '2020'
# print(df[['Country', 'Subject Descriptor', '2020']])


### Inflation

# filter 'Inflation, end of period consumer prices'
idx_inf = df['Subject Descriptor'].str.contains('Inflation, end of') # --> df 'Sub~'에서 문자 'Infla~'를 포함하는 데이터 출력
print(idx_inf) # ---> True == 내가 찾고자 하는 데이터

# filter data
print(df.loc[idx_inf])

# save to df_inf
df_inf = df[idx_inf]

# check data
print(df_inf.head(3))

# check 'Country' and '2021'
df_inf_2021 = df_inf[['Country','2021']]
print(df_inf_2021)

# reset row index (원래 index 값을 0부터 초기화시키기)
df_inf_2021.reset_index(drop=True, inplace=True)
print(df_inf_2021) # ---> 0부터 초기화된 index 확인 가능

# check data year type
print(df_inf_2021.dtypes) # --> objects인 데이터 타입을 숫자로 변경할 예정

# check info
print(df_inf_2021.info()) # --> 2021을 보면, 190개만 데이터가 들어가있고 4개는 값이 없다는 것을 알 수 있음

# get only non-null data type
# df_inf_2021에서 2021d에 데이터가 있는 location만 출력
df_inf_2021 = df_inf_2021.loc[df_inf_2021['2021'].notnull()]
print(df_inf_2021) # --> 190rows x 2columns

# check info again == null 데이터가 들어갔는지 확인하기
print(df_inf_2021.info())

# remove commas == 큰 숫자들에 ','가 들어가 있는데, ',' 없애기
df_inf_2021 = df_inf_2021.replace(',', '',regex=True)
print(df_inf_2021)

# change to numeric
df_inf_2021['2021'] = pd.to_numeric(df_inf_2021['2021']) # --> 2021에 있는 2021데이터타입을 numeric으로 바꾸기
print(df_inf_2021)
print(df_inf_2021.dtypes) # 2021 == float64

# plot bar graph
import matplotlib.pyplot as plt

df_inf_2021.sort_values('2021').plot.bar(x='Country') # 2021 기준으로 정렬 + plot --> 너무 많은 나라때문에 확인하기 힘듦
# plt.show()
# plot first 20 countries
df_inf_2021.sort_values('2021').iloc[0:20,:].plot.bar(x='Country')
# plt.show()


### Unemployment Rate

# filter 'Unemployment Rate'
df_ur = df[df['Subject Descriptor'].str.contains('Unemployment')]
print(df_ur) # --> 실업률이 적힌 부분을 포함하는 데이터를 가진 index를 df에 넣어줌 == df_ur

df_ur = df[df['Subject Descriptor'].str.contains('Unemployment')][['Country','2021']]
print(df_ur)

# check info
print(df_ur.info()) # 194개의 데이터 중, 2021에는 94개의 데이터가 null인 상태

# filter out NaN rows
df_ur = df_ur.loc[df_ur['2021'].notnull()]
print(df_ur) # 100 x 2

# reset row index (0부터 index 시작)
df_ur.reset_index(drop=True, inplace=True)
print(df_ur)


# check info
print(df_ur.info()) # object --> 숫자 데이터로 바꿀 예정

# convert to numeric data
df_ur['2021'] = pd.to_numeric(df_ur['2021'])
print(df_ur.info()) # object --> float64

# plot bar graph
df_ur.plot.bar(x='Country')
# plt.show()

# sort and plot
df_ur.sort_values('2021',ascending=False).plot.bar(x='Country',
                                                   title='Unemployment Rate',
                                                   figsize=(15,5))
# plt.show()
print(df_ur) # --> 높은 것부터 출력됨



### Dataframe changes to numpy

# to numpy
df_ur_numpy = df_ur.sort_values('2021', ascending=False).to_numpy()
print(df_ur_numpy)

# import numpy and matplotlib
import matplotlib.pyplot as plt
import numpy as np

# plot
plt.rcParams['figure.figsize']=(20,3) # fig 사이즈 정하기
plt.xticks(rotation='vertical') # 제목을 세로방향으로 넣기
plt.bar(df_ur_numpy[:,0], df_ur_numpy[:,1]) # country , unemployment
plt.title('2021 Unemployment Rate - IMF Outlook Database, April 2020')


# plot Korea Data again
idx = np.where(df_ur_numpy=='Korea')
print(idx) # (array([80], dtype=int64), array([0], dtype=int64))
plt.bar(df_ur_numpy[idx[0],0], df_ur_numpy[idx[0],1], label='Korea')

idx = np.where(df_ur_numpy=='United States')
plt.bar(df_ur_numpy[idx[0],0], df_ur_numpy[idx[0],1], label='United States')

idx = np.where(df_ur_numpy=='China')
plt.bar(df_ur_numpy[idx[0],0], df_ur_numpy[idx[0],1], label='China')# --> 새로운 데이터 들어가서 색 바뀜
plt.legend()
# plt.show()




### Pandas tip1 1) Column Exchange

# snapping columns
print(df.head(3))

# check length
print(len(df.columns)) # 52

# set up a list
idx = list(range(0,52))
print(idx)
# update indices to swap col 3 and 0
idx[3] = 0
idx[0] = 3
print(idx)

print(df.iloc[:,idx]) # country name이 0번째, code가 3번째로 옮겨짐 (칼럼이 바껴짐)



### Pandas tip2 2) Add new col and group analysis

df_ur['Criteria'] = 'NONE'
print(df_ur) # --> Criteria라는 col를 만들어서 값은 NONE으로 채워줌


# update Criteria to low, medium and high (3그룹으로 나누기)
df_ur.loc[df_ur['2021'] < 5, 'Criteria'] = 'Low' # 5%보다 낮은 실업률 선택 -> Criteria에 Low라고 입력
df_ur.loc[(5 <= df_ur['2021']) & (df_ur['2021']< 10), 'Criteria'] = 'Medium'
df_ur.loc[10 <= df_ur['2021'], 'Criteria'] = 'High'
print(df_ur)

# find mean for each Criteria
print(df_ur.groupby(['Criteria']).mean())
#                2021
# Criteria
# High      14.767680
# Low        3.547000
# Medium     7.100353

# find mean for each Criteria and sort
print(df_ur.groupby(['Criteria']).mean().sort_values('2021'))
#                2021
# Criteria
# Low        3.547000
# Medium     7.100353
# High      14.767680

# count the number of countries in each Criteria
print(df_ur.groupby(['Criteria']).count())
#           Country  2021
# Criteria
# High           25    25
# Low            24    24
# Medium         51    51


### Pandas tip2 2) Loading Small data Chunks using Forloop

#파일의 크기가 너무 클 경우에는, 내가 원하는 chunksize 만큼 나눠서 출력 가능
for df_chunk in pd.read_csv('WEOApr2020all.csv', chunksize=5):
    print(df_chunk)

df_new = pd.DataFrame(columns=df.columns) # --> data 없이 칼럼만 출력
print(df_new)

for df_chunk in pd.read_csv("WEOApr2020all.csv", chunksize=5):
    temp = df_chunk.loc[df_chunk['Subject Descriptor']=='Unemployment rate'] # 'Sub~'에서 실업률과 같은게 있는 자리만 뽑아서 출력
    df_new = pd.concat([df_new, temp]) # df_new에다가 temp 연결
    print(df_new)
