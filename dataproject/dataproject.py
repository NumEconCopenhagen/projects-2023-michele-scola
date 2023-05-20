import matplotlib.pyplot as plt
import numpy as np

def plot(cov, country): 
    I = cov['country'] == country
    ax=cov.loc[I,:].plot(x='day', y='death', style='-', legend=False)
    ax.set_title('covid-19 death by country')
    ax.set_ylabel('death');

def plot2(bed, country): 
    I = bed['country'] == country
    ax=bed.loc[I,:].plot(x='year', y='beds', style='-o', legend=False)
    plt.axhline(y=np.mean(bed.loc[I,'beds']), color='r', label='mean')
    ax.set_title('beds per 1000 people by country')
    ax.set_ylabel('beds/1000 people')
    ax.legend(loc='upper left', prop={'size': 10});

def plot3(den, country): 
    I = den['country'] == country
    ax=den.loc[I,:].plot(x='year', y='density', style='-o', legend=False)
    plt.axhline(y=np.mean(den.loc[I,'density']), color='r', label='mean')
    ax.set_title('density (people per sq.km) by country')
    ax.set_ylabel('density')
    ax.legend(loc='upper left', prop={'size': 10});

def plot4(inner2, country):
    I = inner2['country'] == country
    ax = inner2.loc[I,:].plot(x='year', y='density', style='-o', legend=False)
    plt.axhline(y=np.mean(inner2.loc[I,'density']), color='r', label='mean')
    ax.set_title('density (people per sq.km) by country')
    ax.set_ylabel('density')
    ax.legend(loc='upper left', prop={'size': 10});
    ax = inner2.loc[I,:].plot(x='year', y='beds', style='-o', legend=False)
    plt.axhline(y=np.mean(inner2.loc[I,'beds']), color='r', label='mean')
    ax.set_title('beds by country')
    ax.set_ylabel('beds per 1000 people')
    ax.legend(loc='upper left', prop={'size': 10});

def red_country(inner, country):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    I = inner['country'] == country
    ax.scatter(inner.loc[:,'mean'],inner.iloc[:,1143])
    ax.scatter(inner.loc[I,'mean'],inner.loc[I,'tot_death'], color='r', label=country)
    a, b = np.polyfit(inner.loc[:,'mean'],inner.iloc[:,1143], 1)
    ax.plot(inner.loc[:,'mean'], a*(inner.loc[:,'mean'])+b, color='y', linestyle='-.', label='fitting plot line')
    ax.set_title('covid-19 deaths against hospital beds')
    ax.set_xlabel('hospital beds per 1000 people')
    ax.set_ylabel('covid-19 deaths')
    ax.legend(loc='upper right', prop={'size': 10});

def red_country2(inner, country):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    I = inner['country'] == country
    ax.scatter(inner.loc[:,'mean'],inner.loc[:,'last'])
    ax.scatter(inner.loc[I,'mean'],inner.loc[I,'last'], color = 'r', label=country)
    a, b = np.polyfit(inner.loc[:,'mean'],inner.loc[:,'last'], 1)
    ax.plot(inner.loc[:,'mean'], a*(inner.loc[:,'mean'])+b, color='y', linestyle='-.', label='fitting plot line')
    ax.set_title('density in 2020 against hospital beds')
    ax.set_xlabel('hospital beds per 1000 people')
    ax.set_ylabel('density')
    ax.legend(loc='upper right', prop={'size': 10});