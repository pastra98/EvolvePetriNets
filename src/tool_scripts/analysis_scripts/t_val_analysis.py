# %%
from scipy import stats
import pandas as pd
has_comp = [1,0,0,1,1,1,0,1,0,0,]

fitness = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,]


df = pd.DataFrame({'has_comp': has_comp, 'fitness': fitness})
df
# %%
with_c = df[df['has_comp'] == 1]['fitness']
without_c = df[df['has_comp'] == 0]['fitness']

print(f"with c mean: {with_c.mean()},\nwithout: {without_c.mean()}")

# %%
# from the video
def vid_student_t_val(s1: pd.Series, s2: pd.Series):
    nom = s1.mean() - s2.mean()
    df1, df2 = len(s1)-1, len(s2)-1 # deg freedom always same in my ga
    pooled_var = (s1.var()*df1 + s2.var()*df2) / (df1+df2)
    denom = (pooled_var/len(s1) + pooled_var/len(s2))**0.5
    return nom/denom

# from my slides
def slide_student_t_val(s1: pd.Series, s2: pd.Series):
    nom = s1.mean() - s2.mean()
    df1, df2 = len(s1)-1, len(s2)-1 # deg freedom always same in my ga
    denom = (((len(s1)+len(s2))/
              (len(s1)*len(s2)))
              *
              ((df1*(s1.var()**2) + df2*(s2.var()**2))/
                df1+df2)
            )**0.5
    return nom/denom

# from investopedia
def investo_student_t_val(s1: pd.Series, s2: pd.Series):
    nom = s1.mean() - s2.mean()
    df1, df2 = len(s1)-1, len(s2)-1 # deg freedom always same in my ga
    pooled_var = (s1.var()*df1 + s2.var()*df2) / (df1+df2)
    denom = (pooled_var * (1/len(s1) + 1/len(s2)))**0.5
    return nom/denom

# from chatgpt
def chatgpt_student_t_val(s1: pd.Series, s2: pd.Series):
    # Calculate the means of the two samples
    mean1 = s1.mean()
    mean2 = s2.mean()
    
    # Calculate the standard deviations of the two samples
    std1 = s1.std()
    std2 = s2.std()
    
    # Calculate the sample sizes
    n1 = len(s1)
    n2 = len(s2)
    
    # Calculate the pooled standard deviation
    pooled_std = (((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))**0.5
    
    # Calculate the standard error of the difference between the means
    se = pooled_std * ((1/n1 + 1/n2)**0.5)
    
    # Calculate the t-statistic
    t_stat = (mean1 - mean2) / se
    
    return t_stat


print('vid:', vid_student_t_val(with_c, without_c))
print('slide:', slide_student_t_val(with_c, without_c))
print('investo:', investo_student_t_val(with_c, without_c))
print('chatgpt:', chatgpt_student_t_val(with_c, without_c))
print('scipy:', stats.ttest_ind(with_c, without_c, equal_var=True)[0])

# %%
with_c.var()**0.5
# with_c.std()

# %%
import numpy as np

# np 
a = np.array([1,2,3,4,5])
a-1