#%% [markdown]
# # PR HW2
# ## 1. load data

# %%
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('wine.data', header=None)
train_size = int(df.shape[0] * 0.5)
test_size = df.shape[0] - train_size
data = df.values
np.random.shuffle(data)
df = pd.DataFrame(data)

y_train = df.iloc[:train_size,0].values
X_train = df.iloc[:train_size,1:].values

y_test = df.iloc[train_size:,0].values
X_test = df.iloc[train_size:,1:].values

#%% [markdown]
# ## 2. group by label(y)

# %%
X_class1 = []
X_class2 = []
X_class3 = []

for i in range(X_train.shape[0]):
    X_temp = X_train[i,:]
    if y_train[i] == 1:
        X_class1.append(X_temp)
    if y_train[i] == 2:
        X_class2.append(X_temp)
    if y_train[i] == 3:
        X_class3.append(X_temp)

X_class1 = np.array(X_class1)
X_class2 = np.array(X_class2)
X_class3 = np.array(X_class3)

#%% [markdown]
# ## 3. Covariance Matrix & Mean Vector

# %%
X_class1_mean = np.mean(X_class1, axis=0)
X_class1_cov = X_class1.T
X_class1_cov = np.cov(X_class1_cov)

X_class2_mean = np.mean(X_class2, axis=0)
X_class2_cov = X_class2.T
X_class2_cov = np.cov(X_class2_cov)

X_class3_mean = np.mean(X_class3, axis=0)
X_class3_cov = X_class3.T
X_class3_cov = np.cov(X_class3_cov)

prior_class1 = X_class1.shape[0] / X_train.shape[0]
prior_class2 = X_class2.shape[0] / X_train.shape[0]
prior_class3 = X_class3.shape[0] / X_train.shape[0]

#%% [markdown]
# ## 4. upper bound

#%%
def up_bound(prior1, prior2, mean1, mean2, cov1, cov2):
    a = np.sqrt(prior1*prior2)
    b = (1/8)*(mean2-mean1).reshape((1, -1))
    c = np.linalg.inv((cov1+cov2)/2)
    d = (mean2-mean1).reshape((-1, 1))
    e = np.linalg.det((cov1+cov2)/2)
    f = np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    g = (1/2)*np.log(e/f)

    answer = a*np.exp(-1*(np.dot(np.dot(b, c), d) + g))
    return answer

up_c1_c2 = up_bound(prior_class1, prior_class2, X_class1_mean, X_class2_mean, X_class1_cov, X_class2_cov)
up_c2_c3 = up_bound(prior_class2, prior_class3, X_class2_mean, X_class3_mean, X_class2_cov, X_class3_cov)
up_c3_c1 = up_bound(prior_class3, prior_class1, X_class3_mean, X_class1_mean, X_class3_cov, X_class1_cov)

print("class1 and class2 upper bound: ", up_c1_c2[0,0])
print("class2 and class3 upper bound: ", up_c2_c3[0,0])
print("class3 and class1 upper bound: ", up_c3_c1[0,0])

# answer = np.sqrt(prior_class1*prior_class2)
# a = (1/8)*(X_class2_mean-X_class1_mean).reshape((1, -1))
# b = np.linalg.inv((X_class1_cov+X_class2_cov)/2)
# c = (X_class2_mean-X_class1_mean).reshape((-1, 1))
# d = np.linalg.det((X_class1_cov+X_class2_cov)/2)
# e = np.sqrt(np.linalg.det(X_class1_cov) * np.linalg.det(X_class1_cov))
# f = (1/2)*np.log(d/e)

# answer = answer * np.exp(-1*(np.dot(np.dot(a, b), c) + f))
# answer


#%% [markdown]
# ## 3. calculate mean, standard deviateion, prior

# %%
# mean_class1 = np.mean(X_class1, axis=0)
# mean_class2 = np.mean(X_class2, axis=0)
# mean_class3 = np.mean(X_class3, axis=0)

# std_class1 = np.std(X_class1, axis=0)
# std_class2 = np.std(X_class2, axis=0)
# std_class3 = np.std(X_class3, axis=0)

# prior_class1 = X_class1.shape[0] / X_train.shape[0]
# prior_class2 = X_class2.shape[0] / X_train.shape[0]
# prior_class3 = X_class3.shape[0] / X_train.shape[0]


# #%% [markdown]
# # ## 4. calculate posterior

# # %%
# def likelyhood(x, mean, sigma):
#     return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))

# def posterior(x, mean, std, prior):
#     return np.prod(likelyhood(x, mean, std))*prior

# result_list = []
# for item in X_test:
#     p1 = posterior(item, mean_class1, std_class1, prior_class1)
#     p2 = posterior(item, mean_class2, std_class2, prior_class2)
#     p3 = posterior(item, mean_class3, std_class3, prior_class3)

#     if p1 > p2 and p1 > p3:
#         result_list.append(1)
#     elif p2 > p1 and p2 > p3:
#         result_list.append(2)
#     else:
#         result_list.append(3)


# # %%
# count = 0
# for label, predict in zip(y_test, result_list):
#     # print(label, predict)
#     if label == predict:
#         count += 1

# accuracy = count / y_train.shape[0]
# print("accuracy: ", accuracy)

# %%
# class0_data, class1_data, class2_data = [], [], []


# for i in range(data.shape[0]):
#     if data.loc[i, 'target'] == 0:
#         class0_data.append(data.loc[i, data.columns[:-1].values])
#     elif data.loc[i, 'target'] == 1:
#         class1_data.append(data.loc[i, data.columns[:-1].values])
#     elif data.loc[i, 'target'] == 2:
#         class2_data.append(data.loc[i, data.columns[:-1].values])

# class0_data = pd.DataFrame(class0_data)
# class1_data = pd.DataFrame(class1_data)
# class2_data = pd.DataFrame(class2_data)


# %%
# data.describe()
# class0_data.describe()
# class1_data.describe()
# class2_data.describe()

# class0_mean = class0_data.mean()
# class0_std = class0_data.std()

# class1_mean = class1_data.mean()
# class1_std = class1_data.std()

# class2_mean = class2_data.mean()
# class2_std = class2_data.std()


# %%
# def likelyhood(x, mean, sigma):
#     return np.exp(-(x-mean)**2/(2*sigma**2))*(1/(np.sqrt(2*np.pi)*sigma))

# class0_prior = class0_data.shape[0]/data.shape[0]
# class1_prior = class1_data.shape[0]/data.shape[0]
# class2_prior = class2_data.shape[0]/data.shape[0]
# print(class0_prior+class1_prior+class2_prior)
# print(data.values[0][:-1])
# print(class0_mean.values)
# product1 = np.prod(likelyhood(data.values[0][:-1], class0_mean.values, class0_std.values))*class0_prior
# product2 = np.prod(likelyhood(data.values[0][:-1], class1_mean.values, class1_std.values))*class1_prior
# product3 = np.prod(likelyhood(data.values[0][:-1], class2_mean.values, class2_std.values))*class2_prior

# def posterior(x, mean, std, prior):
#     product = np.prod(likelyhood(x, mean, std))*prior
    
#     return product

# t0 = posterior(data.values[0][:-1],class0_mean, class0_std, class0_prior)
# t1 = posterior(data.values[0][:-1],class1_mean, class1_std, class1_prior)
# t2 = posterior(data.values[0][:-1],class2_mean, class2_std, class2_prior)
# t_all = posterior(data.values[:][:-1],class0_mean, class0_std, class0_prior)

# # %%
# sns.distplot(data['alcohol'],kde=0)

# # %%
# for i in data.target.unique():
#     sns.distplot(data['alcohol'][data.target==i], kde=1, label='{}'.format(i))
#     # kde 估計真實分佈函式

# plt.legend() #　show explain


# # %%
# import matplotlib.gridspec as gridspec
# for feature in raw_data['feature_names']:
#     # print(feature)
#     # sns.boxplot(data=data,x=data.target,y=data[feature])
#     gs1 = gridspec.GridSpec(3,1)
#     ax1 = plt.subplot(gs1[:-1])
#     ax2 = plt.subplot(gs1[-1])
#     gs1.update(right=0.60)
#     sns.boxplot(x=feature,y='class',data=data,ax=ax2)
#     sns.kdeplot(data[feature][data.target==0],ax=ax1,label='0')
#     sns.kdeplot(data[feature][data.target==1],ax=ax1,label='1')
#     sns.kdeplot(data[feature][data.target==2],ax=ax1,label='2')
#     ax2.yaxis.label.set_visible(False)
#     ax1.xaxis.set_visible(False)
#     plt.show()

# %%


