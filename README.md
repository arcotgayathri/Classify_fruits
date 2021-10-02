# Classify_fruits
In this competition you are going to implement your first Machine Learning model by name K Nearest Neighbors. You are going to use the fruits_train dataset given for training and test your implementation using fruits_test data set.

Note on Distance metric
Use Euclidean Distance. Either you can implement a function to compute the Euclidean distance or you can use SK learn built-in function. Refer to the following link.

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html

Important Note
You are not supposed to use SK-learn or any other library's K-NN function

Note on Normalization
If you need to normalize the data you can use from sklearn.preprocessing module.

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
/kaggle/input/classify-fruits/sample_submission.csv
/kaggle/input/classify-fruits/fruits_test.csv
/kaggle/input/classify-fruits/fruits_train.csv
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from sklearn.preprocessing import StandardScaler
train_data = pd.read_csv('/kaggle/input/classify-fruits/fruits_train.csv')
test_data = pd.read_csv('/kaggle/input/classify-fruits/fruits_test.csv')
X_train =train_data.iloc[:,0:4].values
X_train
array([[  1. , 160. ,   7.1,   7.6],
       [  2. , 194. ,   7.2,  10.3],
       [  3. , 154. ,   7.2,   7.2],
       [  4. , 154. ,   7. ,   7.1],
       [  5. , 162. ,   7.4,   7.2],
       [  6. , 164. ,   7.2,   7. ],
       [  7. , 154. ,   7.1,   7.5],
       [  8. , 116. ,   6.1,   8.5],
       [  9. , 170. ,   7.6,   7.9],
       [ 10. , 116. ,   5.9,   8.1],
       [ 11. , 144. ,   6.8,   7.4],
       [ 12. , 160. ,   7.5,   7.5],
       [ 13. , 166. ,   6.9,   7.3],
       [ 14. , 142. ,   7.6,   7.8],
       [ 15. , 156. ,   7.4,   7.4],
       [ 16. , 116. ,   6. ,   7.5],
       [ 17. , 356. ,   9.2,   9.2],
       [ 18. , 152. ,   6.5,   8.5],
       [ 19. , 164. ,   7.3,   7.7],
       [ 20. , 162. ,   7.5,   7.1],
       [ 21. , 158. ,   7.1,   7.5],
       [ 22. , 140. ,   7.3,   7.1],
       [ 23. , 186. ,   7.2,   9.2],
       [ 24. , 174. ,   7.3,  10.1],
       [ 25. , 180. ,   8. ,   6.8],
       [ 26. , 168. ,   7.5,   7.6],
       [ 27. , 216. ,   7.3,  10.2],
       [ 28. , 160. ,   7. ,   7.4],
       [ 29. , 172. ,   7.1,   7.6],
       [ 30. , 140. ,   6.7,   7.1],
       [ 31. , 180. ,   7.6,   8.2],
       [ 32. , 362. ,   9.6,   9.2],
       [ 33. , 342. ,   9. ,   9.4],
       [ 34. , 152. ,   7.6,   7.3],
       [ 35. , 200. ,   7.3,  10.5],
       [ 36. , 116. ,   6.3,   7.7],
       [ 37. , 178. ,   7.1,   7.8],
       [ 38. , 192. ,   8.4,   7.3],
       [ 39. , 118. ,   5.9,   8. ],
       [ 40. , 132. ,   5.8,   8.7]])
y_train = train_data.iloc[:,-1].values
y_train
array([2, 3, 2, 1, 1, 2, 2, 3, 1, 3, 2, 1, 1, 2, 1, 3, 2, 3, 1, 1, 2, 1,
       3, 3, 1, 1, 3, 2, 1, 2, 2, 2, 2, 1, 3, 3, 1, 1, 3, 3])
X_test= test_data.iloc[:,:].values
X_test
array([[  1. , 118. ,   6.1,   8.1],
       [  2. , 158. ,   7.2,   7.8],
       [  3. , 120. ,   6. ,   8.4],
       [  4. , 210. ,   7.8,   8. ],
       [  5. , 156. ,   7.6,   7.5],
       [  6. , 176. ,   7.4,   7.2],
       [  7. , 154. ,   7.3,   7.3],
       [  8. , 196. ,   7.3,   9.7],
       [  9. , 130. ,   6. ,   8.2],
       [ 10. , 150. ,   7.1,   7.9],
       [ 11. , 172. ,   7.4,   7. ],
       [ 12. , 156. ,   7.7,   7.1],
       [ 13. , 190. ,   7.5,   8.1],
       [ 14. , 204. ,   7.5,   9.2]])
from sklearn.preprocessing import StandardScaler
standard= StandardScaler()
standard.fit(X_train)
X_train = standard.transform(X_train)
X_test = standard.transform(X_test)
classifier =KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train,y_train)
KNeighborsClassifier(n_neighbors=3)
final = classifier.predict(X_test)
final

 
array([3, 2, 3, 1, 1, 2, 2, 3, 3, 2, 1, 1, 1, 3])
predictions = pd.DataFrame({'ID':test_data['Id'],'Category':np.array(final)})
predictions.to_csv("final_output.csv")
