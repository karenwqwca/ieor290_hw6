from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn import discriminant_analysis
from sklearn import metrics


x_1 = np.random.multivariate_normal([-2, -2], ([2, 1], [1, 2]), 200)
x_neg_1 = np.random.multivariate_normal([1, 1], ([3, -2], [-2, 3]), 100)

x_1_target = np.ones([200,1])
x_neg_1_target = np.ones([100,1])*(-1)

x = np.vstack((x_1,x_neg_1))
y = np.vstack((x_1_target,x_neg_1_target))

# logistic regression
logit = linear_model.LogisticRegression(C=10)
logit.fit(x,np.ravel(y))

x_min_logit,x_max_logit = x[:,0].min() - 0.5, x[:,0].max() + 0.5
y_min_logit,y_max_logit = x[:,0].min() - 0.5, x[:,0].max() + 0.5

xx, yy = np.meshgrid(np.linspace(x_min_logit,x_max_logit,50),np.linspace(y_min_logit,y_max_logit,50))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

ax = plt.gca()
z = logit.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
z = z.reshape(xx.shape)
cs = ax.contourf(xx,yy,z,cmap='RdBu',alpha=.5)
cs2 = ax.contour(xx,yy,z,cmap='RdBu',alpha=.5)
plt.clabel(cs2,fmt= '%2.1f', colors = 'k', fontsize=14)

plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',cmap=plt.cm.Paired)
plt.show()

logit_labels_train = logit.predict(x)
print ("logit train accuracy:",format(metrics.accuracy_score(y,logit_labels_train)))
# ('logit train accuracy:', '0.936666666667')

# fisher's linear discriminant
lda = discriminant_analysis.LinearDiscriminantAnalysis(store_covariance=True,n_components=2)
lda.fit(x,np.ravel(y))
lda_labels_train = lda.predict(x)
zz_lda = logit.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
zz_lda = zz_lda.reshape(xx.shape)

plt.figure()
splot = plt.subplot(1, 2, 1)
plt.scatter(x[:,0],x[:,1], c=y,edgecolors='k',cmap=plt.cm.Paired)
plt.contour(xx, yy, zz_lda, [0.5], linewidths=2., colors='k')
plt.legend()
plt.axis('tight')
plt.title('Linear Discriminant Analysis')

qda = discriminant_analysis.QuadraticDiscriminantAnalysis()
qda.fit(x,y)
qda_labels_train = qda.predict(x)
zz_qda = logit.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
zz_qda = zz_qda.reshape(xx.shape)
splot = plt.subplot(1, 2, 2)
plt.scatter(x[:,0],x[:,1], c=y,edgecolors='k',cmap=plt.cm.Paired)
plt.contour(xx, yy, zz_qda, [0.5], linewidths=2., colors='k')
plt.legend()
plt.axis('tight')
plt.title('Quadratic Discriminant Analysis')
plt.show()

print ("lda train accuracy:",format(metrics.accuracy_score(y,lda_labels_train)))
print ("qda train accuracy:",format(metrics.accuracy_score(y,qda_labels_train)))
# ('lda train accuracy:', '0.943333333333')
# ('qda train accuracy:', '0.953333333333')