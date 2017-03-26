from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
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
plt.contour(xx, yy, z, [0.5], linewidths=2., colors='k')
plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',cmap=plt.cm.Paired)
plt.axis('tight')
plt.title('Logistic Regression Analysis')
plt.show()

logit_labels_train = logit.predict(x)
print ("logit train accuracy:",format(metrics.accuracy_score(y,logit_labels_train)))

# fisher's linear discriminant & quadratic discriminant analysis
lda = discriminant_analysis.LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
lda_labels_train = lda.fit(x,np.ravel(y)).predict(x)
zz_lda = lda.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
zz_lda = zz_lda.reshape(xx.shape)


splot = plt.subplot(1, 2, 1)
plt.scatter(x[:,0],x[:,1], c=y,edgecolors='k')
plt.contour(xx, yy, zz_lda, [0.5], linewidths=2., colors='k')
plt.legend()
plt.axis('tight')
plt.title('Linear Discriminant Analysis')

qda = discriminant_analysis.QuadraticDiscriminantAnalysis(store_covariances=True)
qda_labels_train = qda.fit(x,y).predict(x)
zz_qda = qda.predict_proba(np.c_[xx.ravel(),yy.ravel()])[:,1]
zz_qda = zz_qda.reshape(xx.shape)
splot = plt.subplot(1, 2, 2)
plt.scatter(x[:,0],x[:,1], c=y,edgecolors='k')
plt.contour(xx, yy, zz_qda, [0.5], linewidths=2., colors='k')
plt.legend()
plt.axis('tight')
plt.title('Quadratic Discriminant Analysis')
plt.show()

print ("lda train accuracy:",format(metrics.accuracy_score(y,lda_labels_train)))
print ("qda train accuracy:",format(metrics.accuracy_score(y,qda_labels_train)))

# svm
svm_linear_kernel = svm.SVC(kernel='linear', C=1).fit(x,y)
svm_labels_train = svm_linear_kernel.predict(x)
zz_svm_linear_kernel = svm_linear_kernel.predict(np.c_[xx.ravel(),yy.ravel()])
zz_svm_linear_kernel = zz_svm_linear_kernel.reshape(xx.shape)

splot = plt.subplot(1, 2, 1)
plt.scatter(x[:,0],x[:,1], c=y,edgecolors='k')
plt.contour(xx, yy, zz_svm_linear_kernel, [0.5], linewidths=2., colors='k')
plt.legend()
plt.axis('tight')
plt.title('SVC with linear kernel')

svm_rbf = svm.SVC(kernel='rbf', C=1).fit(x,y)
rbf_svc_labels_train = svm_rbf.predict(x)
zz_svm_rbf = svm_rbf.predict(np.c_[xx.ravel(),yy.ravel()])
zz_svm_rbf = zz_svm_rbf.reshape(xx.shape)
splot = plt.subplot(1, 2, 2)
plt.scatter(x[:,0],x[:,1], c=y,edgecolors='k')
plt.contour(xx, yy, zz_svm_rbf, [0.5], linewidths=2., colors='k')
plt.legend()
plt.axis('tight')
plt.title('SVC with RBF kernel')
plt.show()

print ("svm linear kernel train accuracy:",format(metrics.accuracy_score(y,svm_labels_train)))
print ("svm rbf train accuracy:",format(metrics.accuracy_score(y,rbf_svc_labels_train)))