import logging

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt

def print_evaluations(Y_true, Y_pred, classification=True):
    
    if classification:
        report = classification_report(Y_true, Y_pred)
        logging.info('Classification report:\n%s' % str(report))

        cm = confusion_matrix(Y_true, Y_pred)
        logging.info('Confusion Matrix:\n%s' % str(cm))
    
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # cax = ax.matshow(cm)
        # fig.colorbar(cax)
        #
        # ax.set_xticklabels(['']+['-1', '0', '1'])
        # ax.set_yticklabels(['']+['-1', '0', '1'])
        #
        # plt.title('Confusion Matrix')
        # plt.ylabel('True label')
        # plt.xlabel('Predicted label')
        # plt.show(block=False)

    else:
        var = explained_variance_score(Y_true, Y_pred)
        logging.info('Explained variance (best=1.0): %f' % var)
        
        mae = mean_absolute_error(Y_true, Y_pred)
        logging.info('Mean absolute error (best=0.0): %f' % mae)
        
        mse = mean_squared_error(Y_true, Y_pred)
        logging.info('Mean squared error (best=0.0): %f' % mse)
        
        r2 = r2_score(Y_true, Y_pred)
        logging.info('R squared score (best=1.0): %f' % r2)
