from sklearn import ensemble
import models

MODELS = {
    "randomforest":
    ensemble.RandomForestClassifier(n_estimators=300, verbose=2, n_jobs=-1),
    "extratrees":
    ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, verbose=2),
    models.XGBoostModel()
}
