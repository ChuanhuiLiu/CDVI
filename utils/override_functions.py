import numpy as np
import pandas as pd
import warnings
import os
import sys

from sksurv import metrics, util
from pycox.evaluation.concordance import concordance_td
from pycox.evaluation import EvalSurv
from auton_survival.metrics import _concordance_index_ipcw,_cumulative_dynamic_auc, _integrated_brier_score,_brier_score
from sklearn.metrics import auc




current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
def survival_regression_metric_modified(metric, outcomes, predictions,
                               times, outcomes_train=None, 
                               n_bootstrap=None, td_quantile = 0.95,random_seed=0):
  """ Compute metrics to assess survival model performance, forked from auton-survival.metric. Additional metrics are added.

  Parameters
  -----------
  metric: string
      Measure used to assess the survival regression model performance.
      Options include:
      - `brs` : brier score
      - `ibs` : integrated brier score
      - `auc`: cumulative dynamic area under the curve
      - `ctd` : concordance index inverse probability of censoring
                weights (ipcw)
  outcomes : pd.DataFrame (test true survival time and test true event label) [n,2]
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for evaluation data.
  predictions: np.array [n,k]
      A numpy array of survival function or survival time predictions for the samples.
  times: np.array 
      The time points at which to compute metric value(s) or to add necessary additional argument(s)
  outcomes_train : pd.DataFrame
      A pandas dataframe with rows corresponding to individual samples and
      columns 'time' and 'event' for training data.
  n_bootstrap : int, default=None
      The number of bootstrap samples to use.
      If None, bootrapping is not performed.
  size_bootstrap : float, default=1.0
      The fraction of the population to sample for each bootstrap sample.
  random_seed: int, default=0
      Controls the reproducibility random sampling for bootstrapping.
  
  Returns
  -----------
  float: The metric value for the specified metric.

  """

  if isinstance(times, (float,int)):
    times = [times]

  if outcomes_train is None:
    outcomes_train = outcomes
    warnings.warn("You are are evaluating model performance on the \
same data used to estimate the censoring distribution.")

  if max(times)> outcomes_train.time.max() or min(times)<outcomes_train.time.min():
    warnings.warn("Exterpolation. Test times is not within the range of event times.")

  if metric == 'ctd': # original auton_lab's implementation of ctd via sksurv with ipcw, used by authors of Deep Cox mixture/Deep survival Machine, 
    # This implementation results in a simpler time-dependent c index by abusing the argument tau (float, optional; Truncation time).    
    # which is actually a infinite time-dependent c-index with test-dataset truncation at tau which is exactly at test times
    # There is no proper time-dependent c-index implementaion with independent truncations.
    survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train) 
    survival_test = util.Surv.from_dataframe('event', 'time', outcomes) # necessary for broadcasting, making it hard to slice
    if n_bootstrap is None:
      return _concordance_index_ipcw(survival_train, survival_test, predictions, times)
    else:
      return [_concordance_index_ipcw(survival_train, survival_test, predictions, times, random_seed=i) for i in range(n_bootstrap)]
  elif metric == 'ctdipcw':
    # a modified version to include time-dependent feature based on sksurv with ipcw, taking pd.dataframe as input
    if n_bootstrap is None:
      return _concordance_index_ipcw_td(outcomes_train, outcomes, predictions, times,  td_quantile=td_quantile)
    else:
      return [_concordance_index_ipcw_td(outcomes_train, outcomes,predictions, times,  td_quantile=td_quantile, random_seed=i) for i in range(n_bootstrap)]
  elif metric == 'ctdpycox':
    # original time-depenent c-index implementation via pycox.Evasurv by authors of SAVAE. It is actually an infinite-time ctd or not a time-dependent ctd.
    if n_bootstrap is None:
      return _concordance_index_pycox(outcomes_train, outcomes, predictions, times)
    else:
      return [_concordance_index_pycox(outcomes_train, outcomes,predictions, times, random_seed=i) for i in range(n_bootstrap)]
  elif metric == 'brs':
    # delayed tuple transformation from pd.dataframe
    survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train) 
    survival_test = util.Surv.from_dataframe('event', 'time', outcomes) 
    if n_bootstrap is None:
      return _brier_score(survival_train, survival_test, predictions, times)
    else:
      return [_brier_score(survival_train, survival_test, predictions, times, random_seed=i) for i in range(n_bootstrap)]
  elif metric == 'ibs':
    survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train)
    survival_test = util.Surv.from_dataframe('event', 'time', outcomes) 
    if n_bootstrap is None:
      return _integrated_brier_score(survival_train, survival_test, predictions, times)
    else:
      return [_integrated_brier_score(survival_train, survival_test, predictions, times, random_seed=i) for i in range(n_bootstrap)]    
  elif metric == 'auc':
    survival_train = util.Surv.from_dataframe('event', 'time', outcomes_train) 
    survival_test = util.Surv.from_dataframe('event', 'time', outcomes) 
    if n_bootstrap is None:
      return  _cumulative_dynamic_auc(survival_train, survival_test, predictions, times)
    else:
      return [ _cumulative_dynamic_auc(survival_train, survival_test, predictions, times, random_seed=i) for i in range(n_bootstrap)]

  else:
    raise NotImplementedError()

def _concordance_index_ipcw_td(survival_train, survival_test, predictions, times, td_quantile = None,tau=None):
  """Wrapper function to compute time-dependent concordance index with ipcw, forked from sksurv. 
      Some unnecessary validate function such as _check_times are excluded or downgraded from Errors to Warnings.

  # survival_test: pd.dataframe with time and event column for test dataset.
  # predictions: predicted risk (1-survival) function of test dataset [n,n_times].
  # times: the time of which the prediction is computed. 
  # tau: cutoff time for evluating the ctd on survival_test. Any subject who survived longer than times in survival_test will be deleted.
  # td_quantile: the quantile of event time in survival_train, truncating for estimating time-dependent c-index. 
  #               Any subject who survived longer than this quantile and has an event in survival_test will be deleted. 
  # Note that times and td_quantile are not necessarily the same, and td_quantile is processed first.
  """
  cutoff_time = np.inf
  if td_quantile is not None and (int(td_quantile>0)|int(td_quantile<1)):
    # subseting survival_test based on the quantile of survival_train["time"]
    cutoff_time = np.quantile(survival_train.loc[survival_train["event"]==True,"time"],td_quantile)
    if not np.isnan(cutoff_time) and min(survival_test["time"])< cutoff_time:
      td_predictions = predictions[survival_test["time"]< cutoff_time,:]
      td_survival_test= survival_test[survival_test["time"]< cutoff_time]
    else:
      raise RuntimeError("Truncation error and potential extrapolation warning, cutoff time is "+str(cutoff_time)+" max test survival time is "+str(max(survival_test["time"])))
    if td_survival_test.empty or np.sum(td_survival_test["event"])==0:
      raise RuntimeError("time dependent index can't be computed, no event observed in the test dataset")
  else:
    cutoff_time = np.inf
    td_predictions = predictions
    td_survival_test = survival_test
  # avoid operations on tuple
  #print(td_survival_test)
  survival_train = Surv.from_dataframe('event', 'time', survival_train) # use modifed Surv below instead of sksurv.utils.Surv, allowing 
  survival_test = Surv.from_dataframe('event', 'time', td_survival_test) 

  # find the index of the closest time that is smaller than cutoff_time
  closest_ind = None
  min_diff = float('inf')

  for i in np.arange(len(times)):
    num = times[i]
    if num < cutoff_time and cutoff_time - num < min_diff:
      closest_ind = i
      min_diff = cutoff_time - num

  ctd = ctd_ipcw(survival_train, survival_test,1-td_predictions[:,closest_ind],tau=None)[0] # no longer have the same trucations times with prediction time for each column
  return ctd

def _concordance_index_pycox(survival_train, survival_test, predictions, times, random_seed=None):
  # Import from Pycox metric.evaluation; survival_train is not used since there is no ipcw.
  predictions= pd.DataFrame(predictions.T, index=times)
  d= EvalSurv(surv=predictions,durations= np.array(survival_test['time']).flatten(), events = np.array(survival_test['event']).flatten(),censor_surv="km")
  ctd= d.concordance_td() 
  return ctd

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
def plot_performance_metrics(results, times):
  """Plot Brier Score, ROC-AUC, and time-dependent concordance index, forked from Demo of Auton Suvival
  for survival model evaluation.

  Parameters
  -----------
  results : dict
      Python dict with key as the evaulation metric
  times : float or list
      A float or list of the times at which to compute
      the survival probability.

  Returns
  -----------
  matplotlib subplots

  """

  colors = ['blue', 'purple', 'orange', 'green']
  gs = gridspec.GridSpec(1, len(results), wspace=0.3)

  for fi, result in enumerate(results.keys()):
    val = results[result]
    x = [str(round(t, 1)) for t in times]
    ax = plt.subplot(gs[0, fi]) # row 0, col 0
    ax.set_xlabel('Time')
    ax.set_ylabel(result)
    ax.set_ylim(0, 1)
    ax.bar(x, val, color=colors[fi])
    plt.xticks(rotation=30)
  plt.show()

# Modifying sksurv.utils.Surv.
import numpy as np
import pandas as pd
from sklearn.utils import check_consistent_length

class Surv:
    """
    Helper class to construct structured array of event indicator and observed time.
    """

    @staticmethod
    def from_arrays(event, time, name_event=None, name_time=None):
        """Create structured array.

        Parameters
        ----------
        event : array-like
            Event indicator. A boolean array or array with values 0/1.
        time : array-like
            Observed time.
        name_event : str|None
            Name of event, optional, default: 'event'
        name_time : str|None
            Name of observed time, optional, default: 'time'

        Returns
        -------
        y : np.array
            Structured array with two fields.
        """
        name_event = name_event or "event"
        name_time = name_time or "time"
        if name_time == name_event:
            raise ValueError("name_time must be different from name_event")

        time = np.asanyarray(time, dtype=float)
        y = np.empty(time.shape[0], dtype=[(name_event, bool), (name_time, float)])
        y[name_time] = time

        event = np.asanyarray(event)
        check_consistent_length(time, event)

        if np.issubdtype(event.dtype, np.bool_):
            y[name_event] = event
        else:
            events = np.unique(event)
            events.sort()
            if len(events) > 2:          # modified !=2 because it can be all events and all censoring
                raise ValueError("event indicator must be binary")

            if np.all(events == np.array([0, 1], dtype=events.dtype)):
                y[name_event] = event.astype(bool)
            else:
                raise ValueError("non-boolean event indicator must contain 0 and 1 only")

        return y

    @staticmethod
    def from_dataframe(event, time, data):
        """Create structured array from data frame.

        Parameters
        ----------
        event : object
            Identifier of column containing event indicator.
        time : object
            Identifier of column containing time.
        data : pandas.DataFrame
            Dataset.

        Returns
        -------
        y : np.array
            Structured array with two fields.
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"expected pandas.DataFrame, but got {type(data)!r}")

        return Surv.from_arrays(
            data.loc[:, event].values, data.loc[:, time].values, name_event=str(event), name_time=str(time)
        )

# from .exceptions import NoComparablePairException
# from .nonparametric import CensoringDistributionEstimator, SurvivalFunctionEstimator
# from .util import check_y_survival

# def ctd_ipcw(survival_train, survival_test, estimate, tau=None, tied_tol=1e-8):
#     """Concordance index for right-censored data based on inverse probability of censoring weights, forked from sksurv.metrics.concordance_index_ipcw

#     This is an independent implementation of ctd with ipcw and proper time truncations. 
#     The original implementation in DSM/DCM is using argument tau (float, optional; Truncation time) for a simpler time-dependent c index.   

#     It is based on inverse probability of censoring weights, thus requires
#     access to survival times from the training data to estimate the censoring
#     distribution. Note that this requires that survival times `survival_test`
#     lie within the range of survival times `survival_train`. This can be
#     achieved by specifying the truncation time `tau`.
#     The resulting `cindex` tells how well the given prediction model works in
#     predicting events that occur in the time range from 0 to `tau`.

#     The estimator uses the Kaplan-Meier estimator to estimate the
#     censoring survivor function. Therefore, it is restricted to
#     situations where the random censoring assumption holds and
#     censoring is independent of the features.

#     See the :ref:`User Guide </user_guide/evaluating-survival-models.ipynb>`
#     and [1]_ for further description.

#     Parameters
#     ----------
#     survival_train : structured array, shape = (n_train_samples,)
#         Survival times for training data to estimate the censoring
#         distribution from.
#         A structured array containing the binary event indicator
#         as first field, and time of event or time of censoring as
#         second field.

#     survival_test : structured array, shape = (n_samples,)
#         Survival times of test data.
#         A structured array containing the binary event indicator
#         as first field, and time of event or time of censoring as
#         second field.

#     estimate : array-like, shape = (n_samples,)
#         Estimated risk of experiencing an event of test data.

#     tau : float, optional
#         Truncation time. The survival function for the underlying
#         censoring time distribution :math:`D` needs to be positive
#         at `tau`, i.e., `tau` should be chosen such that the
#         probability of being censored after time `tau` is non-zero:
#         :math:`P(D > \\tau) > 0`. If `None`, no truncation is performed.

#     tied_tol : float, optional, default: 1e-8
#         The tolerance value for considering ties.
#         If the absolute difference between risk scores is smaller
#         or equal than `tied_tol`, risk scores are considered tied.

#     Returns
#     -------
#     cindex : float
#         Concordance index

#     concordant : int
#         Number of concordant pairs

#     discordant : int
#         Number of discordant pairs

#     tied_risk : int
#         Number of pairs having tied estimated risks

#     tied_time : int
#         Number of comparable pairs sharing the same time

#     See also
#     --------
#     concordance_index_censored
#         Simpler estimator of the concordance index.

#     as_concordance_index_ipcw_scorer
#         Wrapper class that uses :func:`concordance_index_ipcw`
#         in its ``score`` method instead of the default
#         :func:`concordance_index_censored`.

#     References
#     ----------
#     .. [1] Uno, H., Cai, T., Pencina, M. J., D’Agostino, R. B., & Wei, L. J. (2011).
#            "On the C-statistics for evaluating overall adequacy of risk prediction
#            procedures with censored survival data".
#            Statistics in Medicine, 30(10), 1105–1117.
#     """
#     test_event, test_time = check_y_survival(survival_test)

#     if tau is not None:
#         mask = test_time < tau
#         survival_test = survival_test[mask]

#     estimate = _check_estimate_1d(estimate, test_time)

#     cens = CensoringDistributionEstimator()
#     cens.fit(survival_train)
#     ipcw_test = cens.predict_ipcw(survival_test)
#     if tau is None:
#         ipcw = ipcw_test
#     else:
#         ipcw = np.empty(estimate.shape[0], dtype=ipcw_test.dtype)
#         ipcw[mask] = ipcw_test
#         ipcw[~mask] = 0

#     w = np.square(ipcw)

#     return _estimate_concordance_index(test_event, test_time, estimate, w, tied_tol)