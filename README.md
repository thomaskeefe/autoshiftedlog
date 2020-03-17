# Auto Shifted Log

The family of shifted log transformations helps correct nonnormality and skewness.
The [Automatic Shifted Log](https://arxiv.org/pdf/1601.01986.pdf) transformation of Feng, Hannig, and Marron automatically
selects the shift parameter that minimizes either
* the Anderson-Darling statistic
(to correct nonnormality)
* the sample skewness (to correct skewness).

### Usage

```python
from autoshiftedlog import autoshiftedlog

data = np.array([2, 5, 23, 5, np.nan])
data = autoshiftedlog(data)
```
