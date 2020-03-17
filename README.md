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

data = numpy.array([1, 1, 2, 2, 2.5, 3, 3, 4.5, 6, 8, 10, 20, 30, 80, 0, numpy.nan])
data = autoshiftedlog(data)
```
