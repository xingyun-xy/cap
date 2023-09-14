cap.callbacks
=============

Callbacks widely used while training in CAP.

Callbacks
---------

.. py:currentmodule:: cap.callbacks

.. autosummary::
    :nosignatures:

    CallbackMixin

    AdasEval

    Checkpoint

    CosLrUpdater

    PolyLrUpdater

    StepDecayLrUpdater

    NoamLrUpdater

    SaveTraced

    MetricUpdater

    StatsMonitor

    FreezeModule

    FuseBN

    TensorBoard

    Validation

    ExponentialMovingAverage

    GradScale

    CompactorUpdater

    ModelTracking

    CAPEval

    CAPEvalTaskType

save_eval_results
^^^^^^^^^^^^^^^^^

.. py:currentmodule:: cap.callbacks.save_eval_results

.. autosummary::
    :nosignatures:

    SaveDet2dResult

    SaveDetConsistencyResult

    SaveSegConsistencyResult

    SaveEvalResult

task_visualize
^^^^^^^^^^^^^^

.. py:currentmodule:: cap.callbacks.task_visualize

.. autosummary::
    :nosignatures:

    BaseVisualize

    ComposeVisualize

    Det2dVisualize

    DetMultitaskVisualize

    InputsVisualize

API Reference
--------------

.. automodule:: cap.callbacks
    :members:

.. automodule:: cap.callbacks.save_eval_results
    :members:

.. automodule:: cap.callbacks.task_visualize
    :members:

