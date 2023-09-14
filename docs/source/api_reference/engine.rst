cap.engine
==========

Engine of the main train loop in CAP.

Engine
------

.. py:currentmodule:: cap.engine

.. autosummary::
    :nosignatures:

    build_launcher

    LoopBase

    Predictor

    Calibrator

    Trainer

    ApexDistributedDataParallelTrainer

    DistributedDataParallelTrainer

    DataParallelTrainer

processors
^^^^^^^^^^

.. py:currentmodule:: cap.engine.processors

.. autosummary::
    :nosignatures:

    BatchProcessorMixin

    BasicBatchProcessor

    MultiBatchProcessor

    collect_loss_by_index

    collect_loss_by_regex

API Reference
--------------

.. automodule:: cap.engine
    :members:

.. automodule:: cap.engine.processors
    :members:

