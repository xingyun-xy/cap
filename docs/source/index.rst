ChangAn Perception (CAP)
================================

ChangAn Perception是长安提供基于Pytorch的深度学习训练
框架，由Pytorch plugin和CAP算法包两部分组成。

Pytorch plugin是基于Pytorch开发的一套量化算法工具，利用该工具训练得到的量化模型均可以正常编译和运行
在英伟达GPU上。

CAP算法包是基于Pytorch和Pytorch plugin的接口开发的一套高效且用户友好的AI算法工具。同时它还可以提供包含分类，检测，分割等常见的图像
任务的SOTA(state-of-the-art)深度学习模型。


.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   quick_start/evaluation.md
   quick_start/evaluation_report.md

.. toctree::
   :maxdepth: 1
   :caption: ChangeLog

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/data.rst
   api_reference/callbacks.rst
   api_reference/engine.rst
   api_reference/models.rst
   api_reference/metrics.rst
   api_reference/profiler.rst
   api_reference/visualize.rst

   changelog/CHANGELOG.md


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
