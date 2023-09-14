from importlib import import_module

from common import tasks, training_step

from cap.core.task_sampler import TaskSampler

task_names = [t["name"] for t in tasks]
TASK_CONFIGS = [import_module(t) for t in task_names]

task_sampler_configs = {
    T.task_name: dict(sampling_factor=1) for T in TASK_CONFIGS
}

# roi 3d tasks only
if "sparse_3d" in training_step:
    task_sampler_configs = {
        t: v for t, v in task_sampler_configs.items() if "roi_3d" in t
    }

    task_sampler = TaskSampler(
        task_config=task_sampler_configs,
        method="sample_all",
    )

# sample unimportant tasks less frequently
else:
    imp_task_names = [t["name"] for t in tasks if t.get("important", False)]

    task_sampler_configs["chosen_tasks"] = [
        imp_task_names,
        imp_task_names,
        imp_task_names,
        task_names,
    ]

    task_sampler = TaskSampler(
        task_config=task_sampler_configs,
        method="sample_repeat",
    )


loaders = {T.task_name: T.data_loader for T in TASK_CONFIGS}
data_loader = dict(
    type="MultitaskInfLoader",
    loaders=loaders,
    task_sampler=task_sampler,
    return_task=True,
    __build_recursive=False,
)
